"""Write the text record of every clean SMP video with Gemini (design V4).

The record replaces the video in later analysis, so it captures every channel:
overall attributes, a timeline of segments, the verbatim transcript, every
on-screen text, and the audio. Design V4 was chosen in
smp/research/video_extraction/; this version adds one fix: usernames and handles
in on-screen text are written as [handle] (the prompt asks for it, and a regex
replaces any that slip through).

Gemini reads the .mp4 directly (frames and audio). It sees only the video: no
caption, account or popularity.

Checkpointing (built so the run never has to be repeated):
  - Each finished video is appended to data/clean/video_content.jsonl and
    flushed to disk at once; an interrupted run loses nothing finished.
  - Re-running skips every post already in the file and only fills gaps.
  - Each row stores the raw model output, the model, a hash of the prompt and
    schema, and token usage. The run refuses to add rows made with a different
    prompt or model, so the file stays consistent.
  - Calls that still fail after all retries go to video_content_failures.jsonl
    and are retried on the next run.
  - A few videos are blocked by Gemini's recitation filter (finish reason
    RECITATION), typically because a verbatim transcript would reproduce song
    lyrics. Those are not retried in the main run; `--no-lyrics --only-blocked`
    redoes them with one extra rule (describe songs, do not transcribe lyrics).
    They are saved with their own prompt version and `variant = "no_lyrics"`.
  - When every video is done, data/clean/video_content.parquet is written
    (one row per post, the record as JSON text plus a flat text rendering).

Run:
    python smp/pipeline/describe_videos.py --dry-run     # counts and estimate, no calls
    python smp/pipeline/describe_videos.py               # describe what is missing
    python smp/pipeline/describe_videos.py --no-lyrics --only-blocked   # recitation-blocked ones
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parents[1]  # smp/
REPO_ROOT = HERE.parents[1]  # the library repo
CLEAN = HERE / "data" / "clean"
VIDEOS = CLEAN / "videos"
OUT = CLEAN / "video_content.jsonl"
FAILURES = CLEAN / "video_content_failures.jsonl"
FINAL = CLEAN / "video_content.parquet"

MODEL = "gemini-3.5-flash"
INLINE_LIMIT = 18 * 1024 * 1024  # request limit is 20 MB; leave room for the prompt
CALL_TIMEOUT = 300  # seconds; a stuck request is retried instead of hanging
HANDLE = re.compile(r"@[\w.]{2,}")


RULES = """\
Rules:
- Describe only what is actually seen or heard. If something is unclear, write
  "unclear" rather than guessing.
- Be neutral and factual. Do not judge quality, creativity or appeal, and do not
  say or imply how popular, trending, viral, famous or well known the video, its
  sound, its format or anyone in it is.
- Do not infer race, ethnicity, nationality, religion, sexual orientation or
  health. Describe people by visible, neutral attributes only: apparent age band
  (child, teen, adult, older adult), clothing, hair, actions.
- Do not name real people unless the name is shown on screen or spoken.
- Write any username or @handle, wherever it appears, as [handle]. Do not
  transcribe watermarks, or view, like or follower numbers shown on screen.
- Quote speech and on-screen text in the original wording.
- Times are seconds from the start of the video."""

CONTENT_TYPES = (
    "tutorial or how-to; recipe or food preparation; product review or demo; "
    "unboxing or haul; comedy skit; prank; lip-sync or dance to audio; music "
    "performance; vlog or day in the life; story time or talking head; reaction; "
    "educational or facts; beauty, fashion or styling; fitness; pets or animals; "
    "sports; gaming; art, craft or DIY; travel or scenery; ASMR; slideshow or "
    "photo montage; other"
)

PROMPT = f"""\
You are writing the complete text record of a short-form social media video.
The record will replace the video in a research dataset: later analysis will see
only this text, never the video. Capture everything a viewer would perceive.

Watch and listen to the whole video, then fill the JSON schema provided:
overall attributes, a timeline of consecutive segments, the full spoken
transcript, every piece of on-screen text, and the audio.

{RULES}
- The transcript and on-screen text must be complete and verbatim, not
  summarised. If speech is too unclear to transcribe, write "unclear" for that part."""


class Segment(BaseModel):
    start: float = Field(description="Start time in seconds.")
    end: float = Field(description="End time in seconds.")
    visual: str = Field(description="What is shown: people, actions, objects, setting, camera.")
    speech: str = Field(description="Words spoken in this segment, verbatim, with who says them; 'none' if none.")
    on_screen_text: str = Field(description="Text shown in this segment, verbatim; 'none' if none.")
    audio: str = Field(description="Music and sounds in this segment; 'none' if silent.")


class Utterance(BaseModel):
    time: float = Field(description="Approximate start time in seconds.")
    speaker: str = Field(description="Who speaks, e.g. 'person on camera', 'voice-over', 'second person', 'audio track lip-synced'.")
    text: str = Field(description="The words, verbatim.")


class TextOverlay(BaseModel):
    time: float = Field(description="Approximate time it first appears, in seconds.")
    text: str = Field(description="The text, verbatim.")


class VideoRecord(BaseModel):
    summary: str = Field(description="2-3 sentences: what happens, start to finish.")
    content_type: str = Field(description=f"The single best fit from: {CONTENT_TYPES}.")
    topic: str = Field(description="The subject in a few words.")
    format: str = Field(description="How it is built, e.g. setup then punchline, step-by-step list, before/after reveal, single continuous shot, montage, POV.")
    hook: str = Field(description="Exactly what is seen and heard in the first 3 seconds.")
    people: str = Field(description="Each person who appears: neutral visible description, role, whether they speak to camera. 'none' if no people.")
    setting: str = Field(description="The specific place(s) when visible.")
    filming: str = Field(description="Camera and production: handheld selfie, tripod, screen recording, professional camera, green screen, stock footage; framing; lighting.")
    editing: str = Field(description="Number of cuts, pace, transitions, effects, filters, captions style.")
    segments: List[Segment] = Field(description="Consecutive segments covering the whole video.")
    transcript: List[Utterance] = Field(description="Every spoken line, verbatim, in order; empty list if no speech.")
    on_screen_text: List[TextOverlay] = Field(description="Every text overlay, caption or sign, verbatim, in order; empty list if none.")
    audio: str = Field(description="Music (vocals or instrumental, genre, tempo, mood), sound effects, ambient sound, silence; name a song only if its title is shown or sung clearly.")
    tone: str = Field(description="Emotional tone.")
    call_to_action: str = Field(description="Explicit requests to follow, like, comment, share, buy or visit a link, quoted, or 'none'.")
    products_or_brands: str = Field(description="Products, brands or businesses clearly shown or mentioned, or 'none'.")


NO_LYRICS_RULE = """
- Do not transcribe song lyrics or other recorded music. When a song plays or is
  lip-synced, describe it instead (e.g. "upbeat pop song with female vocals about
  a breakup") and record it in the transcript as one line with speaker
  'audio track' and text '[song lyrics not transcribed]'."""
PROMPT_NO_LYRICS = PROMPT + NO_LYRICS_RULE


def prompt_version(prompt: str) -> str:
    blob = MODEL + prompt + json.dumps(VideoRecord.model_json_schema(), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


PROMPT_VERSION = prompt_version(PROMPT)
PROMPT_VERSION_NO_LYRICS = prompt_version(PROMPT_NO_LYRICS)
VERSIONS = {PROMPT_VERSION: "main", PROMPT_VERSION_NO_LYRICS: "no_lyrics"}


class Blocked(Exception):
    """Gemini returned no text because of a content filter; retrying will not help."""


# ------------------------------------------------------------------ helpers

def load_env() -> None:
    """Export the library repo's .env (API keys) without overriding the shell."""
    env = REPO_ROOT / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a line cut off by a crash mid-write; that post is redone
    return rows


def append(path: Path, row: dict) -> None:
    """Append one row and force it to disk, so a crash cannot lose it."""
    with path.open("a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def scrub_handles(obj):
    if isinstance(obj, str):
        return HANDLE.sub("[handle]", obj)
    if isinstance(obj, list):
        return [scrub_handles(x) for x in obj]
    if isinstance(obj, dict):
        return {k: scrub_handles(v) for k, v in obj.items()}
    return obj


def flat_text(rec: dict) -> str:
    """A readable plain-text rendering of one record, for text-only analysis."""
    lines = [
        f"Summary: {rec['summary']}",
        f"Content type: {rec['content_type']}",
        f"Topic: {rec['topic']}",
        f"Format: {rec['format']}",
        f"Hook (first 3 s): {rec['hook']}",
        f"People: {rec['people']}",
        f"Setting: {rec['setting']}",
        f"Filming: {rec['filming']}",
        f"Editing: {rec['editing']}",
        f"Audio: {rec['audio']}",
        f"Tone: {rec['tone']}",
        f"Call to action: {rec['call_to_action']}",
        f"Products or brands: {rec['products_or_brands']}",
        "Timeline:",
        *[f"  {s['start']:.0f}-{s['end']:.0f}s: {s['visual']} | speech: {s['speech']} | "
          f"text: {s['on_screen_text']} | audio: {s['audio']}" for s in rec["segments"]],
        "Transcript:" if rec["transcript"] else "Transcript: none",
        *[f"  {u['time']:.0f}s {u['speaker']}: {u['text']}" for u in rec["transcript"]],
        "On-screen text:" if rec["on_screen_text"] else "On-screen text: none",
        *[f"  {t['time']:.0f}s: {t['text']}" for t in rec["on_screen_text"]],
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------- calls

async def video_part(client: genai.Client, path: Path):
    """Inline small files; upload large ones. Returns (part, uploaded file name)."""
    if path.stat().st_size <= INLINE_LIMIT:
        return types.Part.from_bytes(data=path.read_bytes(), mime_type="video/mp4"), None
    f = await client.aio.files.upload(file=str(path), config=types.UploadFileConfig(mime_type="video/mp4"))
    while f.state and f.state.name == "PROCESSING":
        await asyncio.sleep(3)
        f = await client.aio.files.get(name=f.name)
    if f.state and f.state.name != "ACTIVE":
        raise RuntimeError(f"upload ended in state {f.state.name}")
    return types.Part.from_uri(file_uri=f.uri, mime_type="video/mp4"), f.name


async def describe(client, row, args, lock, stats) -> None:
    prompt = PROMPT_NO_LYRICS if args.no_lyrics else PROMPT
    version = prompt_version(prompt)
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=VideoRecord,
        temperature=0.2,
        max_output_tokens=16384,
    )
    last_error = ""
    for attempt in range(1, args.retries + 1):
        uploaded = None
        try:
            part, uploaded = await video_part(client, VIDEOS / row.video_path)
            t0 = time.time()
            resp = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=MODEL,
                    contents=[types.Content(role="user", parts=[part, types.Part.from_text(text=prompt)])],
                    config=config,
                ),
                timeout=CALL_TIMEOUT,
            )
            if resp.text is None:
                cand = resp.candidates[0] if resp.candidates else None
                reason = getattr(getattr(cand, "finish_reason", None), "name", "UNKNOWN")
                if reason in ("RECITATION", "SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"):
                    raise Blocked(reason)
                raise ValueError(f"empty response, finish reason {reason}")
            record = scrub_handles(VideoRecord.model_validate_json(resp.text).model_dump())
            usage = resp.usage_metadata
            async with lock:
                append(OUT, {
                    "pid": row.pid,
                    "vid": row.vid,
                    "model": MODEL,
                    "prompt_version": version,
                    "variant": VERSIONS[version],
                    "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "record": record,
                    "raw": resp.text,
                    "in_tokens": getattr(usage, "prompt_token_count", None),
                    "out_tokens": getattr(usage, "candidates_token_count", None),
                    "seconds": round(time.time() - t0, 1),
                    "attempts": attempt,
                })
                stats["ok"] += 1
            return
        except Blocked as e:
            last_error = f"Blocked: {e}"
            break
        except Exception as e:  # overload, timeout, network, malformed or cut-off JSON
            last_error = f"{type(e).__name__}: {str(e)[:200]}"
            if attempt < args.retries:
                await asyncio.sleep(min(args.max_wait, 2 ** attempt) + random.random() * 5)
        finally:
            if uploaded:
                try:
                    await client.aio.files.delete(name=uploaded)
                except Exception:
                    pass
    async with lock:
        append(FAILURES, {
            "pid": row.pid,
            "prompt_version": version,
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "error": last_error,
        })
        stats["failed"] += 1


def write_final(meta: pd.DataFrame) -> None:
    rows = {}
    for r in read_jsonl(OUT):  # if a post has both, keep the main-prompt record
        if r["pid"] not in rows or VERSIONS.get(r["prompt_version"]) == "main":
            rows[r["pid"]] = r
    if set(meta.pid) - set(rows):
        return
    df = pd.DataFrame({
        "pid": list(meta.pid),
        "vid": [rows[p]["vid"] for p in meta.pid],
        "record_json": [json.dumps(rows[p]["record"], ensure_ascii=False) for p in meta.pid],
        "record_text": [flat_text(rows[p]["record"]) for p in meta.pid],
        "model": [rows[p]["model"] for p in meta.pid],
        "prompt_version": [rows[p]["prompt_version"] for p in meta.pid],
        "variant": [VERSIONS.get(rows[p]["prompt_version"], "unknown") for p in meta.pid],
    })
    df.to_parquet(FINAL)
    print(f"all {len(df)} videos described -> {FINAL}")


async def main_async(args) -> None:
    meta = pd.read_parquet(CLEAN / "metadata.parquet")
    done_rows = read_jsonl(OUT)
    other = {r.get("prompt_version") for r in done_rows} - set(VERSIONS)
    if other:
        raise SystemExit(
            f"{OUT.name} has rows from an unknown prompt/model ({sorted(other)}); known "
            f"versions are {VERSIONS}. Move that file aside rather than mixing records."
        )
    done = {r["pid"] for r in done_rows}
    todo = meta[~meta.pid.isin(done)]
    if args.only_blocked:
        last = {}
        for f in read_jsonl(FAILURES):
            last[f["pid"]] = f["error"]
        # "JSON input should be string" is how blocked answers failed before
        # the Blocked check existed (the response text was empty).
        blocked = {p for p, e in last.items()
                   if e.startswith("Blocked") or "JSON input should be string" in e}
        todo = todo[todo.pid.isin(blocked)]
    if args.limit:
        todo = todo.head(args.limit)

    secs = todo.video_duration.sum()
    print(f"prompt version {prompt_version(PROMPT_NO_LYRICS if args.no_lyrics else PROMPT)} | {len(meta)} videos, {len(done)} done, "
          f"{len(todo)} to describe ({secs / 60:.0f} min of video)")
    print(f"rough tokens: ~{secs * 95 / 1e6:.1f}M video + ~{len(todo) * 1300 / 1e6:.1f}M prompt in, "
          f"~{len(todo) * 1300 / 1e6:.1f}M out")
    if args.dry_run or todo.empty:
        if todo.empty:
            write_final(meta)
        return

    load_env()
    client = genai.Client(api_key=os.environ["GOOGLE_AI_API_KEY"])
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    stats = {"ok": 0, "failed": 0}
    start = time.time()

    async def one(row):
        async with sem:
            await describe(client, row, args, lock, stats)
            n = stats["ok"] + stats["failed"]
            if n % 25 == 0 or n == len(todo):
                rate = n / max(time.time() - start, 1) * 3600
                print(f"{n}/{len(todo)} ({stats['failed']} failed) | {rate:.0f}/hour", flush=True)

    await asyncio.gather(*(one(r) for r in todo.itertuples()))
    print(f"\nthis run: {stats['ok']} described, {stats['failed']} failed "
          f"(failures are retried on the next run)", flush=True)
    write_final(meta)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=0, help="Describe at most this many (0 = all missing).")
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--retries", type=int, default=10)
    p.add_argument("--max-wait", type=int, default=240, help="Longest pause between retries, in seconds.")
    p.add_argument("--dry-run", action="store_true", help="Print counts and estimate; no API calls.")
    p.add_argument("--no-lyrics", action="store_true",
                   help="Add the rule: describe songs, do not transcribe lyrics.")
    p.add_argument("--only-blocked", action="store_true",
                   help="Only posts whose last failure was a content block (recitation).")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
