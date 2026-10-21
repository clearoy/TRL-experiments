"""Describe what happens in each SMP video, as structured text, with Gemini.

Gemini reads the .mp4 directly (frames and audio track), so nothing is
pre-extracted. Each video gets one structured JSON description, written as one
line of data/clean/video_content.jsonl. The run is resumable: posts already in
the output file are skipped, so re-running the same command continues.

The model sees only the video. It is not given the caption, the account, or
the popularity, so the description reflects the video alone and cannot echo
the label. It is told to describe, not to judge quality or predict popularity.

Videos up to 18 MB are sent inline; larger ones go through the Files API and
are deleted after use.

Run:
    python smp/pipeline/describe_videos.py --dry-run          # counts and token estimate, no calls
    python smp/pipeline/describe_videos.py --limit 5          # pilot on 5 videos
    python smp/pipeline/describe_videos.py                    # everything not yet described
    python smp/pipeline/describe_videos.py --low-res          # ~1/3 of the video tokens
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parents[1]  # smp/
REPO_ROOT = HERE.parents[1]  # the library repo
CLEAN = HERE / "data" / "clean"
VIDEOS = CLEAN / "videos"
OUT = CLEAN / "video_content.jsonl"
INLINE_LIMIT = 18 * 1024 * 1024  # request limit is 20 MB; leave room for the prompt


PROMPT = """\
You are annotating short-form social media videos for a research dataset.
Watch and listen to the whole video, then describe it in the JSON schema provided.

Rules:
- Describe only what is actually seen or heard. If something is unclear,
  write "unclear" rather than guessing.
- Be neutral and factual. Do not judge quality, creativity, or appeal, and do
  not guess how popular or engaging the video is.
- Do not name real people unless their name is shown on screen or spoken.
  Describe them instead (e.g. "a young woman", "two men in their 30s").
- Do not transcribe usernames, handles, watermarks, or any view, like, or
  follower numbers that appear on screen.
- Quote on-screen text and speech in the original wording, briefly.
- Times are seconds from the start of the video.
- Keep each field short; fill every field."""


class VideoContent(BaseModel):
    summary: str = Field(
        description="2-3 sentences: what happens in the video, start to finish."
    )
    content_type: str = Field(
        description="One main type, e.g. tutorial/how-to, recipe/cooking, comedy skit, "
        "prank, vlog/day-in-the-life, story time, talking head/opinion, reaction, "
        "dance, lip-sync, music performance, product demo/review, unboxing/haul, "
        "beauty/fashion, fitness, pets/animals, sports, gaming, news/commentary, "
        "educational/facts, ASMR, art/craft/DIY, travel/scenery, slideshow, other."
    )
    topic: str = Field(description="The subject in a few words, e.g. 'budget meal prep'.")
    hook: str = Field(
        description="What the viewer sees and hears in the first 3 seconds."
    )
    structure: str = Field(
        description="How it unfolds, e.g. 'setup then punchline', 'step-by-step list', "
        "'before/after reveal', 'single continuous shot', 'montage'."
    )
    key_moment: str = Field(
        description="The main payoff, twist, reveal or punchline and roughly when it "
        "happens, or 'none'."
    )
    people: str = Field(
        description="How many people appear, a neutral description of them, and "
        "whether anyone speaks to the camera. 'none' if no people."
    )
    setting: str = Field(description="Where it is filmed, e.g. kitchen, car, outdoors, studio.")
    filming_style: str = Field(
        description="e.g. selfie/handheld phone, tripod, screen recording, "
        "professional camera, green screen, stock or edited footage."
    )
    editing_pace: str = Field(
        description="slow / medium / fast, with a rough number of cuts."
    )
    on_screen_text: str = Field(
        description="The main text overlays, quoted briefly, or 'none'."
    )
    speech: str = Field(
        description="Who speaks (on camera, voice-over, none) and a short summary of "
        "what is said, with one representative quote."
    )
    audio: str = Field(
        description="Background audio: speech only, popular song, instrumental music, "
        "sound effects, ambient, or silent. Name a song only if it is clearly "
        "identifiable."
    )
    tone: str = Field(
        description="Emotional tone, e.g. funny, wholesome, informative, dramatic, "
        "calm, sarcastic."
    )
    call_to_action: str = Field(
        description="Any explicit request to follow, like, comment, share, buy, or "
        "visit a link, quoted, or 'none'."
    )
    products_or_brands: str = Field(
        description="Products, brands or businesses shown or mentioned, or 'none'."
    )


def load_env() -> None:
    """Export the library repo's .env (API keys) without overriding the shell."""
    env = REPO_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def done_pids() -> set[str]:
    if not OUT.exists():
        return set()
    return {json.loads(line)["pid"] for line in OUT.read_text().splitlines() if line.strip()}


async def video_part(client: genai.Client, path: Path) -> tuple[types.Part, str | None]:
    """Inline small files; upload large ones. Returns (part, uploaded file name)."""
    if path.stat().st_size <= INLINE_LIMIT:
        return types.Part.from_bytes(data=path.read_bytes(), mime_type="video/mp4"), None
    f = await client.aio.files.upload(
        file=str(path), config=types.UploadFileConfig(mime_type="video/mp4")
    )
    while f.state and f.state.name == "PROCESSING":
        await asyncio.sleep(3)
        f = await client.aio.files.get(name=f.name)
    if f.state and f.state.name != "ACTIVE":
        raise RuntimeError(f"upload ended in state {f.state.name}")
    return types.Part.from_uri(file_uri=f.uri, mime_type="video/mp4"), f.name


async def describe(
    client: genai.Client, row, args: argparse.Namespace, lock: asyncio.Lock
) -> bool:
    path = VIDEOS / row.video_path
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=VideoContent,
        temperature=0.2,
        media_resolution=(
            types.MediaResolution.MEDIA_RESOLUTION_LOW if args.low_res else None
        ),
    )
    for attempt in range(1, args.retries + 1):
        uploaded = None
        try:
            part, uploaded = await video_part(client, path)
            resp = await client.aio.models.generate_content(
                model=args.model,
                contents=[types.Content(role="user", parts=[part, types.Part.from_text(text=PROMPT)])],
                config=config,
            )
            content = VideoContent.model_validate_json(resp.text)
            usage = resp.usage_metadata
            record = {
                "pid": row.pid,
                "vid": row.vid,
                "model": args.model,
                "low_res": args.low_res,
                **content.model_dump(),
                "prompt_tokens": getattr(usage, "prompt_token_count", None),
                "output_tokens": getattr(usage, "candidates_token_count", None),
            }
            async with lock:
                with OUT.open("a") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            return True
        except Exception as e:  # network errors, rate limits, malformed JSON
            wait = min(60, 2**attempt)
            print(f"  {row.pid} attempt {attempt} failed: {type(e).__name__}: {str(e)[:120]}")
            if attempt < args.retries:
                await asyncio.sleep(wait)
        finally:
            if uploaded:
                try:
                    await client.aio.files.delete(name=uploaded)
                except Exception:
                    pass
    return False


async def main_async(args: argparse.Namespace) -> None:
    meta = pd.read_parquet(CLEAN / "metadata.parquet")
    todo = meta[~meta.pid.isin(done_pids())]
    if args.limit:
        todo = todo.head(args.limit)

    secs = todo.video_duration.sum()
    per_sec = 100 if args.low_res else 300  # ~frames + audio tokens per second
    print(f"{len(meta)} videos, {len(meta) - len(meta[~meta.pid.isin(done_pids())])} already done, "
          f"{len(todo)} to describe ({secs / 60:.0f} min of video)")
    print(f"rough input estimate: ~{secs * per_sec / 1e6:.1f}M video tokens "
          f"+ ~{len(todo) * 700 / 1e6:.1f}M prompt tokens; output ~{len(todo) * 600 / 1e6:.1f}M tokens")
    if args.dry_run or todo.empty:
        return

    load_env()
    key = os.environ.get("GOOGLE_AI_API_KEY")
    if not key:
        raise SystemExit("GOOGLE_AI_API_KEY not set (looked in the library repo's .env).")
    client = genai.Client(api_key=key)

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    ok = fail = 0
    start = time.time()

    async def one(row):
        nonlocal ok, fail
        async with sem:
            if await describe(client, row, args, lock):
                ok += 1
            else:
                fail += 1
            n = ok + fail
            if n % 10 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} done ({fail} failed), {time.time() - start:.0f}s")

    await asyncio.gather(*(one(r) for r in todo.itertuples()))
    print(f"\nwrote {ok} descriptions to {OUT}; {fail} failed (re-run to retry them)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gemini-3.5-flash")
    p.add_argument("--limit", type=int, default=0, help="Describe at most this many (0 = all).")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--low-res", action="store_true",
                   help="Gemini low media resolution: far fewer video tokens, less visual detail.")
    p.add_argument("--dry-run", action="store_true", help="Print counts and estimates; no API calls.")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
