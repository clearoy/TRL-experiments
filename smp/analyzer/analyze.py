"""Analyze one video against the SMP content checklist.

Steps, all on the same footing as the training posts:
  1. Read the video's length and size from the .mp4 itself.
  2. Describe it with Gemini, using the same prompt as the dataset
     (pipeline/describe_videos.py), and build the same `model_text` block
     (pipeline/make_ready.py).
  3. Place the account in its size group from followers, following and videos
     posted (the group model fitted on the dataset).
  4. Ask Jev the checklist's questions about the post.
  5. Return the checklist: for each rule, the answer, whether it matches posts
     that did better or worse for accounts of this size, and how strong that
     pattern was in the data; plus an overall estimate, kept secondary.

Run on the command line:
    python smp/analyzer/analyze.py video.mp4 --caption "..." --followers 84000 \
        --following 300 --videos 450
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
SMP = HERE.parent
REPO_ROOT = SMP.parents[1]
for p in (str(REPO_ROOT), str(SMP / "pipeline")):
    if p not in sys.path:
        sys.path.insert(0, p)

import describe_videos as dv  # noqa: E402  (also loads the library repo's .env)
import make_ready as mr  # noqa: E402

dv.load_env()

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402
from think_reason_learn.core.llms import JevBudget, JevChoice, JevRequest, NoulQuestion, llm  # noqa: E402

MODEL_PATH = HERE / "model.json"


# --------------------------------------------------------------- mp4 facts

def mp4_facts(path: Path) -> dict:
    """Duration (s), width and height from the mp4's moov/mvhd and tkhd boxes."""
    data = path.read_bytes()
    out = {"duration": None, "width": None, "height": None}

    def boxes(buf: bytes, start: int, end: int):
        i = start
        while i + 8 <= end:
            size, kind = struct.unpack(">I4s", buf[i:i + 8])
            header = 8
            if size == 1:
                size = struct.unpack(">Q", buf[i + 8:i + 16])[0]; header = 16
            elif size == 0:
                size = end - i
            if size < header:
                return
            yield kind.decode("latin-1"), i + header, i + size
            i += size

    def walk(start: int, end: int):
        for kind, b, e in boxes(data, start, end):
            if kind in ("moov", "trak", "mdia"):
                walk(b, e)
            elif kind == "mvhd":
                v = data[b]
                if v == 1:
                    ts, dur = struct.unpack(">IQ", data[b + 20:b + 32])
                else:
                    ts, dur = struct.unpack(">II", data[b + 12:b + 20])
                if ts:
                    out["duration"] = dur / ts
            elif kind == "tkhd":
                w, h = struct.unpack(">II", data[e - 8:e])
                w, h = w >> 16, h >> 16
                if w and h and not out["width"]:
                    out["width"], out["height"] = w, h

    walk(0, len(data))
    return out


def resolution_label(w: int | None, h: int | None) -> str:
    """The platform's label for the short side, as used in the dataset
    (576-pixel-wide videos are labelled 540p)."""
    if not w or not h:
        return "unknown"
    short = min(w, h)
    for label, top in (("360p", 400), ("480p", 500), ("540p", 600), ("720p", 800)):
        if short <= top:
            return label
    return "1080p"


# --------------------------------------------------------------- describe

async def describe(path: Path) -> dict:
    """Gemini text record of the video, same prompt and schema as the dataset."""
    client = genai.Client(api_key=os.environ["GOOGLE_AI_API_KEY"])
    for prompt, variant in ((dv.PROMPT, "main"), (dv.PROMPT_NO_LYRICS, "no_lyrics")):
        config = types.GenerateContentConfig(response_mime_type="application/json",
                                             response_schema=dv.VideoRecord, temperature=0.2,
                                             max_output_tokens=16384)
        last = None
        for attempt in range(1, 9):
            uploaded = None
            try:
                part, uploaded = await dv.video_part(client, path)
                resp = await asyncio.wait_for(client.aio.models.generate_content(
                    model=dv.MODEL, contents=[types.Content(role="user", parts=[part, types.Part.from_text(text=prompt)])],
                    config=config), timeout=dv.CALL_TIMEOUT)
                if resp.text is None:
                    cand = resp.candidates[0] if resp.candidates else None
                    raise dv.Blocked(getattr(getattr(cand, "finish_reason", None), "name", "UNKNOWN"))
                rec = dv.scrub_handles(dv.VideoRecord.model_validate_json(resp.text).model_dump())
                rec["_variant"] = variant
                return rec
            except dv.Blocked as e:
                last = e
                break  # try the no-lyrics prompt
            except Exception as e:  # overload, timeout
                last = e
                await asyncio.sleep(min(30, 2 ** attempt))
            finally:
                if uploaded:
                    try:
                        await client.aio.files.delete(name=uploaded)
                    except Exception:
                        pass
        if not isinstance(last, dv.Blocked):
            raise RuntimeError(f"Gemini could not describe the video: {last}")
    raise RuntimeError("Gemini blocked the description (recitation filter), even without lyrics.")


# ----------------------------------------------------------------- scoring

def account_group(model: dict, followers: int, following: int, videos: int) -> tuple[int, float]:
    gm = model["group_model"]
    x = {"user_follower_count": followers, "user_following_count": following, "user_video_count": videos}
    expected = gm["intercept"] + sum(gm["coef"][k] * math.log10(max(x[k], 1)) for k in gm["coef"])
    inner = model["group_edges"][1:-1]
    group = 1 + sum(expected >= e for e in inner)
    return int(group), float(expected)


async def ask_jev(model: dict, model_text: str) -> dict:
    qs = {r["id"]: NoulQuestion(instructions=r["question"]) for r in model["rules"]}
    release = model["created_with"].get("jev_release") or "jev-latest"
    for name in dict.fromkeys([release, "jev-latest"]):
        try:
            res = await llm.answer_many([JevChoice(model=name, max_cost_usd=1.0)],
                                        [JevRequest(state=f"post: {model_text}", questions=qs)],
                                        budget=JevBudget(1.0), caller="smp_analyzer")
            ans = {k: v for k, v in res[0].answers.items() if isinstance(v, float)}
            if ans:
                return {"answers": ans, "jev_model": name}
        except Exception as e:  # e.g. a pinned release no longer served
            last = e
    raise RuntimeError(f"Jev did not answer: {last if 'last' in dir() else 'no answers'}")


def checklist(model: dict, answers: dict) -> dict:
    items, logit = [], model["intercept"]
    for r in model["rules"]:
        p = answers.get(r["id"])
        if p is None:
            items.append({**r, "answer": None, "status": "unanswered"})
            continue
        z = (p - r["mean"]) / r["sd"]
        logit += r["weight"] * z
        present = p >= 0.5
        if r["direction"] == "helps":
            status = "match_hit" if present else "missing_hit"
        else:
            status = "match_weak" if present else "avoids_weak"
        items.append({**r, "answer": p, "present": present, "contribution": r["weight"] * z, "status": status})
    prob = 1 / (1 + math.exp(-logit))
    return {"items": items, "probability": prob}


async def analyze(video: Path, caption: str, followers: int, following: int, videos: int,
                  music: str = "", posted_at: datetime | None = None,
                  progress: Callable[[str], None] = lambda s: None) -> dict:
    model = json.loads(MODEL_PATH.read_text())
    posted_at = posted_at or datetime.now(timezone.utc)

    progress("Reading the video file")
    facts = mp4_facts(video)
    progress("Describing the video with Gemini (about 20-60 s)")
    rec = await describe(video)

    row = {
        "caption": caption.strip() or "(no caption)",
        "post_weekday": posted_at.strftime("%A"),
        "posted_at_utc": posted_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "music_title": music.strip() or "original sound",
        "video_duration_seconds": int(facts["duration"] or 0),  # the dataset rounds down
        "video_orientation": mr.orientation(facts["height"] or 1, facts["width"] or 1),
        "video_resolution": resolution_label(facts["width"], facts["height"]),
    }
    text = mr.model_text(row, rec)

    progress("Placing the account in its size group")
    group, expected = account_group(model, followers, following, videos)
    progress("Asking Jev the checklist questions")
    jev = await ask_jev(model, text)
    progress("Building the checklist")
    cl = checklist(model, jev["answers"])
    return {
        "video": {**row, "width": facts["width"], "height": facts["height"]},
        "account": {"followers": followers, "following": following, "videos": videos,
                    "group": group, "expected_popularity": expected},
        "probability": cl["probability"], "base_rate": model["base_rate"],
        "checklist": cl["items"],
        "description": {k: v for k, v in rec.items() if not k.startswith("_")},
        "description_variant": rec.get("_variant"),
        "model_text": text, "jev_model": jev["jev_model"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--caption", default="")
    p.add_argument("--music", default="")
    p.add_argument("--followers", type=int, required=True)
    p.add_argument("--following", type=int, required=True)
    p.add_argument("--videos", type=int, required=True, help="Videos the account has posted.")
    p.add_argument("--json", action="store_true", help="Print the full JSON result.")
    a = p.parse_args()
    res = asyncio.run(analyze(a.video, a.caption, a.followers, a.following, a.videos, a.music,
                              progress=lambda s: print(f"... {s}", file=sys.stderr)))
    if a.json:
        print(json.dumps(res, indent=2, default=str))
        return
    print(f"\nAccount group {res['account']['group']} of 5 | estimate {res['probability']:.0%} "
          f"(typical {res['base_rate']:.0%})\n")
    labels = {"match_hit": "✓ matches hits", "missing_hit": "· missing", "match_weak": "✗ matches weaker posts",
              "avoids_weak": "✓ avoids", "unanswered": "? unanswered"}
    for it in res["checklist"]:
        print(f"{labels[it['status']]:24s} {it['question']}")


if __name__ == "__main__":
    main()
