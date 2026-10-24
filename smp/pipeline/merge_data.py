"""Merge the video descriptions into the clean metadata and tidy data/clean/.

Run after describe_videos.py. Every post that has a description is kept; posts
without one (Gemini's recitation filter blocked their description) are
discarded from the clean dataset, with the post id and the reason recorded in
clean_meta.json. Their videos stay in data/raw/.

Writes:
    data/clean/merged_data.parquet   metadata + description, one row per kept post
    data/clean/metadata.parquet      the same posts, metadata only
    data/clean/clean_meta.json       updated counts and the discarded posts
and deletes the videos of discarded posts from data/clean/videos/, plus the
run's log and failure files.

Run:
    python smp/pipeline/merge_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from describe_videos import CLEAN, FAILURES, OUT, VERSIONS, VIDEOS, flat_text, read_jsonl

LOG = CLEAN / "describe_videos.log"
MERGED = CLEAN / "merged_data.parquet"
FIELDS = [
    "summary", "content_type", "topic", "format", "hook", "people", "setting",
    "filming", "editing", "audio", "tone", "call_to_action", "products_or_brands",
]


def main() -> None:
    meta = pd.read_parquet(CLEAN / "metadata.parquet")
    records = {}
    for r in read_jsonl(OUT):
        if r["pid"] not in records or VERSIONS.get(r["prompt_version"]) == "main":
            records[r["pid"]] = r

    last_error = {f["pid"]: f["error"] for f in read_jsonl(FAILURES)}
    missing = meta[~meta.pid.isin(records)]
    discarded = [
        {"pid": p, "reason": last_error.get(p, "no description")} for p in missing.pid
    ]
    kept = meta[meta.pid.isin(records)].reset_index(drop=True)

    desc = pd.DataFrame([
        {
            "pid": p,
            **{f"desc_{k}": records[p]["record"][k] for k in FIELDS},
            "desc_n_segments": len(records[p]["record"]["segments"]),
            "desc_n_transcript_lines": len(records[p]["record"]["transcript"]),
            "desc_n_on_screen_text": len(records[p]["record"]["on_screen_text"]),
            "description_text": flat_text(records[p]["record"]),
            "description_json": json.dumps(records[p]["record"], ensure_ascii=False),
            "desc_model": records[p]["model"],
            "desc_prompt_version": records[p]["prompt_version"],
        }
        for p in kept.pid
    ])
    merged = kept.merge(desc, on="pid", how="left", validate="1:1")
    assert merged.description_text.notna().all() and len(merged) == len(kept)

    # Every kept post must have its video on disk.
    no_video = [p for p, v in zip(merged.pid, merged.video_path) if not (VIDEOS / v).exists()]
    if no_video:
        raise SystemExit(f"{len(no_video)} kept posts have no video, e.g. {no_video[0]}")

    merged.to_parquet(MERGED)
    kept.to_parquet(CLEAN / "metadata.parquet")

    for v in missing.video_path:  # discarded posts: drop their copied videos
        (VIDEOS / v).unlink(missing_ok=True)
        try:
            (VIDEOS / v).parent.rmdir()  # only removes the user folder if now empty
        except OSError:
            pass

    meta_path = CLEAN / "clean_meta.json"
    cm = json.loads(meta_path.read_text())
    prev_discarded = cm.get("discarded_posts", [])
    cm.update({
        "n_posts": len(merged),
        "n_users": int(merged.uid.nunique()),
        "popularity_mean": float(merged.popularity.mean()),
        "group_sizes": {int(k): int(v) for k, v in merged.user_group.value_counts().sort_index().items()},
        "group_model_note": "fitted on the 2,192 posts before descriptions; discarded posts keep their edges",
        "descriptions": {
            "script": "smp/pipeline/describe_videos.py (design V4)",
            "model": sorted(set(merged.desc_model)),
            "prompt_versions": sorted(set(merged.desc_prompt_version)),
        },
        "discarded_posts": prev_discarded + [d for d in discarded if d["pid"] not in {x["pid"] for x in prev_discarded}],
        "columns": list(merged.columns),
    })
    meta_path.write_text(json.dumps(cm, indent=2))

    for f in (LOG, FAILURES):
        f.unlink(missing_ok=True)
    for f in CLEAN.rglob(".DS_Store"):
        f.unlink()

    n_videos = sum(1 for _ in VIDEOS.rglob("*.mp4"))
    print(f"kept {len(merged)} posts with descriptions, discarded {len(discarded)} "
          f"(recitation-blocked); {n_videos} videos on disk")
    print(f"wrote {MERGED.name}: {merged.shape[0]} rows x {merged.shape[1]} columns")


if __name__ == "__main__":
    main()
