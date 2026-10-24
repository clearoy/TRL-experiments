"""Turn the merged clean data into the model-ready table, ready_data.parquet.

One row per post. Columns fall into two roles:

  bookkeeping (never a model input)
      pid, uid                  ids; uid lets a split keep each user on one side
      popularity                the raw label; the split turns it into YES/NO
      user_group, expected_popularity
                                the account group (refitted on train by the split)

  content features (the model inputs, for PolicyInduction and the ML baselines)
      caption                   the post's caption, words restored from the comma list
      posted_at_utc             post time, "YYYY-MM-DD HH:MM UTC"
      post_hour_utc, post_weekday
      video_duration_seconds, video_resolution, video_orientation
      music_title, music_duration_seconds
      video_content_type, video_structure
      n_scenes, n_spoken_lines, n_on_screen_texts
      video_description         the full text record of the video (stands in for it)

Column names are written to be read: PolicyInduction shows each row to the LLM
as `column: value` lines, so the name is part of the signal.

Left out on purpose: post_suggested_words (platform-generated search
suggestions that track popularity, so possibly a leak), the user counts (already
reduced to user_group), and constant or bookkeeping columns (video_format,
video_path, vid, the description model and prompt version).

Run:
    python smp/pipeline/make_ready.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SMP = Path(__file__).resolve().parents[1]
CLEAN = SMP / "data" / "clean"
MERGED = CLEAN / "merged_data.parquet"
READY = CLEAN / "ready_data.parquet"

ID_COLUMNS = ["pid", "uid", "popularity", "user_group", "expected_popularity"]
FEATURES = [
    "caption", "posted_at_utc", "post_hour_utc", "post_weekday",
    "video_duration_seconds", "video_resolution", "video_orientation",
    "music_title", "music_duration_seconds",
    "video_content_type", "video_structure",
    "n_scenes", "n_spoken_lines", "n_on_screen_texts",
    "video_description",
]


def caption(raw: str) -> str:
    """The raw caption is its words joined by commas; restore the spaces."""
    words = [w.strip() for w in (raw or "").split(",") if w.strip()]
    return " ".join(words) if words else "(no caption)"


def orientation(h: int, w: int) -> str:
    if h > w * 1.1:
        return "vertical"
    if w > h * 1.1:
        return "horizontal"
    return "square"


def main() -> None:
    m = pd.read_parquet(MERGED)
    t = pd.to_datetime(m.post_time)
    ready = pd.DataFrame({
        "pid": m.pid,
        "uid": m.uid,
        "popularity": m.popularity,
        "user_group": m.user_group,
        "expected_popularity": m.expected_popularity,
        "caption": m.post_content.map(caption),
        "posted_at_utc": t.dt.strftime("%Y-%m-%d %H:%M UTC"),
        "post_hour_utc": t.dt.hour,
        "post_weekday": t.dt.day_name(),
        "video_duration_seconds": m.video_duration,
        "video_resolution": m.video_ratio,
        "video_orientation": [orientation(h, w) for h, w in zip(m.video_height, m.video_width)],
        "music_title": m.music_title.fillna("").replace("", "(none)"),
        "music_duration_seconds": m.music_duration,
        "video_content_type": m.desc_content_type,
        "video_structure": m.desc_format,
        "n_scenes": m.desc_n_segments,
        "n_spoken_lines": m.desc_n_transcript_lines,
        "n_on_screen_texts": m.desc_n_on_screen_text,
        "video_description": m.description_text,
    })[ID_COLUMNS + FEATURES]

    assert ready.pid.is_unique and not ready.isna().any().any()
    ready.to_parquet(READY)

    meta_path = CLEAN / "clean_meta.json"
    cm = json.loads(meta_path.read_text())
    cm["ready_data"] = {
        "file": READY.name,
        "rows": len(ready),
        "id_columns": ID_COLUMNS,
        "feature_columns": FEATURES,
        "excluded": {
            "post_suggested_words": "platform-generated search suggestions; tracks popularity "
            "within every account group, so possibly a leak",
            "user_* counts": "reduced to user_group",
            "video_format, video_path, vid, desc_model, desc_prompt_version": "constant or bookkeeping",
            "desc_* single fields, description_json": "contained in video_description",
        },
    }
    meta_path.write_text(json.dumps(cm, indent=2))
    print(f"wrote {READY.name}: {ready.shape[0]} rows x {ready.shape[1]} columns "
          f"({len(FEATURES)} features)")


if __name__ == "__main__":
    main()
