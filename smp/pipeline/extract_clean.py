"""Extract the US + English posts of the full SMP-Video train file, with videos.

Reads the four raw train jsonl files, joins them per post, keeps posts whose
`post_location` is US and `post_text_language` is en, groups the posts by
account strength, and copies each post's video next to the metadata.

Grouping (scheme E): a linear model predicts popularity from log10 followers,
following and video count (the likes/heart/digg counts are left out because
they partly include the post's own likes). Its prediction, `expected_popularity`,
is cut into quintiles, `user_group` 1 (weakest accounts) to 5 (strongest).
The user columns are then dropped, so the metadata keeps only the group, the
label, and the post and video columns. `post_location` and `post_text_language`
are dropped too: after the filter they are the same on every row. `uid` is kept
as an identifier only, so a split can keep each user's posts on one side.

    smp/data/clean/
        metadata.parquet            one row per post: ids, label, group, post + video columns
        videos/<video_path>         e.g. videos/train/USER.../VIDEO....mp4
        clean_meta.json

No train/test split and no binary label are applied here: `popularity` is the
raw continuous score. The group model is fitted on all posts here; refit it on
the training posts only once a split exists.

Videos are copied with `cp -c` (APFS clone: instant, no extra disk until a copy
is modified), falling back to a normal copy elsewhere.

Run:
    python smp/pipeline/extract_clean.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

HERE = Path(__file__).resolve().parents[1]  # smp/
DATA = HERE / "data"
RAW = DATA / "raw" / "train"
RAW_VIDEO = DATA / "raw" / "video" / "video_file"
CLEAN = DATA / "clean"

GROUP_INPUTS = ["user_follower_count", "user_following_count", "user_video_count"]
N_GROUPS = 5
KEEP = [
    "pid", "vid", "uid",
    "popularity",
    "expected_popularity", "user_group",
    "post_content", "post_suggested_words", "post_time",
    "video_duration", "video_height", "video_width", "video_ratio", "video_format",
    "music_title", "music_duration",
    "video_path",
]


def read(name: str) -> pd.DataFrame:
    return pd.read_json(RAW / f"SMP-Video_anonymized_{name}_train.jsonl", lines=True)


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if subprocess.run(["cp", "-c", str(src), str(dst)], capture_output=True).returncode:
        shutil.copy2(src, dst)


def main() -> None:
    df = (
        read("popularity")
        .merge(read("posts").drop(columns="uid"), on="pid", how="left", validate="1:1")
        .merge(read("users"), on="uid", how="left", validate="m:1")
        .merge(read("videos").drop(columns="uid"), on="pid", how="left", validate="1:1")
    )
    assert len(df) == 4000 and not df.isna().any().any(), "unexpected raw data"

    df = df[(df.post_location == "US") & (df.post_text_language == "en")]
    df = df.reset_index(drop=True)
    missing = [p for p in df.video_path if not (RAW_VIDEO / p).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} videos missing from {RAW_VIDEO}, e.g. {missing[0]}")

    # Scheme E: one expected-popularity score from the clean user columns, cut
    # into equal-sized groups.
    X = np.log10(df[GROUP_INPUTS].clip(lower=1))
    model = LinearRegression().fit(X, df.popularity)
    df["expected_popularity"] = model.predict(X)
    df["user_group"] = pd.qcut(df.expected_popularity, N_GROUPS, labels=False) + 1
    edges = pd.qcut(df.expected_popularity, N_GROUPS, retbins=True)[1]
    df = df[KEEP]

    for p in df.video_path:
        dst = CLEAN / "videos" / p
        if not dst.exists():
            copy(RAW_VIDEO / p, dst)
    df.to_parquet(CLEAN / "metadata.parquet")

    meta = {
        "filter": "post_location == 'US' and post_text_language == 'en'",
        "source": "smp/data/raw/train (all 4,000 labelled posts)",
        "n_posts": len(df),
        "n_users": int(df.uid.nunique()),
        "popularity_mean": float(df.popularity.mean()),
        "split": "none",
        "group_model": {
            "inputs": [f"log10({c})" for c in GROUP_INPUTS],
            "intercept": float(model.intercept_),
            "coef": dict(zip(GROUP_INPUTS, map(float, model.coef_))),
            "fitted_on": "all posts in this file",
        },
        "group_edges": [float(e) for e in edges],
        "group_sizes": {int(k): int(v) for k, v in df.user_group.value_counts().sort_index().items()},
        "columns": KEEP,
    }
    (CLEAN / "clean_meta.json").write_text(json.dumps(meta, indent=2))
    print(meta)


if __name__ == "__main__":
    main()
