"""Extract the US + English posts of the full SMP-Video train file, with videos.

Reads the four raw train jsonl files, joins them per post, keeps posts whose
`post_location` is US and `post_text_language` is en, and copies each post's
video next to the metadata:

    smp/data/clean/
        metadata.parquet            one row per post, every column from the raw files
        videos/<video_path>         e.g. videos/train/USER.../VIDEO....mp4
        clean_meta.json

No train/test split and no binary label are applied here: `popularity` is the
raw continuous score.

Videos are copied with `cp -c` (APFS clone: instant, no extra disk until a copy
is modified), falling back to a normal copy elsewhere.

Run:
    python smp/extract_clean.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RAW = DATA / "raw" / "train"
RAW_VIDEO = DATA / "raw" / "video" / "video_file"
CLEAN = DATA / "clean"


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
    }
    (CLEAN / "clean_meta.json").write_text(json.dumps(meta, indent=2))
    print(meta)


if __name__ == "__main__":
    main()
