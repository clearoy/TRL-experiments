"""Prepare SMP-Video (social media popularity) for the PolicyInduction benchmark.

Only the train split of the Hugging Face dataset has labels, so the evaluation
split is carved out of it: take the first N_TOTAL rows of the popularity file,
join each to its post, user and video metadata, then split them into
N_TRAIN / N_TEST with a seeded shuffle. The binary label is YES for the top 10%
of popularity (cutoff taken from train only).

Prerequisite (gated dataset, needs `hf auth login` and accepted terms):
    snapshot_download("smpchallenge/SMP-Video", repo_type="dataset",
                      local_dir="smp/data/raw")

Run:
    python smp/prepare_data.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RAW = DATA_DIR / "raw" / "train"
TARGET = "popularity"
TOP_FRACTION = 0.10  # label YES = top 10% of popularity


def read(name: str) -> pd.DataFrame:
    path = RAW / f"SMP-Video_anonymized_{name}_train.jsonl"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Download the dataset into data/raw first.")
    return pd.read_json(path, lines=True)


def build(n_total: int, n_train: int, seed: int) -> None:
    pop, posts, users, vids = (read(n) for n in ("popularity", "posts", "users", "videos"))

    df = (
        pop.head(n_total)
        .merge(posts.drop(columns="uid"), on="pid", how="left", validate="1:1")
        .merge(users, on="uid", how="left", validate="m:1")
        .merge(vids.drop(columns="uid"), on="pid", how="left", validate="1:1")
    )
    # A silent NaN here would mean a post lost its user or video metadata.
    gaps = df[["video_path", "user_follower_count", "vid"]].isna().sum()
    if gaps.any():
        raise SystemExit(f"Refusing to build: unjoined rows\n{gaps[gaps > 0]}")

    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train, test = shuffled.iloc[:n_train], shuffled.iloc[n_train:].reset_index(drop=True)
    assert len(train) == n_train and len(test) == n_total - n_train

    # Cutoff comes from train only and is applied unchanged to test.
    cutoff = float(train[TARGET].quantile(1 - TOP_FRACTION))
    train, test = train.copy(), test.copy()
    for d in (train, test):
        d["label"] = (d[TARGET] >= cutoff).map({True: "YES", False: "NO"})

    train.to_parquet(DATA_DIR / "train.parquet")
    test.to_parquet(DATA_DIR / "test.parquet")

    meta = {
        "source": "smpchallenge/SMP-Video, train split only (test has no labels)",
        "rows_used": n_total,
        "n_train": len(train),
        "n_test": len(test),
        "split": f"first {n_total} popularity rows, seeded shuffle",
        "seed": seed,
        "target": TARGET,
        "label": f"YES = popularity >= train 90th percentile ({cutoff:.4f})",
        "popularity_cutoff": cutoff,
        "n_yes_train": int((train["label"] == "YES").sum()),
        "n_yes_test": int((test["label"] == "YES").sum()),
        "columns": list(df.columns),
        "popularity_mean_train": float(train[TARGET].mean()),
        "popularity_mean_test": float(test[TARGET].mean()),
        "popularity_std_train": float(train[TARGET].std()),
        "popularity_std_test": float(test[TARGET].std()),
        "overlapping_users": int(len(set(train["uid"]) & set(test["uid"]))),
    }
    (DATA_DIR / "split_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-total", type=int, default=1500)
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    build(args.n_total, args.n_train, args.seed)


if __name__ == "__main__":
    main()
