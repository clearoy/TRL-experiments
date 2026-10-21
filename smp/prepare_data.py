"""Split the US + English SMP-Video posts by time: the later posts are the test set.

Reads data/clean/metadata.parquet (written by extract_clean.py), sorts the posts
by `post_time` and cuts at a fixed fraction, so every test post is newer than
every train post. Ties on `post_time` are broken by `pid`, so the order is
deterministic. `popularity` stays the raw continuous score; no binary label is
added here.

Run:
    python smp/prepare_data.py
    python smp/prepare_data.py --test-frac 0.2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
CLEAN = DATA_DIR / "clean" / "metadata.parquet"


def build(test_frac: float) -> None:
    if not CLEAN.exists():
        raise SystemExit(f"Missing {CLEAN}. Run extract_clean.py first.")
    df = pd.read_parquet(CLEAN).sort_values(["post_time", "pid"]).reset_index(drop=True)

    n_test = round(len(df) * test_frac)
    n_train = len(df) - n_test
    train, test = df.iloc[:n_train].copy(), df.iloc[n_train:].copy()
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    assert train.post_time.max() <= test.post_time.min()

    train.to_parquet(DATA_DIR / "train.parquet")
    test.to_parquet(DATA_DIR / "test.parquet")

    meta = {
        "source": "data/clean/metadata.parquet (US + en posts of the 4,000 labelled train posts)",
        "split": "by post_time: earliest posts are train, latest posts are test",
        "test_frac": test_frac,
        "n_train": len(train),
        "n_test": len(test),
        "train_time_range": [str(train.post_time.min()), str(train.post_time.max())],
        "test_time_range": [str(test.post_time.min()), str(test.post_time.max())],
        "same_timestamp_at_cut": bool(train.post_time.iloc[-1] == test.post_time.iloc[0]),
        "popularity_mean_train": float(train.popularity.mean()),
        "popularity_mean_test": float(test.popularity.mean()),
        "popularity_std_train": float(train.popularity.std()),
        "popularity_std_test": float(test.popularity.std()),
        "users_in_both": int(len(set(train.uid) & set(test.uid))),
        "test_posts_from_train_users": int(test.uid.isin(set(train.uid)).sum()),
        "columns": list(df.columns),
    }
    (DATA_DIR / "split_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "columns"}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-frac", type=float, default=0.2)
    args = p.parse_args()
    build(args.test_frac)


if __name__ == "__main__":
    main()
