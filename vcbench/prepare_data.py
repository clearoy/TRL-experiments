"""Prepare VCBench (founder success prediction) for the PolicyInduction benchmark.

VCBench ships as two fixed files of 4,500 founders each (9% positive): a public
set and a private set. The public set is the training pool and the private set
is the held-out test, so every model shares the split the benchmark defines
rather than one we draw ourselves.

Source CSVs are read from the library repo's old_vcbench_evaluation/data/.

Run:
    python vcbench/prepare_data.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
SRC_DIR = HERE.parent.parent / "old_vcbench_evaluation" / "data"
TRAIN_CSV = SRC_DIR / "vcbench_final_public.csv"
TEST_CSV = SRC_DIR / "vcbench_final_private.csv"

TARGET = "success"

# The only input either method sees. industry and the *_json columns are left
# out: the prose is the sole model input for both train and test.
FEATURES = ["anonymised_prose"]


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Expected the library's old_vcbench_evaluation/data/.")
    return pd.read_csv(path)


def build() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train, test = load(TRAIN_CSV), load(TEST_CSV)

    overlap = set(train["founder_uuid"]) & set(test["founder_uuid"])
    if overlap:
        raise SystemExit(f"Refusing to build: {len(overlap)} founders in both splits")

    cols = ["founder_uuid", *FEATURES, TARGET]
    out = {}
    for name, df in (("train", train), ("test", test)):
        df = df[cols].dropna(subset=[TARGET]).reset_index(drop=True)
        df["label"] = df[TARGET].map({1: "YES", 0: "NO"})
        df.to_parquet(DATA_DIR / f"{name}.parquet")
        out[name] = df

    meta = {
        "source": "library old_vcbench_evaluation/data (public=train, private=test)",
        "features": FEATURES,
        "target": TARGET,
        "n_train": len(out["train"]),
        "n_test": len(out["test"]),
        "positive_rate_train": float(out["train"][TARGET].mean()),
        "positive_rate_test": float(out["test"][TARGET].mean()),
    }
    (DATA_DIR / "split_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    build()


if __name__ == "__main__":
    main()
