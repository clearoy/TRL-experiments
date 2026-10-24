"""Split the ready SMP posts by user, then label the top 20% inside each user group.

Reads data/clean/ready_data.parquet (written by make_ready.py) and writes one
folder per account group, keeping every column of ready_data plus `label`:

    smp/data/split/group_1/{train,test}.parquet   ... group_5/
    smp/data/split_meta.json                      (tracked: how the split was made)

Steps
  1. Split users about 80/20 with StratifiedGroupKFold: every post of a user is on
     one side, and the split is stratified by follower quintile so each tier
     is represented on both sides.
  2. Refit the group model (scheme E) on the training posts only: a linear fit
     of popularity on log10 followers, following and video count. Cut its
     prediction into quintiles using training edges, and apply the same model
     and edges to test.
  3. Inside each group, YES = popularity in the top 20% of that group's
     training posts; the same cutoff is applied to the group's test posts.

The user counts needed for step 2 are read from the raw users file by `uid` and
are not written to the output.

Run:
    python smp/pipeline/prepare_data.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parents[1]  # smp/
DATA_DIR = HERE / "data"
CLEAN = DATA_DIR / "clean" / "ready_data.parquet"
USERS = DATA_DIR / "raw" / "train" / "SMP-Video_anonymized_users_train.jsonl"
OUT = DATA_DIR / "split"

GROUP_INPUTS = ["user_follower_count", "user_following_count", "user_video_count"]
N_GROUPS = 5


def build(top_frac: float, n_folds: int, seed: int) -> None:
    if not CLEAN.exists():
        raise SystemExit(f"Missing {CLEAN}. Run pipeline/make_ready.py first.")
    df = pd.read_parquet(CLEAN).drop(columns=["expected_popularity", "user_group"])
    users = pd.read_json(USERS, lines=True)[["uid", *GROUP_INPUTS]]
    df = df.merge(users, on="uid", how="left", validate="m:1")
    assert not df[GROUP_INPUTS].isna().any().any()

    # 1. User-level split, stratified by follower quintile (no labels involved).
    strata = pd.qcut(np.log10(df.user_follower_count.clip(lower=1)), 5, labels=False)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(df, strata, groups=df.uid))
    df["split"] = "train"
    df.loc[df.index[test_idx], "split"] = "test"
    assert not set(df.uid[df.split == "train"]) & set(df.uid[df.split == "test"])

    # 2. Group model on training posts only.
    X = np.log10(df[GROUP_INPUTS].clip(lower=1))
    is_tr = df.split == "train"
    model = LinearRegression().fit(X[is_tr], df.popularity[is_tr])
    df["expected_popularity"] = model.predict(X)
    edges = np.quantile(df.expected_popularity[is_tr], np.linspace(0, 1, N_GROUPS + 1))
    inner = edges[1:-1]  # test posts outside the train range fall into groups 1 or 5
    df["user_group"] = np.searchsorted(inner, df.expected_popularity, side="right") + 1

    meta = {
        "source": "data/clean/ready_data.parquet",
        "split": f"by user (StratifiedGroupKFold, {n_folds} folds, fold 0 = test), "
        "stratified by follower quintile",
        "seed": seed,
        "label": f"YES = popularity >= the group's train {1 - top_frac:.0%} quantile (top {top_frac:.0%})",
        "group_model": {
            "fitted_on": "train posts only",
            "inputs": [f"log10({c})" for c in GROUP_INPUTS],
            "intercept": float(model.intercept_),
            "coef": dict(zip(GROUP_INPUTS, map(float, model.coef_))),
            "train_edges": [float(e) for e in edges],
        },
        "users_in_both": 0,
        "groups": {},
    }

    # 3. Label inside each group and write one folder per group.
    keep = [c for c in df.columns if c not in GROUP_INPUTS + ["split"]] + ["label"]
    for g in range(1, N_GROUPS + 1):
        tr = df[(df.user_group == g) & is_tr].copy()
        te = df[(df.user_group == g) & ~is_tr].copy()
        cutoff = float(tr.popularity.quantile(1 - top_frac))
        for part in (tr, te):
            part["label"] = np.where(part.popularity >= cutoff, "YES", "NO")
        folder = OUT / f"group_{g}"
        folder.mkdir(parents=True, exist_ok=True)
        tr[keep].reset_index(drop=True).to_parquet(folder / "train.parquet")
        te[keep].reset_index(drop=True).to_parquet(folder / "test.parquet")
        meta["groups"][f"group_{g}"] = {
            "popularity_cutoff": cutoff,
            "n_train": len(tr),
            "n_test": len(te),
            "n_yes_train": int((tr.label == "YES").sum()),
            "n_yes_test": int((te.label == "YES").sum()),
            "n_users_train": int(tr.uid.nunique()),
            "n_users_test": int(te.uid.nunique()),
        }

    (DATA_DIR / "split_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["groups"], indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top-frac", type=float, default=0.2)
    p.add_argument("--n-folds", type=int, default=5, help="Test share is 1/n-folds.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    build(args.top_frac, args.n_folds, args.seed)


if __name__ == "__main__":
    main()
