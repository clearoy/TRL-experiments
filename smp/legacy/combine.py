"""Combine saved PolicyInduction policy scores with the ML baseline features.

No LLM calls: reuses the per-sample policy scores each run already produced
(train scores from <run>/policy_predictions.parquet, test scores from
results/policy_induction_<run>_predictions.csv). Fits one model on the baseline
features plus the policy scores, on train only, and scores it on the same 500
test rows.

Run:
    python smp/combine.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from baselines import TEXT, _log1p_clipped, score  # noqa: E402
from common import model_inputs  # noqa: E402

RUNS = {
    30: ["all_columns", "all_columns_r2", "all_columns_r3"],
    10: ["all_columns_p10", "all_columns_p10_r2", "all_columns_p10_r3"],
}


def policy_matrices(run: str, n_train: int, test_index: pd.Index):
    pp = pd.read_parquet(HERE / run / "policy_predictions.parquet")
    tr = pp.pivot(index="sample_index", columns="policy_id", values="pred").sort_index().astype(float)
    assert len(tr) == n_train, (run, len(tr))
    te = pd.read_csv(HERE / "results" / f"policy_induction_{run}_predictions.csv")
    te = te.set_index("sample_index").sort_index()
    cols = [c for c in te.columns if c.startswith("policy_")]
    te = te.reindex(test_index)[cols]  # rows the run skipped become NaN
    tr = tr.copy()
    tr.columns = [f"policy_{c}" for c in tr.columns]
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def pipeline(X: pd.DataFrame, policy_cols: list[str], model) -> Pipeline:
    cat = ["post_location", "post_text_language", "video_ratio"]
    num = [c for c in X.select_dtypes("number").columns if c not in policy_cols]
    steps = [
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", Pipeline([("log", FunctionTransformer(_log1p_clipped)), ("sc", StandardScaler())]), num),
        ("pol", StandardScaler(), policy_cols),
    ]
    for c in TEXT:
        steps.append((c, TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), c))
    return Pipeline([("pre", ColumnTransformer(steps)), ("clf", model)])


def main() -> None:
    train = pd.read_parquet(HERE / "data" / "train.parquet")
    test = pd.read_parquet(HERE / "data" / "test.parquet")
    y_tr = (train["label"] == "YES").astype(int).to_numpy()
    y_te = (test["label"] == "YES").astype(int).to_numpy()
    Xb_tr, Xb_te = model_inputs(train), model_inputs(test)

    out: dict = {}
    for n_pol, runs in RUNS.items():
        for run in runs:
            p_tr, p_te = policy_matrices(run, len(train), Xb_te.index)
            keep = p_te.notna().all(axis=1).to_numpy()  # drop test rows the run skipped
            cols = list(p_tr.columns)
            X_tr = pd.concat([Xb_tr, p_tr], axis=1)
            X_te = pd.concat([Xb_te, p_te], axis=1)[keep]
            res = {}
            for name, model in {
                "logistic_regression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0),
                "gradient_boosting": GradientBoostingClassifier(random_state=0),
            }.items():
                prob = pipeline(X_tr, cols, model).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
                m = score(y_te[keep], (prob >= 0.5).astype(int), prob)
                m["n_scored"] = int(keep.sum())
                res[name] = m
                print(f"{n_pol:2d} policies {run:20s} {name:20s} auc={m['roc_auc']:.4f} f0.5={m['f0.5']:.4f} "
                      f"p={m['precision']:.4f} r={m['recall']:.4f}")
            out[run] = res
    (HERE / "results" / "combined.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
