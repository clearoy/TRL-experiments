"""Baselines on the SMP-Video split (label: top 10% of popularity).

Same inputs as PolicyInduction (common.model_inputs): numeric columns are
log-scaled, categories one-hot, free text TF-IDF. Run with and without the
account-size columns, matching run_policy_induction.py's --no-user-stats.

Run:
    python smp/baselines.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import model_inputs  # noqa: E402

DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
TEXT = ["post_content", "post_suggested_words", "music_title"]


def _log1p_clipped(x):
    # A few users have negative likes/heart counts (sentinel values); log1p of
    # those is NaN, so clip to 0 first.
    return np.log1p(np.clip(x, 0, None))


def build(X: pd.DataFrame, model) -> Pipeline:
    cat = [c for c in ("post_location", "post_text_language", "video_ratio") if c in X]
    num = list(X.select_dtypes("number").columns)
    steps = [
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        (
            "num",
            Pipeline([("log", FunctionTransformer(_log1p_clipped)), ("sc", StandardScaler())]),
            num,
        ),
    ]
    for c in TEXT:
        steps.append((c, TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), c))
    return Pipeline([("pre", ColumnTransformer(steps)), ("clf", model)])


def score(y, pred, prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f0.5": float(fbeta_score(y, pred, beta=0.5, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def main() -> None:
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")
    y_tr = (train["label"] == "YES").astype(int).to_numpy()
    y_te = (test["label"] == "YES").astype(int).to_numpy()

    results = {
        "n_train": len(train),
        "n_test": len(test),
        "positive_rate_test": float(y_te.mean()),
        "majority_class_accuracy": float(1 - y_te.mean()),
        "variants": {},
    }
    for variant, user_stats in (("all_columns", True), ("content_only", False)):
        X_tr, X_te = model_inputs(train, user_stats), model_inputs(test, user_stats)
        block = {}
        for name, model in {
            "logistic_regression": LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=0
            ),
            "gradient_boosting": GradientBoostingClassifier(random_state=0),
        }.items():
            prob = build(X_tr, model).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
            m = score(y_te, (prob >= 0.5).astype(int), prob)
            m["threshold"] = 0.5
            block[name] = m
            print(
                f"{variant:13s} {name:20s} auc={m['roc_auc']:.4f} f0.5={m['f0.5']:.4f} "
                f"p={m['precision']:.4f} r={m['recall']:.4f}"
            )
        results["variants"][variant] = block

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "baselines.json").write_text(json.dumps(results, indent=2))
    print(f"majority class accuracy {results['majority_class_accuracy']:.4f}")


if __name__ == "__main__":
    main()
