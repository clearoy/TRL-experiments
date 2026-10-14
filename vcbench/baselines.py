"""TF-IDF baselines on the VCBench split PolicyInduction will use.

Each model sees only `anonymised_prose`, vectorised with TF-IDF, on the exact
train/test split written by prepare_data.py. Fitted twice:

  full      - on all training rows.
  subsample - on the first `--n-train` rows, matching what PolicyInduction can
              afford at LLM prices. This is the fair head-to-head.

VCBench is ~9% positive (always-NO scores 91% accuracy), so F0.5 and precision
are the headline metrics, not accuracy.

Run:
    python vcbench/baselines.py
    python vcbench/baselines.py --n-train 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
TARGET = "success"
TEXT = "anonymised_prose"


def make_models(seed: int) -> Dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
    }


def build_pipeline(name: str, model: Any) -> Pipeline:
    # Trees on a full bigram vocabulary are slow and overfit; cap their features.
    max_features = None if name == "logistic_regression" else 5000
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=max_features
    )
    return Pipeline([("tfidf", tfidf), ("clf", model)])


def score(y_true, y_pred, y_prob) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f0.5": float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def run(n_train: int, n_test: int, seed: int) -> None:
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")

    test_eval = test.head(n_test) if n_test else test
    X_test, y_test = test_eval[TEXT], test_eval[TARGET]

    regimes = {"full": train}
    if n_train and n_train < len(train):
        regimes["subsample"] = train.head(n_train)

    results: Dict[str, Any] = {
        "n_test": int(len(test_eval)),
        "features": [TEXT],
        "majority_class_accuracy": float(1 - y_test.mean()),
        "positive_rate_test": float(y_test.mean()),
        "regimes": {},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for regime, tr in regimes.items():
        X_tr, y_tr = tr[TEXT], tr[TARGET]
        block: Dict[str, Any] = {"n_train": int(len(tr))}
        print(f"\n=== {regime}  (n_train={len(tr)}, n_test={len(test_eval)}) ===")

        detail = pd.DataFrame(
            {"sample_index": test_eval.index, "y_true": y_test.to_numpy()}
        ).set_index("sample_index")

        for name, model in make_models(seed).items():
            pipe = build_pipeline(name, model).fit(X_tr, y_tr)
            y_prob = pipe.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            m = score(y_test, y_pred, y_prob)
            m["threshold"] = 0.5
            block[name] = m
            detail[f"{name}_prob"] = y_prob
            detail[f"{name}_pred"] = y_pred
            print(
                f"  {name:22s} f0.5={m['f0.5']:.4f}  p={m['precision']:.4f}  "
                f"r={m['recall']:.4f}  auc={m['roc_auc']:.4f}  acc={m['accuracy']:.4f}"
            )

        detail.to_csv(RESULTS_DIR / f"predictions_baselines_{regime}.csv")
        results["regimes"][regime] = block

    print(f"\n  {'majority class (all NO)':22s} acc={results['majority_class_accuracy']:.4f}")
    out = RESULTS_DIR / "baselines.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument(
        "--n-test",
        type=int,
        default=0,
        help="Test rows; 0 (default) = full test set. Must match "
        "run_policy_induction.py or the comparison is invalid.",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args.n_train, args.n_test, args.seed)


if __name__ == "__main__":
    main()
