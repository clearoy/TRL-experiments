"""ML baselines on the SMP split (pipeline/prepare_data.py).

Task: predict the label (YES = top 20% of popularity within the post's user
group) from the post's content. Every model reads the same input PolicyInduction
will: TF-IDF of `model_text`, plus the video length as a number. Nothing about
the account (ids, user_group, expected_popularity) is an input.

Reported on the test set:
  roc_auc            ranking quality, the main metric (0.5 = chance)
  roc_auc_ci         95% bootstrap interval
  roc_auc_by_group   mean of the per-group AUCs (each group has its own cutoff)
  avg_precision      area under the precision-recall curve (chance = share of YES)
  precision_at_20    precision among the 20% of test posts the model ranks highest

Checks alongside the baselines:
  account size only  expected_popularity as the score; near 0.5 if the
                     within-group label really controls for account size
  input ablations    caption only / video description only (TF-IDF + logistic)

Run:
    python smp/baselines.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
SPLIT = HERE / "data" / "split"
RESULTS = HERE / "results"
SEED = 0


def metrics(y: np.ndarray, score: np.ndarray, groups: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            boots.append(roc_auc_score(y[i], score[i]))
    k = int(round(0.2 * len(y)))
    top = np.argsort(-score)[:k]
    by_group = [roc_auc_score(y[groups == g], score[groups == g])
                for g in np.unique(groups) if 0 < y[groups == g].sum() < (groups == g).sum()]
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "roc_auc_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "roc_auc_by_group": float(np.mean(by_group)),
        "roc_auc_each_group": [round(float(a), 4) for a in by_group],
        "avg_precision": float(average_precision_score(y, score)),
        "precision_at_20": float(y[top].mean()),
    }


def tfidf(max_features: int | None):
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                           max_features=max_features, stop_words="english")


def text_plus_length(vec, train: pd.DataFrame, test: pd.DataFrame, column: str):
    """TF-IDF of one text column plus standardised log video length."""
    Xtr_t = vec.fit_transform(train[column])
    Xte_t = vec.transform(test[column])
    lt, le = np.log1p(train.video_duration_seconds), np.log1p(test.video_duration_seconds)
    mu, sd = lt.mean(), lt.std()
    Xtr = hstack([Xtr_t, csr_matrix(((lt - mu) / sd).to_numpy()[:, None])]).tocsr()
    Xte = hstack([Xte_t, csr_matrix(((le - mu) / sd).to_numpy()[:, None])]).tocsr()
    return Xtr, Xte


def main() -> None:
    train = pd.read_parquet(SPLIT / "train.parquet")
    test = pd.read_parquet(SPLIT / "test.parquet")
    ytr = (train.label == "YES").to_numpy().astype(int)
    yte = (test.label == "YES").to_numpy().astype(int)
    gte = test.user_group.to_numpy()

    out = {
        "n_train": len(train), "n_test": len(test),
        "yes_rate_test": float(yte.mean()),
        "input": "TF-IDF(model_text, 1-2 grams, min_df 2) + log video length",
        "models": {}, "checks": {},
    }
    preds = pd.DataFrame({"pid": test.pid, "user_group": gte, "y_true": yte})

    models = {
        "logistic_regression": (LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000), None),
        "random_forest": (RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample",
                                                 n_jobs=-1, random_state=SEED), 5000),
        "gradient_boosting": (GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                                         subsample=0.8, random_state=SEED), 5000),
    }
    for name, (model, max_feat) in models.items():
        Xtr, Xte = text_plus_length(tfidf(max_feat), train, test, "model_text")
        score = model.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        out["models"][name] = metrics(yte, score, gte)
        preds[name] = score

    # Check 1: account size alone should carry ~no signal about a within-group label.
    out["checks"]["account_size_only (expected_popularity)"] = metrics(yte, test.expected_popularity.to_numpy(), gte)
    # Check 2: which part of the text carries the signal.
    for name, col in (("caption_only", "caption"), ("video_description_only", "video_description")):
        Xtr, Xte = text_plus_length(tfidf(None), train, test, col)
        score = LogisticRegression(class_weight="balanced", max_iter=5000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        out["checks"][f"{name} (logistic)"] = metrics(yte, score, gte)
    # Check 3: video length alone.
    out["checks"]["video_length_only"] = metrics(yte, -np.abs(np.log1p(test.video_duration_seconds) - np.log1p(train.video_duration_seconds[ytr == 1]).median()), gte)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "baselines.json").write_text(json.dumps(out, indent=2))
    preds.to_csv(RESULTS / "baselines_predictions.csv", index=False)

    def row(name, m):
        lo, hi = m["roc_auc_ci"]
        return (f"{name:42s} {m['roc_auc']:.3f} [{lo:.2f}-{hi:.2f}]   {m['roc_auc_by_group']:.3f}"
                f"      {m['avg_precision']:.3f}     {m['precision_at_20']:.3f}")
    print(f"test: {len(test)} posts, {yte.sum()} YES ({yte.mean():.1%})\n")
    print(f"{'model':42s} ROC-AUC [95% CI]   by-group  avg-prec  prec@20%")
    for n, m in out["models"].items():
        print(row(n, m))
    print("\nchecks")
    for n, m in out["checks"].items():
        print(row(n, m))
    print(f"\nwrote {RESULTS / 'baselines.json'}")


if __name__ == "__main__":
    main()
