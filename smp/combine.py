"""Combine a PolicyInduction run's policy scores with the TF-IDF baselines.

No LLM calls. Training policy scores come from <run>/policy_scores.parquet
(Jev's P(YES) per policy for every training post); test scores come from
results/policy_induction_<run>_predictions.csv, where only the policies the
fitted model kept were scored. The combination therefore uses those kept
policies, on both sides.

Compares, on the same test posts:
  tfidf               TF-IDF(model_text) + log video length      (the baseline)
  policies            the kept policy scores only
  tfidf + policies    both together
with a paired bootstrap of the AUC difference against the baseline.

Run:
    python smp/combine.py --run run01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from baselines import SEED, metrics, text_plus_length, tfidf

HERE = Path(__file__).resolve().parent
SPLIT = HERE / "data" / "split"
RESULTS = HERE / "results"


def paired_diff(y, a, b, n=2000):
    """Bootstrap the AUC difference b - a on the same resampled posts."""
    rng = np.random.default_rng(SEED)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            d.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="run01")
    args = p.parse_args()

    train = pd.read_parquet(SPLIT / "train.parquet")
    test = pd.read_parquet(SPLIT / "test.parquet")
    run_data = pd.read_parquet(HERE / args.run / "data.parquet")
    assert (run_data.post.to_numpy() == train.model_text.to_numpy()).all(), "train rows out of order"

    scores_tr = pd.read_parquet(HERE / args.run / "policy_scores.parquet")
    pred = pd.read_csv(RESULTS / f"policy_induction_{args.run}_predictions.csv").set_index("sample_index").sort_index()
    meta = json.loads((RESULTS / f"policy_induction_{args.run}.json").read_text())
    kept = [j for j, w in enumerate(meta["weights"]) if w != 0]
    test = test.loc[pred.index]  # posts the run scored

    P_tr = scores_tr[[str(j) for j in kept]].to_numpy(float)
    P_te = pred[[f"policy_{j}" for j in kept]].to_numpy(float)
    sc = StandardScaler().fit(P_tr)
    P_tr, P_te = sc.transform(P_tr), sc.transform(P_te)

    ytr = (train.label == "YES").to_numpy().astype(int)
    yte = (test.label == "YES").to_numpy().astype(int)
    g = test.user_group.to_numpy()

    Xtr, Xte = text_plus_length(tfidf(None), train, test, "model_text")
    Xtr_c = hstack([Xtr, csr_matrix(P_tr)]).tocsr()
    Xte_c = hstack([Xte, csr_matrix(P_te)]).tocsr()
    lr = lambda: LogisticRegression(class_weight="balanced", max_iter=5000)

    s = {
        "tfidf (logistic)": lr().fit(Xtr, ytr).predict_proba(Xte)[:, 1],
        "policies only (logistic)": lr().fit(P_tr, ytr).predict_proba(P_te)[:, 1],
        "PolicyInduction as fitted": pred.probability.to_numpy(),
        "tfidf + policies (logistic)": lr().fit(Xtr_c, ytr).predict_proba(Xte_c)[:, 1],
    }
    Xtr5, Xte5 = text_plus_length(tfidf(5000), train, test, "model_text")
    rf = lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample",
                                        n_jobs=-1, random_state=SEED)
    s["tfidf (random forest)"] = rf().fit(Xtr5, ytr).predict_proba(Xte5)[:, 1]
    s["tfidf + policies (random forest)"] = rf().fit(hstack([Xtr5, csr_matrix(P_tr)]).tocsr(), ytr).predict_proba(
        hstack([Xte5, csr_matrix(P_te)]).tocsr())[:, 1]

    out = {"run": args.run, "kept_policies": kept, "n_test": int(len(test)), "results": {}}
    base = s["tfidf (logistic)"]
    print(f"{args.run}: {len(kept)} kept policies, {len(test)} test posts\n")
    print(f"{'model':36s} ROC-AUC [95% CI]   by-group  prec@20%  vs tfidf logistic (paired)")
    for name, score in s.items():
        m = metrics(yte, score, g)
        d = paired_diff(yte, base, score)
        m["diff_vs_tfidf_logistic"] = d
        out["results"][name] = m
        lo, hi = m["roc_auc_ci"]
        print(f"{name:36s} {m['roc_auc']:.3f} [{lo:.2f}-{hi:.2f}]   {m['roc_auc_by_group']:.3f}     "
              f"{m['precision_at_20']:.3f}    {d[0]:+.3f} [{d[1]:+.2f}, {d[2]:+.2f}]")
    (RESULTS / f"combined_{args.run}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
