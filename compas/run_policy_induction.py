"""Run PolicyInduction on the COMPAS split and score it like the baselines.

Reads the same train/test parquet files baselines.py uses, so the two are
directly comparable. Structured rows are handed to PolicyInduction as-is --
`_render_sample()` flattens each row to `column: value` lines, so no prose
conversion is needed, and the model sees the same information the sklearn
baselines do.

Fit and predict both checkpoint into --outdir, so an interrupted run resumes
where it stopped when you re-run the same command.

Run:
    python experiments/compas/run_policy_induction.py --dry-run
    python experiments/compas/run_policy_induction.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# think_reason_learn is not pip-installed in this venv; it resolves only when
# the repo root is on sys.path. Drop this once `make install` has been run.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from think_reason_learn.core.llms import GoogleChoice  # noqa: E402
from think_reason_learn.policy_induction import (  # noqa: E402
    PolicyInduction,
    WeightTrainerConfig,
)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
TARGET = "two_year_recid"

TASK_DESCRIPTION = (
    "Predict whether a criminal defendant will be arrested for another offense "
    "within two years of their COMPAS screening date. Each sample describes one "
    "defendant from Broward County, Florida, with these fields:\n"
    "- sex, age_years, age_group: demographics at screening time.\n"
    "- prior_offense_count: how many prior offenses are on their adult record.\n"
    "- juvenile_felony_convictions, juvenile_misdemeanor_convictions, "
    "juvenile_other_offenses: offenses committed as a minor. These are zero for "
    "most defendants; a non-zero value is uncommon and notable.\n"
    "- current_charge_severity: whether the charge that brought them in is a "
    "Felony (more serious) or a Misdemeanor (less serious).\n"
    "Answer YES if the defendant is likely to be arrested again within two "
    "years, NO otherwise. This is a research benchmark built from a public "
    "dataset released by ProPublica."
)


def trl_commit() -> str:
    """The library commit this result was produced with.

    The library and these experiments live in separate repos, so a result
    without this stamp cannot be tied back to the code that produced it.
    """
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def load_split(n_train: int, n_test: int):
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")
    features = [c for c in train.columns if c not in (TARGET, "label")]

    if n_train:
        train = train.head(n_train)
    if n_test:
        test = test.head(n_test)

    X_train = train[features]
    y_train = train["label"].tolist()
    X_test = test[features]
    return X_train, y_train, X_test, test


def score(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f0.5": float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


async def main_async(args: argparse.Namespace) -> None:
    X_train, y_train, X_test, test_df = load_split(args.n_train, args.n_test)
    outdir = HERE / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"train: {len(X_train)} rows ({y_train.count('YES')} YES)")
    print(f"test:  {len(X_test)} rows")
    print(f"features: {list(X_train.columns)}")

    pi = PolicyInduction(
        gen_llmc=[GoogleChoice(model=args.gen_model)],
        predict_llmc=[GoogleChoice(model=args.predict_model)],
        # beta=0.5, not 1.0. F-beta ignores true negatives entirely, which is
        # the wrong family for a ~45% positive dataset where correctly
        # identifying non-recidivists matters as much as identifying
        # recidivists. F1 in particular is maximised by over-predicting the
        # positive class (trivial always-YES scores F1=0.625 here), which drove
        # an earlier run to predict YES on 89% of samples. Measured offline on
        # the saved policy scores, beta=0.5 reproduces accuracy-optimal and
        # balanced-accuracy-optimal threshold selection exactly, so it buys the
        # correct operating point without needing to modify _fit_weights.
        config=WeightTrainerConfig(beta=0.5, penalty="l1"),
        max_policy_length=args.max_policies,
        class_ratio=(1.0, 1.0),
        max_samples_as_context=args.samples_per_batch,
        max_gen_batches=args.max_gen_batches,
        policy_batch_size=args.policy_batch_size,
        llm_semaphore_limit=args.concurrency,
        save_path=outdir,
        name="compas_policy_induction",
        confirm_requests=False,
        random_state=0,
    )

    if args.dry_run:
        # set_task() normally asks the LLM to write the generation template,
        # which is itself a billable call -- so a dry run must supply its own
        # stub template to stay at genuinely zero cost.
        await pi.set_task(
            task_description=TASK_DESCRIPTION,
            instructions_template="STUB (dry run). Max <max_policy_length> policies.",
        )
        pi._set_data(X_train, y_train)
        est = pi._estimate_fit_requests()
        print("\nEstimated fit requests (no calls made):")
        for label, count in est.items():
            print(f"  {label}: ~{count}")
        n_predict = len(X_test) * -(-args.max_policies // args.policy_batch_size)
        print(f"  {args.predict_model} (predict, upper bound): ~{n_predict}")
        print(f"  TOTAL upper bound: ~{sum(est.values()) + n_predict} requests")
        print("\nDry run complete. Drop --dry-run to execute.")
        return

    await pi.set_task(task_description=TASK_DESCRIPTION)

    await pi.fit(X_train, y_train)
    pi.save()
    print(f"\nModel saved to {outdir}")

    # Keep the policy vector, not just the label: it is what the decision
    # threshold is applied to, so discarding it makes threshold analysis and
    # ROC-AUC impossible after the fact.
    rows, vectors = [], []
    async for sample_index, vector, prediction, token_counter in pi.predict(X_test):
        rows.append({"sample_index": sample_index, "prediction": prediction})
        vectors.append(vector)
        last_counter = token_counter

    V = np.array(vectors, dtype=float)
    pred_df = pd.DataFrame(rows)
    # Probability behind each decision, recovered from the fitted LR.
    pred_df["probability"] = pi.lr.predict_proba(V)[:, 1]
    pred_df["threshold"] = pi.threshold
    # One column per policy (its 1/0 answer for this sample). Attached before
    # the sort so the vectors stay row-aligned with their sample_index.
    for j, name in enumerate(pi._feature_order_):
        pred_df[f"policy_{name}"] = V[:, j]
    pred_df = pred_df.set_index("sample_index").sort_index()
    if len(pred_df) < len(X_test):
        # predict() skips samples where no policy answer could be obtained.
        print(
            f"WARNING: {len(X_test) - len(pred_df)} of {len(X_test)} samples were "
            "skipped (no policy answers). They are excluded from the metrics below."
        )

    y_true = test_df.loc[pred_df.index, TARGET].tolist()
    y_pred = [1 if p == "YES" else 0 for p in pred_df["prediction"]]
    prob = pred_df["probability"].to_numpy()
    metrics = score(y_true, y_pred)
    metrics["threshold"] = float(pi.threshold)
    # Threshold-independent, so this is the metric directly comparable to the
    # sklearn baselines regardless of where either model's cut point sits.
    metrics["roc_auc"] = float(roc_auc_score(y_true, prob))
    # Full sweep so the operating point can be re-chosen without re-running.
    # NOTE: picking a threshold off this table means picking it on test data,
    # which is optimistically biased. Use a validation split for any reported
    # number; this is a diagnostic.
    metrics["threshold_sweep"] = [
        {"threshold": round(float(t), 2), **score(y_true, (prob >= t).astype(int))}
        for t in np.arange(0.05, 0.96, 0.05)
    ]
    metrics["n_scored"] = len(pred_df)
    metrics["trl_commit"] = trl_commit()
    metrics["config"] = {
        "gen_model": args.gen_model,
        "predict_model": args.predict_model,
        "max_policies": args.max_policies,
        "samples_per_batch": args.samples_per_batch,
        "max_gen_batches": args.max_gen_batches,
        "policy_batch_size": args.policy_batch_size,
        "beta": pi.config.beta,
        "penalty": pi.config.penalty,
    }
    metrics["n_test"] = len(X_test)
    metrics["n_train"] = len(X_train)
    metrics["validation_result"] = pi.validation_result
    metrics["fit_token_usage"] = pi.token_usage.to_dict()
    # Predict-time usage is a separate counter that the library does not persist;
    # capture it here so cost is recoverable after the fact.
    metrics["predict_token_usage"] = last_counter.to_dict()

    # Keyed by --outdir so repeated runs accumulate side by side instead of
    # overwriting each other. Repetition matters here: identical configs have
    # been seen to differ by ~0.08 precision on LLM sampling noise alone.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS_DIR / f"policy_induction_{args.outdir}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    pred_df.to_csv(RESULTS_DIR / f"policy_induction_{args.outdir}_predictions.csv")

    print("\n=== PolicyInduction on COMPAS ===")
    for k in ("accuracy", "f1", "f0.5", "precision", "recall", "roc_auc"):
        print(f"  {k:12s} {metrics[k]:.4f}")
    print(f"  {'threshold':12s} {metrics['threshold']:.4f}")
    print(f"\nwrote {metrics_path}")
    print(f"wrote {RESULTS_DIR}/policy_induction_{args.outdir}_predictions.csv")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument(
        "--n-test",
        type=int,
        default=0,
        help="Test rows to evaluate on; 0 (default) = the full test set. Must "
        "match what baselines.py used or the comparison is invalid.",
    )
    p.add_argument("--gen-model", default="gemini-3.5-flash")
    p.add_argument("--predict-model", default="gemini-2.5-flash-lite")
    p.add_argument("--max-policies", type=int, default=20)
    p.add_argument("--samples-per-batch", type=int, default=10)
    p.add_argument("--max-gen-batches", type=int, default=7)
    p.add_argument("--policy-batch-size", type=int, default=10)
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--outdir", default="run01")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the estimated request count and exit without scoring.",
    )
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
