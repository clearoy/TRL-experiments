"""Run PolicyInduction on the SMP split and score it like the baselines.

Input: only `model_text` (caption, timing, music, video length, orientation,
resolution and the text record of the video), shown to the LLM as one block per
post. Nothing about the account is an input. Label: YES = top 20% of popularity
within the post's user group.

Policies are written and merged by an OpenAI model (default gpt-6.1-sol) and
scored by Jev (the library default, reached with OPENROUTER_API_KEY or
TYPESAFE_API_KEY). Fit and predict checkpoint into --outdir, so re-running the
same command resumes.

Run:
    python smp/run_policy_induction.py --dry-run
    python smp/run_policy_induction.py --outdir run01
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))


def _load_env() -> None:
    """Export the library repo's .env (API keys) without overriding the shell."""
    env = REPO_ROOT / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


_load_env()

from baselines import metrics  # noqa: E402
from think_reason_learn.core.llms import OpenAIChoice  # noqa: E402
from think_reason_learn.policy_induction import (  # noqa: E402
    PolicyInduction,
    WeightTrainerConfig,
)

SPLIT = HERE / "data" / "split"
RESULTS = HERE / "results"

TASK_DESCRIPTION = (
    "Predict whether a short-form social media video post was one of the most "
    "popular posts compared with other posts from accounts of a similar size. "
    "Each sample is one post from a US, English-language account, shown as text: "
    "the creator's caption, when it was posted (UTC), the music or sound used, "
    "the video's length, orientation and resolution, and a detailed description "
    "of the video itself (what happens, the opening seconds, people, setting, "
    "filming, editing, audio, tone, a scene-by-scene timeline, the spoken words "
    "and the on-screen text). The size of the posting account is not shown and "
    "should not be guessed: the question is which content does better than other "
    "content from similar accounts. About 20% of posts are YES. Answer YES if the "
    "post is likely to be in the top 20% of its account-size group, NO otherwise."
)


def trl_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def load_split():
    train = pd.read_parquet(SPLIT / "train.parquet")
    test = pd.read_parquet(SPLIT / "test.parquet")
    # One input column. Named "post" so each sample renders as "post: Caption: …".
    X_train = train[["model_text"]].rename(columns={"model_text": "post"})
    X_test = test[["model_text"]].rename(columns={"model_text": "post"})
    return X_train, train.label.tolist(), X_test, test


async def main_async(args: argparse.Namespace) -> None:
    X_train, y_train, X_test, test = load_split()
    outdir = HERE / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"train: {len(X_train)} posts ({y_train.count('YES')} YES) | test: {len(X_test)} posts")

    pi = PolicyInduction(
        gen_llmc=[OpenAIChoice(model=args.gen_model)],
        # decision_llmc omitted: Jev, the library default.
        config=WeightTrainerConfig(beta=args.beta),
        gen_batches=((args.pos_per_batch, args.neg_per_batch), args.n_gen_batches),
        max_raw_policies=args.max_raw_policies,
        save_path=outdir,
        name="smp_policy_induction",
        confirm_requests=False,
        random_state=args.seed,
    )
    pi.decision_llmc.max_cost_usd = args.max_cost_usd

    if args.dry_run:
        # set_task() asks the LLM to write the generator prompt, which is a
        # billable call, so a dry run sets a stub instead.
        pi._task_description = TASK_DESCRIPTION
        pi._pgen_instructions_template = "STUB (dry run)."
        pi._set_data(X_train, y_train)
        pi._batch_indices()  # raises if there are too few YES/NO rows
        est = pi._estimate_fit_requests()
        print("\nEstimated fit requests (no calls made):")
        for label, count in est.items():
            print(f"  {label}: ~{count}")
        n_pred = len(X_test) * pi._jev_requests_per_sample(args.max_raw_policies)
        print(f"  jev (predict, upper bound): ~{n_pred}")
        lens = X_train.post.str.len() / 4
        print(f"  post length ~tokens: median {lens.median():.0f}, max {lens.max():.0f}")
        print(f"  Jev cost cap per fit()/predict(): ${args.max_cost_usd}")
        return

    await pi.set_task(task_description=TASK_DESCRIPTION)
    await pi.fit(X_train, y_train)
    pi.save()
    print(f"model saved to {outdir}")

    rows, vectors = [], []
    last = None
    async for sample_index, vector, prediction, token_counter in pi.predict(X_test):
        rows.append({"sample_index": sample_index, "prediction": prediction})
        vectors.append(vector)
        last = token_counter
    V = np.array(vectors, dtype=float)
    pred = pd.DataFrame(rows)
    pred["probability"] = pi.lr.predict_proba(V)[:, 1]
    for j in range(V.shape[1]):
        pred[f"policy_{j}"] = V[:, j]
    pred = pred.set_index("sample_index").sort_index()
    if len(pred) < len(X_test):
        print(f"WARNING: {len(X_test) - len(pred)} test posts skipped (no Jev answers); excluded below.")

    scored = test.loc[pred.index]
    y = (scored.label == "YES").to_numpy().astype(int)
    out = metrics(y, pred.probability.to_numpy(), scored.user_group.to_numpy())
    yhat = (pred.prediction == "YES").to_numpy().astype(int)
    out.update({
        "threshold": float(pi.threshold),
        "accuracy": float((yhat == y).mean()),
        "precision": float(y[yhat == 1].mean()) if yhat.sum() else 0.0,
        "recall": float(yhat[y == 1].mean()),
        "n_scored": int(len(pred)), "n_test": int(len(X_test)), "n_train": int(len(X_train)),
        "trl_commit": trl_commit(),
        "config": {
            "gen_model": args.gen_model, "decision_model": pi.decision_llmc.model,
            "gen_batches": [[args.pos_per_batch, args.neg_per_batch], args.n_gen_batches],
            "max_raw_policies": args.max_raw_policies, "beta": args.beta, "seed": args.seed,
        },
        "policies": pi.policies,
        "weights": [float(w) for w in pi.lr.coef_[0]],
        "validation_result": pi.validation_result,
        "fit_token_usage": pi.token_usage.to_dict(),
        "predict_token_usage": last.to_dict() if last else None,
    })
    pred.insert(0, "pid", scored.pid.to_numpy())
    pred.insert(1, "user_group", scored.user_group.to_numpy())
    pred.insert(2, "y_true", y)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"policy_induction_{args.outdir}.json").write_text(json.dumps(out, indent=2, default=str))
    pred.to_csv(RESULTS / f"policy_induction_{args.outdir}_predictions.csv")

    lo, hi = out["roc_auc_ci"]
    print(f"\nPolicyInduction ({args.outdir}) on {out['n_scored']} test posts")
    print(f"  ROC-AUC {out['roc_auc']:.3f} [{lo:.2f}-{hi:.2f}] | by group {out['roc_auc_by_group']:.3f} | "
          f"avg precision {out['avg_precision']:.3f} | precision@20% {out['precision_at_20']:.3f}")
    print(f"  threshold {out['threshold']:.2f}: precision {out['precision']:.3f}, recall {out['recall']:.3f}")
    print(f"  policies used: {int(np.count_nonzero(out['weights']))} of {len(out['policies'])}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gen-model", default="gpt-6.1-sol")
    p.add_argument("--pos-per-batch", type=int, default=5)
    p.add_argument("--neg-per-batch", type=int, default=5)
    p.add_argument("--n-gen-batches", type=int, default=10)
    p.add_argument("--max-raw-policies", type=int, default=30)
    p.add_argument("--beta", type=float, default=1.0, help="F-beta used to pick the decision threshold.")
    p.add_argument("--max-cost-usd", type=float, default=10.0, help="Jev cost cap per fit()/predict().")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="run01")
    p.add_argument("--dry-run", action="store_true")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
