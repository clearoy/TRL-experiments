"""Run RRF (Random Rule Forest) on the SMP split and score it like the baselines.

Same setup as run_policy_induction.py: input is only `model_text`, shown as one
block per post; the label is YES = top 20% of popularity within the post's user
group. An OpenAI model (default gpt-6.1-sol) writes YES/NO questions about the
posts, Jev (the library default) answers every question for every post, and an
elastic-net logistic regression combines the answers (RRF's default).

Run:
    python smp/run_rrf.py --outdir rrf01
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

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
from think_reason_learn.rrf import RRF  # noqa: E402

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
    "content from similar accounts. About 20% of posts are YES. Write YES/NO "
    "questions about a post whose answers help tell the top 20% of its "
    "account-size group (YES) from the rest (NO)."
)


def trl_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


async def main_async(args: argparse.Namespace) -> None:
    train = pd.read_parquet(SPLIT / "train.parquet")
    test = pd.read_parquet(SPLIT / "test.parquet")
    X_train = train[["model_text"]].rename(columns={"model_text": "post"})
    X_test = test[["model_text"]].rename(columns={"model_text": "post"})
    y_train = train.label.tolist()
    outdir = HERE / args.outdir
    print(f"train: {len(X_train)} posts ({y_train.count('YES')} YES) | test: {len(X_test)} posts")

    rrf = RRF(
        qgen_llmc=[OpenAIChoice(model=args.gen_model)],
        # qanswer_llmc omitted: Jev, the library default.
        max_generated_questions=args.max_questions,
        max_samples_as_context=args.samples_per_round,
        class_ratio=(1.0, 1.0),
        save_path=outdir,
        name="smp_rrf",
        random_state=args.seed,
    )
    await rrf.set_task(task_description=TASK_DESCRIPTION)
    await rrf.fit(X_train, y_train)
    rrf.save()
    print(f"model saved to {outdir}")

    res = await rrf.predict_founder_level(X_test)
    res = res.reindex(X_test.index)
    y = (test.label == "YES").to_numpy().astype(int)
    out = metrics(y, res.probability.to_numpy(), test.user_group.to_numpy())
    yhat = (res.prediction == "YES").to_numpy().astype(int)
    q = rrf.get_questions()
    out.update({
        "threshold": float(res.threshold.iloc[0]),
        "accuracy": float((yhat == y).mean()),
        "precision": float(y[yhat == 1].mean()) if yhat.sum() else 0.0,
        "recall": float(yhat[y == 1].mean()),
        "n_test": int(len(X_test)), "n_train": int(len(X_train)),
        "unanswered_test_posts": int((res.n_answered < res.n_answered.max()).sum()),
        "trl_commit": trl_commit(),
        "config": {"gen_model": args.gen_model, "max_questions": args.max_questions,
                   "samples_per_round": args.samples_per_round, "seed": args.seed,
                   "aggregation": rrf.aggregation_method},
        "n_questions": int(len(q)),
        "n_active_questions": int(q["exclusion"].isna().sum()) if "exclusion" in q else None,
    })
    pred = pd.DataFrame({"pid": test.pid, "user_group": test.user_group, "y_true": y,
                         "prediction": res.prediction.to_numpy(), "probability": res.probability.to_numpy()})
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"rrf_{args.outdir}.json").write_text(json.dumps(out, indent=2, default=str))
    pred.to_csv(RESULTS / f"rrf_{args.outdir}_predictions.csv", index=False)
    q.to_csv(RESULTS / f"rrf_{args.outdir}_questions.csv")

    lo, hi = out["roc_auc_ci"]
    print(f"\nRRF ({args.outdir}) on {len(X_test)} test posts")
    print(f"  ROC-AUC {out['roc_auc']:.3f} [{lo:.2f}-{hi:.2f}] | by group {out['roc_auc_by_group']:.3f} | "
          f"avg precision {out['avg_precision']:.3f} | precision@20% {out['precision_at_20']:.3f}")
    print(f"  threshold {out['threshold']:.2f}: precision {out['precision']:.3f}, recall {out['recall']:.3f}")
    print(f"  questions: {out['n_active_questions']} active of {out['n_questions']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gen-model", default="gpt-6.1-sol")
    p.add_argument("--max-questions", type=int, default=100)
    p.add_argument("--samples-per-round", type=int, default=30, help="Posts shown per question-generation call.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="rrf01")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
