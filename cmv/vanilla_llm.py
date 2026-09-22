"""Vanilla LLM baseline on the CMV pair task.

No policy induction, no rules, no batch-scoring machinery. One direct question
per pair -- "which reply is more persuasive" -- asked once in each order and
averaged, on gemini-3.5-flash. This is the plainest thing an LLM can do on this
task, with nothing about "policies" anywhere in the prompt.

Contrast with the deleted control (see cmv/README.md "Removed runs"): that
control ran through PolicyInduction's own scoring machinery -- the library's
POLICY_PREDICT_INSTRUCTIONS, framed as "classify this sample against this
policy" -- with one hand-written policy standing in for induced rules. This
script skips PolicyInduction entirely.

Task-description text is reused from run_policy_induction.py's
TASK_DESCRIPTION_PAIRED, but with the "useful comparisons" paragraph removed --
that paragraph was written to steer policy generation and has no place in a
vanilla baseline.

Run:
    python experiments/cmv/vanilla_llm.py --dry-run
    python experiments/cmv/vanilla_llm.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_ENV = REPO_ROOT / ".env"
if _ENV.exists():
    for _line in _ENV.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))

from think_reason_learn.core.llms import GoogleChoice  # noqa: E402

from cache_llm import CachingLLM  # noqa: E402
from run_policy_induction import build_pair_frame, swap_frame  # noqa: E402

TASK_FACTS = (
    "On the /r/ChangeMyView subreddit, a user posts a view they hold and invites "
    "others to change it. If a reply succeeds in changing the poster's mind, the "
    "poster awards it a delta.\n\n"
    "Each sample contains:\n"
    "- op_title and op_view: the original poster's stated view and reasoning.\n"
    "- argument_A and argument_B: two different replies arguing against that "
    "view. Exactly one of the two earned a delta; the other did not."
)

INSTRUCTIONS = """\
You are a careful, deterministic reader of online arguments.

Given the original poster's stated view and two replies arguing against it,
decide which reply is the one that changed the poster's mind and earned a
delta.

Base your decision only on the content of the two replies and the original
post. Be deterministic: the same input must always yield the same output.
Do not explain your reasoning.

Output format:
Return only which reply is more persuasive: A or B.
"""


class Verdict(BaseModel):
    winner: Literal["A", "B"]


def render(row: pd.Series) -> str:
    return (
        f"{TASK_FACTS}\n\n"
        f"op_title: {row['op_title']}\n"
        f"op_view: {row['op_view']}\n"
        f"argument_A: {row['argument_A']}\n"
        f"argument_B: {row['argument_B']}"
    )


async def ask(llm: CachingLLM, model: str, row: pd.Series, sem: asyncio.Semaphore) -> str:
    async with sem:
        for attempt in range(4):
            try:
                r = await llm.respond(
                    query=render(row),
                    llm_priority=[GoogleChoice(model=model)],
                    response_format=Verdict,
                    instructions=INSTRUCTIONS,
                    temperature=0.0,
                )
                # The library can silently return response="" (a plain str)
                # instead of a Verdict or None, when Gemini's structured-output
                # field comes back empty under degraded/high-load conditions --
                # `isinstance` check needed because `"" is not None` is True and
                # `.winner` on a str crashes with AttributeError.
                if isinstance(r.response, Verdict):
                    return r.response.winner
            except Exception:
                if attempt == 3:
                    raise
            await asyncio.sleep(2 * (attempt + 1))
    return "A"  # unreachable; keeps type-checkers happy


async def score_all(
    llm: CachingLLM, model: str, X: pd.DataFrame, concurrency: int
) -> pd.Series:
    sem = asyncio.Semaphore(concurrency)
    tasks = {idx: asyncio.ensure_future(ask(llm, model, row, sem)) for idx, row in X.iterrows()}
    done = 0
    total = len(tasks)
    out = {}
    for idx, t in tasks.items():
        out[idx] = await t
        done += 1
        if done % 50 == 0 or done == total:
            print(f"  {done}/{total}", end="\r", flush=True)
    print()
    return pd.Series(out)


async def main_async(args: argparse.Namespace) -> None:
    heldout = pd.read_parquet(DATA_DIR / "units_heldout.parquet")
    X, y, a_is_pos = build_pair_frame(heldout, args.condition, args.op_words, args.seed)
    if args.n_eval_pairs:
        X, y = X.iloc[: args.n_eval_pairs], y[: args.n_eval_pairs]
        a_is_pos = a_is_pos.iloc[: args.n_eval_pairs]

    print(f"condition : {args.condition}")
    print(f"model     : {args.model}")
    print(f"eval      : {len(X)} pairs")

    if args.dry_run:
        n = len(X) * (2 if args.swap_eval else 1)
        print(f"\nEstimated requests (no calls made): ~{n}")
        print("Dry run complete. Drop --dry-run to execute.")
        return

    cache = CachingLLM(enabled=not args.no_cache)
    print("\nscoring (raw order)...")
    raw = await score_all(cache, args.model, X, args.concurrency)

    swapped = None
    if args.swap_eval:
        print("scoring (swapped order)...")
        X_sw = swap_frame(X)
        swapped = await score_all(cache, args.model, X_sw, args.concurrency)

    truth = pd.Series(y, index=X.index).eq("YES")  # True where argument_A won
    prob_raw = raw.eq("A").astype(float)
    if swapped is not None:
        # In the swapped call, "A" is the ORIGINAL argument_B. So p(original A
        # wins) from that call is 1 - p(swapped-A wins).
        prob_swapped_as_original_A = 1.0 - swapped.eq("A").astype(float)
        prob = (prob_raw + prob_swapped_as_original_A) / 2.0
    else:
        prob = prob_raw

    tied = prob == 0.5
    pred_a = prob > 0.5
    decided = ~tied

    metrics = {
        "condition": args.condition,
        "model": args.model,
        "seed": args.seed,
        "op_words": args.op_words,
        "swap_eval": bool(args.swap_eval),
        "n_pairs_scored": int(len(X)),
        "n_ties": int(tied.sum()),
        "pairwise_accuracy": float(
            np.where(tied, 0.5, (pred_a == truth).astype(float)).mean()
        ),
        "accuracy_on_decided": float((pred_a[decided] == truth[decided]).mean())
        if decided.any() else float("nan"),
        "pred_A_rate": float(pred_a[decided].mean()) if decided.any() else float("nan"),
        "cache": cache.stats,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"vanilla_{args.condition}_{args.model.replace('gemini-','').replace('.','')}_seed{args.seed}"
    (RESULTS_DIR / f"{tag}.json").write_text(json.dumps(metrics, indent=2))

    scores = pd.DataFrame(
        {
            "pair_id": X.index,
            "a_is_positive": a_is_pos.loc[X.index].to_numpy(),
            "label": np.where(truth, "YES", "NO"),
            "probability": prob.to_numpy(),
            "predicted": np.where(tied, "-", np.where(pred_a, "A", "B")),
        }
    )
    if swapped is not None:
        scores["raw_winner"] = raw.to_numpy()
        scores["swapped_winner"] = swapped.to_numpy()
    scores.to_csv(RESULTS_DIR / f"{tag}_scores.csv", index=False)

    print(f"\n=== {tag} ===")
    print(f"  pairwise accuracy : {metrics['pairwise_accuracy']:.4f}")
    n_t = metrics["n_ties"]
    print(f"  undecided (=0.50) : {n_t} pairs = {n_t / max(len(X),1):.0%}")
    print(f"  acc when decided  : {metrics['accuracy_on_decided']:.4f}")
    print(f"  predicted A rate  : {metrics['pred_A_rate']:.1%}  (0.5 = unbiased)")
    print(f"  cache             : {cache.stats}")
    cache.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", default="root_reply",
                    choices=("root_reply", "full_path", "root_truncated"))
    p.add_argument("--model", default="gemini-3.5-flash")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--op-words", type=int, default=300)
    p.add_argument("--n-eval-pairs", type=int, default=0, help="0 = all 807.")
    p.add_argument("--no-swap-eval", dest="swap_eval", action="store_false",
                    help="Score each pair once instead of in both orders.")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
