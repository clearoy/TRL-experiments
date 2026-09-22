"""PolicyInduction on the CMV pair task.

Two formulations, selected with --formulation:

  paired    (default) One row per PAIR. The sample carries both replies as
            argument_A and argument_B; the label is YES when argument_A is the
            one that earned the delta. Which reply lands in slot A is
            randomised, exactly 50/50, so position carries no information.
            Accuracy is read off directly -- no pointwise->pairwise step.
            This is the formulation Tan et al. 2016 used.

  pointwise One row per path-unit (label delta/no_delta), fitted independently,
            then evaluated pairwise by scoring both members of a heldout pair
            and taking the higher. Kept so the published 0.5248 stays
            reproducible.

Paired is cheaper (807 eval calls, not 1,614) and lets a policy compare the two
replies directly, which pointwise structurally cannot. The risk it carries is
positional bias: --swap-eval scores every heldout pair in both orders and
averages, which cancels it. `pred_A_rate` in the metrics is the free version of
that check -- far from 0.5 means the model is answering by position.

--control replaces the induced rules with one fixed generic prompt. If the rule
set does not beat it, the induction added nothing.

Run:
    python experiments/cmv/run_policy_induction.py --dry-run
    python experiments/cmv/run_policy_induction.py --condition root_reply --seed 0
    python experiments/cmv/run_policy_induction.py --control --condition root_reply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# think_reason_learn's Settings uses env_file=".env", which pydantic-settings
# resolves relative to the CWD -- so running from anywhere but the library root
# silently yields empty keys and a misleading "GOOGLE_API_KEY not set" error
# (the setting is actually GOOGLE_AI_API_KEY). Export the library repo's .env
# into the environment before importing, so this works from any directory.
_ENV = REPO_ROOT / ".env"
if _ENV.exists():
    for _line in _ENV.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))

from sklearn.metrics import roc_auc_score  # noqa: E402

from think_reason_learn.core.llms import GoogleChoice  # noqa: E402
from think_reason_learn.policy_induction import (  # noqa: E402
    PolicyInduction,
    WeightTrainerConfig,
)
from cache_llm import CachingLLM  # noqa: E402

CONDITIONS = ("root_reply", "full_path", "root_truncated")

# ── paired formulation ──────────────────────────────────────────────────────────

# The library's label space is binary YES/NO -- _sample() selects on the literal
# string "YES" and _build_feature_matrix maps anything outside {yes,y,true,t,1}
# to 0, so literal "A"/"B" labels would collapse both classes to 0 and the fit
# would fail. YES therefore *means* A, which the task description states outright
# so the model only ever reasons in terms of A and B.
#
# This is the ONLY prompt written here. Every other prompt is the library
# default, including the policy generation template, which set_task() derives
# from this text alone. So the paragraph below is the sole lever on what the
# rules look like -- and it is also prepended to all ~2,600 scoring calls, which
# is why it stays short and says nothing a scorer would find confusing.
#
# The topic sentence carries the weight. The first attempt let the generator
# write its own rules and it produced topic taxonomies ("A is about economic
# motives whereas B is about ethical ones"), which are hidden conjunctions: 7 of
# 10 fired on under 16% of pairs and the set scored 0.5261. Stating that both
# replies always share a topic makes those rules visibly useless a priori.
TASK_DESCRIPTION_PAIRED = (
    "On the /r/ChangeMyView subreddit, a user posts a view they hold and invites "
    "others to change it. If a reply succeeds in changing the poster's mind, the "
    "poster awards it a delta.\n\n"
    "Each sample contains:\n"
    "- op_title and op_view: the original poster's stated view and reasoning.\n"
    "- argument_A and argument_B: two different replies arguing against that "
    "view. Exactly one of the two earned a delta; the other did not.\n\n"
    "Answer YES if argument_A is the one that earned the delta, NO if "
    "argument_B is.\n\n"
    "Both replies answer the same post, so they are always about the same "
    "subject and the subject itself can never separate them. What separates "
    "them is how each one argues: how directly it engages the poster's own "
    "stated reasoning, whether it concedes anything before disagreeing, how "
    "hedged or absolute its claims are, whether it offers evidence or concrete "
    "examples, whether it asks the poster questions, and how confrontational "
    "its tone is. Useful comparisons say which reply does more of one such "
    "thing than the other. Which reply appears as A and which as B is "
    "randomised and carries no information. Length alone is not a reliable "
    "signal and should not drive the answer."
)


CONTROL_POLICY_PAIRED = (
    "Argument A is more persuasive than argument B, and is the one that changed "
    "the original poster's mind and earned a delta."
)

# ── pointwise formulation (kept for reproducibility) ────────────────────────────

TASK_DESCRIPTION_POINTWISE = (
    "On the /r/ChangeMyView subreddit, a user posts a view they hold and invites "
    "others to change it. If a reply succeeds in changing the poster's mind, the "
    "poster awards it a delta.\n\n"
    "Each sample contains:\n"
    "- op_title and op_text: the original poster's stated view and reasoning.\n"
    "- argument: one reply arguing against that view.\n\n"
    "Answer YES if this argument succeeded in changing the original poster's "
    "mind and earned a delta, NO if it did not. Judge how the argument is "
    "constructed and how it engages with the specific view in op_text. Argument "
    "length alone is not a reliable signal and should not drive the answer."
)

CONTROL_POLICY_POINTWISE = (
    "This argument is persuasive enough to change the original poster's mind "
    "and earn a delta."
)



def thl_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _op_columns(df: pd.DataFrame, index: pd.Index, op_words: int) -> pd.DataFrame:
    op = df.groupby("pair_id")[["op_title", "op_text"]].first().reindex(index)
    if op_words:
        op["op_text"] = op["op_text"].str.split().str[:op_words].str.join(" ")
    return op


def build_pair_frame(
    df: pd.DataFrame, condition: str, op_words: int, seed: int
) -> Tuple[pd.DataFrame, List[str], pd.Series]:
    """One row per pair, both replies present, A/B assignment randomised.

    Exactly half the pairs get the delta winner in slot A, so the label is 50/50
    by construction rather than by luck -- class_ratio=(1,1) generation and the
    F-beta threshold sweep both assume that balance.

    Returns (X, y, a_is_positive). y is "YES" where argument_A won the delta.
    """
    wide = df.pivot(index="pair_id", columns="side", values=f"arg_{condition}")
    wide = wide.dropna()
    op = _op_columns(df, wide.index, op_words)

    n = len(wide)
    a_is_pos = np.zeros(n, dtype=bool)
    a_is_pos[: n // 2] = True
    np.random.default_rng(seed).shuffle(a_is_pos)

    pos, neg = wide["positive"].to_numpy(), wide["negative"].to_numpy()
    X = pd.DataFrame(
        {
            "op_title": op["op_title"].to_numpy(),
            "op_view": op["op_text"].to_numpy(),
            "argument_A": np.where(a_is_pos, pos, neg),
            "argument_B": np.where(a_is_pos, neg, pos),
        },
        index=wide.index,
    )
    y = np.where(a_is_pos, "YES", "NO").tolist()
    return X, y, pd.Series(a_is_pos, index=wide.index, name="a_is_positive")


def swap_frame(X: pd.DataFrame) -> pd.DataFrame:
    """The same pairs with A and B exchanged. Column ORDER is preserved, so the
    rendered prompt is structurally identical and only the contents move."""
    out = X.copy()
    out["argument_A"], out["argument_B"] = X["argument_B"], X["argument_A"]
    return out


def sample_pairs(
    X: pd.DataFrame, y: List[str], n_pairs: int, seed: int
) -> Tuple[pd.DataFrame, List[str]]:
    """Stratified sample of whole pairs, keeping the label split exactly even.

    NESTED: the draw is a prefix of one seeded permutation per class, so a larger
    n_pairs is a strict superset of a smaller one at the same seed. Two
    consequences, both wanted. Every already-scored pair is served from the
    prompt cache when the training size is raised, so growing a run costs only
    the new pairs. And a training-size ladder then measures the effect of adding
    data rather than the effect of drawing a different sample.
    """
    if not n_pairs or n_pairs >= len(X):
        return X, y
    ya = np.array(y)
    rng = np.random.default_rng(seed)
    k = n_pairs // 2
    pick = np.concatenate(
        [
            rng.permutation(np.where(ya == "YES")[0])[:k],
            rng.permutation(np.where(ya == "NO")[0])[: n_pairs - k],
        ]
    )
    pick.sort()
    return X.iloc[pick], [y[i] for i in pick]


def build_unit_frame(df: pd.DataFrame, condition: str, op_words: int) -> pd.DataFrame:
    """Pointwise: one row per path-unit."""
    op = df["op_text"]
    if op_words:
        op = op.str.split().str[:op_words].str.join(" ")
    return pd.DataFrame(
        {"op_title": df["op_title"].values, "op_view": op.values,
         "argument": df[f"arg_{condition}"].values},
        index=df.index,
    )


def sample_units(train: pd.DataFrame, n_units: int, seed: int) -> pd.DataFrame:
    """Sample whole pairs so the pointwise training set is exactly 50/50."""
    pair_ids = train["pair_id"].drop_duplicates()
    n_pairs = max(1, n_units // 2)
    chosen = pair_ids.sample(n=min(n_pairs, len(pair_ids)), random_state=seed)
    return train[train["pair_id"].isin(chosen)].reset_index(drop=True)


def pairwise_accuracy(scores: pd.DataFrame) -> Dict[str, float]:
    """Pointwise mode: score both members of each pair, take the higher."""
    w = scores.pivot(index="pair_id", columns="side", values="probability").dropna()
    wins = (w["positive"] > w["negative"]).sum()
    ties = (w["positive"] == w["negative"]).sum()
    return {
        "pairwise_accuracy": float((wins + 0.5 * ties) / len(w)),
        "n_pairs_scored": int(len(w)),
        "n_ties": int(ties),
    }


async def _score_frame(pi: PolicyInduction, X: pd.DataFrame) -> Tuple[pd.Index, np.ndarray]:
    rows, vectors = [], []
    async for idx, vec, _pred, _tc in pi.predict(X):
        rows.append(idx)
        vectors.append(vec)
    return pd.Index(rows), np.array(vectors, dtype=float)


async def main_async(args: argparse.Namespace) -> None:
    train = pd.read_parquet(DATA_DIR / "units_train.parquet")
    heldout = pd.read_parquet(DATA_DIR / "units_heldout.parquet")
    paired = args.formulation == "paired"

    if paired:
        X_all, y_all, _ = build_pair_frame(train, args.condition, args.op_words, args.seed)
        X_train, y_train = sample_pairs(X_all, y_all, args.n_train_pairs, args.seed)
        if args.augment_swap:
            # Each pair a second time with A and B exchanged and the label
            # flipped. Nothing in the pipeline otherwise knows the task is
            # symmetric: policy answers need not flip when the slots do, and the
            # regression is free to learn a non-zero intercept. Showing both
            # orders removes both. Costs a second scoring call per pair, and
            # puts mirror images of one pair in different CV folds, so the
            # cross-validated C is mildly optimistic.
            # reset_index: the concat would otherwise carry each pair_id twice,
            # and the library reindexes scored predictions onto X.index, which
            # raises on duplicate labels.
            X_train = pd.concat([X_train, swap_frame(X_train)]).reset_index(drop=True)
            y_train = y_train + ["NO" if v == "YES" else "YES" for v in y_train]
        X_eval, y_eval, a_is_pos = build_pair_frame(
            heldout, args.condition, args.op_words, args.seed
        )
        if args.n_eval_pairs:
            X_eval, y_eval = X_eval.iloc[: args.n_eval_pairs], y_eval[: args.n_eval_pairs]
            a_is_pos = a_is_pos.iloc[: args.n_eval_pairs]
        task_description = TASK_DESCRIPTION_PAIRED
        control_policy = CONTROL_POLICY_PAIRED
        n_eval_pairs = len(X_eval)
    else:
        fit_on = sample_units(train, args.n_train_units, args.seed)
        X_train = build_unit_frame(fit_on, args.condition, args.op_words)
        y_train = fit_on["label"].tolist()
        if args.n_eval_pairs:
            keep = heldout["pair_id"].drop_duplicates().head(args.n_eval_pairs)
            heldout = heldout[heldout["pair_id"].isin(keep)].reset_index(drop=True)
        X_eval = build_unit_frame(heldout, args.condition, args.op_words)
        y_eval, a_is_pos = [], None
        task_description = TASK_DESCRIPTION_POINTWISE
        control_policy = CONTROL_POLICY_POINTWISE
        n_eval_pairs = heldout["pair_id"].nunique()

    # Every knob that has been observed to move a result goes in the tag, so two
    # runs that differ in any of them land in different directories and different
    # results files instead of one silently overwriting the other. Swapping the
    # scoring model alone moved accuracy 12.8pp here, and it used to be invisible
    # in the name.
    n_train = args.n_train_pairs if paired else args.n_train_units
    # Alphanumeric and underscores only: PolicyInduction rejects anything else in
    # `name`, so a hyphenated model slug crashes the run before the first call.
    slug = lambda mdl: "".join(
        c for c in mdl.replace("gemini-", "").replace("-preview", "") if c.isalnum()
    )
    tag = "_".join(
        [
            "control" if args.control else "rules",
            args.formulation,
            args.condition,
            f"n{n_train}",
            f"gen{slug(args.gen_model)}",
            f"pred{'then'.join(slug(m) for m in args.predict_model)}",
            f"seed{args.seed}",
        ]
        + (["aug"] if paired and args.augment_swap else [])
        + ([] if paired and args.swap_eval else ["noswap"])
    )
    outdir = HERE / "runs" / tag
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"formulation : {args.formulation}")
    print(f"condition   : {args.condition}")
    print(f"mode        : {'CONTROL (1 generic prompt, no induction)' if args.control else 'induced rules'}")
    print(f"train       : {len(X_train)} rows ({y_train.count('YES')} YES)")
    print(f"eval        : {len(X_eval)} rows ({n_eval_pairs} pairs)")

    cache = CachingLLM(enabled=not args.no_cache)
    pi = PolicyInduction(
        gen_llmc=[GoogleChoice(model=args.gen_model)],
        predict_llmc=[GoogleChoice(model=m) for m in args.predict_model],
        # beta=0.5, not 1.0. F-beta ignores true negatives, and on a 50/50 split
        # a trivial always-YES classifier scores F1=0.667 -- so beta=1.0 drove
        # the threshold to 0.01 (predict YES for everything). Same trap as the
        # COMPAS experiment. In paired mode the threshold reaches the headline
        # number directly, so this matters more here than it did pointwise.
        config=WeightTrainerConfig(beta=args.beta, penalty="l1", cv_folds=args.cv_folds),
        max_policy_length=args.max_policies,
        class_ratio=(1.0, 1.0),
        max_samples_as_context=args.samples_per_batch,
        max_gen_batches=args.max_gen_batches,
        policy_batch_size=args.policy_batch_size,
        llm_semaphore_limit=args.concurrency,
        save_path=outdir,
        name=f"cmv_{tag}",
        confirm_requests=False,
        random_state=args.seed,
        _llm=cache,
    )

    if args.dry_run:
        await pi.set_task(
            task_description=task_description,
            instructions_template="STUB (dry run). Max <max_policy_length> policies.",
        )
        pi._set_data(X_train, y_train)
        est = pi._estimate_fit_requests()
        n_pol = 1 if args.control else args.max_policies
        per_sample = -(-n_pol // args.policy_batch_size)
        n_pred = len(X_eval) * per_sample * (2 if (paired and args.swap_eval) else 1)
        print("\nEstimated requests (no calls made):")
        if args.control:
            # Control skips generation but still scores its one policy against
            # every training row before fitting -- omitting that understated the
            # control's cost by len(X_train) calls.
            n_fit = len(X_train) * per_sample
            print(f"  {args.gen_model} (generation): 0  (control: no induction)")
            print(f"  {' -> '.join(args.predict_model)} (scoring): ~{n_fit}")
            total = n_fit + n_pred
        else:
            for k, v in est.items():
                print(f"  {k}: ~{v}")
            total = n_pred + sum(est.values())
        print(f"  {' -> '.join(args.predict_model)} (predict): ~{n_pred}")
        print(f"  TOTAL: ~{total}")
        print("\nDry run complete. Drop --dry-run to execute.")
        return

    if args.control:
        # No generation, no induced rules: one fixed policy, and its raw YES-rate
        # becomes the score.
        await pi.set_task(
            task_description=task_description,
            instructions_template="Unused in control mode. <max_policy_length>",
        )
        pi._set_data(X_train, y_train)
        pi._policy_memory = pd.DataFrame(
            {"policy": [control_policy], "predictions": [None]}
        )
        await pi._score_policies()
        pi._fit_weights()
    else:
        # No instructions_template: set_task() asks the gen model to write
        # its own, from the task description alone. That is the library
        # default, and it costs one extra call to the gen model.
        await pi.set_task(task_description=task_description)
        await pi.fit(X_train, y_train)
    pi.save()

    # predict() resumes from this checkpoint unconditionally, and the checkpoint
    # stores feature vectors with no record of which policies produced them. A
    # leftover one from an earlier run would be replayed against a new rule set,
    # yielding numbers that belong to neither. Interrupted runs still restart
    # cheaply, because the prompt cache serves every call already made.
    for stale in (outdir / "predict_checkpoint.json",
                  outdir / "swap" / "predict_checkpoint.json"):
        stale.unlink(missing_ok=True)

    rows, V = await _score_frame(pi, X_eval)
    prob = pi.lr.predict_proba(V)[:, 1]

    if paired:
        if args.swap_eval:
            # Same pairs, A and B exchanged. p(A wins) from the swapped call is
            # 1 - p(swapped A wins); averaging the two cancels positional bias.
            #
            # predict() checkpoints on the sample index and resumes from that
            # checkpoint unconditionally. Re-running it over the same pair_ids
            # therefore returns the FIRST pass verbatim without issuing a single
            # call, and averaging a value with its own complement collapses
            # every probability to exactly 0.5. So the swapped pass gets both
            # its own index and its own save_path, and can never alias.
            X_sw = swap_frame(X_eval.loc[rows]).reset_index(drop=True)
            main_path, pi.save_path = pi.save_path, pi.save_path / "swap"
            pi.save_path.mkdir(parents=True, exist_ok=True)
            try:
                rows_s, V_s = await _score_frame(pi, X_sw)
            finally:
                pi.save_path = main_path
            # rows_s are positions into `rows`; predict() yields in completion
            # order, so map back to pair_id before combining.
            p_s = pd.Series(
                pi.lr.predict_proba(V_s)[:, 1], index=rows[np.asarray(rows_s)]
            )
            prob_swapped = p_s.reindex(rows).to_numpy()
            # A swapped sample that failed falls back to 1 - prob, leaving that
            # pair's averaged probability equal to its raw one.
            missing = np.isnan(prob_swapped)
            prob_swapped[missing] = 1.0 - prob[missing]
            prob_raw, prob = prob, (prob + (1.0 - prob_swapped)) / 2.0
            if np.ptp(prob) == 0.0:
                raise RuntimeError(
                    "Swapped pass returned probabilities identical to the raw "
                    "pass, so the average is constant at 0.5. The swap did not "
                    "actually re-score anything."
                )
        else:
            prob_raw, prob_swapped = prob, None

        truth = (pd.Series(y_eval, index=X_eval.index).loc[rows] == "YES").to_numpy()
        # 0.5, not the F-beta-tuned threshold. "A beats B" is a symmetric
        # question: swapping the two replies must flip the answer, and only a
        # 0.5 cut satisfies that. An F-beta threshold treats YES and NO
        # asymmetrically, which is coherent for a pointwise class-imbalance
        # problem and incoherent here. The fitted threshold is kept below as a
        # diagnostic only.
        pred_a = prob > 0.5
        # A probability of exactly 0.5 is a refusal to choose, not a vote for A.
        # It matters most in --control: one binary feature scored in both orders
        # yields a three-valued score, and 0.5 is precisely the case where the
        # model named the same slot both times, i.e. answered from position
        # rather than content. Counting those as correct half the time is the
        # expected value of a coin flip and matches baselines.py.
        tied = prob == 0.5
        scores = pd.DataFrame(
            {
                "pair_id": rows,
                "a_is_positive": a_is_pos.loc[rows].to_numpy(),
                "label": np.where(truth, "YES", "NO"),
                "probability": prob,
                "predicted": np.where(tied, "-", np.where(pred_a, "A", "B")),
                "correct": np.where(tied, 0.5, (pred_a == truth).astype(float)),
            }
        ).set_index("pair_id", drop=False)
        if prob_swapped is not None:
            scores["probability_raw"] = prob_raw
            scores["probability_swapped"] = prob_swapped
        for j, name in enumerate(pi._feature_order_):
            scores[f"policy_{name}"] = V[:, j]

        decided = ~tied
        metrics: Dict[str, object] = {
            "pairwise_accuracy": float(
                np.where(tied, 0.5, (pred_a == truth).astype(float)).mean()
            ),
            # The same accuracy over only the pairs the model actually chose on.
            # For --control this is the number that says whether a direct ask
            # carries signal, separated from how often it declines to answer.
            "accuracy_on_decided": float(
                (pred_a[decided] == truth[decided]).mean()
            ) if decided.any() else float("nan"),
            "accuracy_at_fitted_threshold": float(
                ((prob >= pi.threshold) == truth).mean()
            ),
            "roc_auc": float(roc_auc_score(truth, prob)),
            "n_pairs_scored": int(len(prob)),
            "n_ties": int(tied.sum()),
            # Far from 0.5 means the model is answering by slot, not by content.
            "pred_A_rate": float(pred_a[decided].mean()) if decided.any() else float("nan"),
            "swap_eval": bool(args.swap_eval),
        }
        if prob_swapped is not None:
            metrics["positional_bias"] = float(
                np.mean(prob_raw) - np.mean(1.0 - prob_swapped)
            )
    else:
        scores = heldout.loc[rows, ["pair_id", "side", "delta"]].copy()
        scores["probability"] = prob
        for j, name in enumerate(pi._feature_order_):
            scores[f"policy_{name}"] = V[:, j]
        metrics = dict(pairwise_accuracy(scores))
        metrics["pointwise_accuracy"] = float(
            (scores["probability"] >= pi.threshold).astype(int).eq(scores["delta"]).mean()
        )
        metrics["roc_auc"] = float(roc_auc_score(scores["delta"], scores["probability"]))

    fire = {c: float(scores[c].mean()) for c in scores.columns if c.startswith("policy_")}
    dead = [c for c, v in fire.items() if v in (0.0, 1.0)]

    metrics.update(
        {
            "formulation": args.formulation,
            "condition": args.condition,
            "mode": "control" if args.control else "rules",
            "seed": args.seed,
            "n_train_rows": len(X_train),
            "n_units_scored": len(scores),
            "n_policies_generated": int(len(pi._feature_order_)),
            "n_policies_nonzero": int(np.count_nonzero(pi.lr.coef_[0])),
            "n_policies_constant": len(dead),
            "policy_fire_rates": fire,
            "threshold": float(pi.threshold),
            "validation_result": pi.validation_result,
            "thl_commit": thl_commit(),
            "cache": cache.stats,
            "config": {
                "gen_model": args.gen_model, "predict_model": args.predict_model,
                "max_policies": args.max_policies,
                "samples_per_batch": args.samples_per_batch,
                "max_gen_batches": args.max_gen_batches,
                "policy_batch_size": args.policy_batch_size,
                "op_words": args.op_words, "beta": args.beta,
            },
        }
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{tag}.json"
    # Belt and braces. The tag covers the knobs known to matter, but a knob not
    # in the tag (op_words, max_policies, beta, ...) could still change the
    # result. Refuse to overwrite a result produced under a different config
    # rather than replacing it with no trace.
    if out.exists():
        prev = json.loads(out.read_text()).get("config")
        if prev and prev != metrics["config"]:
            differing = [k for k in metrics["config"]
                         if prev.get(k) != metrics["config"][k]]
            raise SystemExit(
                f"\n{out.name} already exists and was produced with a different "
                f"config.\nDiffering keys: {', '.join(differing)}\n"
                "Refusing to overwrite. Rename or delete the existing result, or "
                "change a\nsetting that appears in the run tag."
            )
    out.write_text(json.dumps(metrics, indent=2))
    scores.to_csv(RESULTS_DIR / f"{tag}_scores.csv", index=False)

    print(f"\n=== {tag} ===")
    print(f"  pairwise accuracy : {metrics['pairwise_accuracy']:.4f}  "
          f"({metrics['n_pairs_scored']} pairs)")
    print(f"  roc auc           : {metrics['roc_auc']:.4f}")
    if paired:
        n_t = int(metrics["n_ties"])
        print(f"  undecided (=0.50) : {n_t} pairs = "
              f"{n_t / max(metrics['n_pairs_scored'], 1):.0%}  (counted 0.5 each)")
        print(f"  acc when decided  : {metrics['accuracy_on_decided']:.4f}")
        print(f"  acc @ fitted thr  : {metrics['accuracy_at_fitted_threshold']:.4f}  "
              f"(diagnostic only, thr={metrics['threshold']:.2f})")
        print(f"  predicted A rate  : {metrics['pred_A_rate']:.1%}  (0.5 = unbiased)")
        if abs(float(metrics["pred_A_rate"]) - 0.5) > 0.15:
            print("  WARNING: predictions are lopsided towards one slot. Re-run "
                  "with --swap-eval\n           before trusting the accuracy.")
        if "positional_bias" in metrics:
            print(f"  positional bias   : {metrics['positional_bias']:+.4f}")
    else:
        print(f"  pointwise accuracy: {metrics['pointwise_accuracy']:.4f}")
    print(f"  policies          : {metrics['n_policies_nonzero']}/"
          f"{metrics['n_policies_generated']} non-zero, {len(dead)} constant")
    print("  fire rates        : " + " ".join(f"{v:.0%}" for v in fire.values()))
    if len(dead) > len(fire) / 3:
        print(f"  WARNING: {len(dead)}/{len(fire)} policies never vary -- likely "
              "not transferable.\n           Check report.md before trusting "
              "this number.")
    print(f"  cache             : {cache.stats}")
    cache.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--formulation", choices=("paired", "pointwise"), default="paired",
                   help="paired: one row per pair, argument_A vs argument_B.")
    p.add_argument("--condition", choices=CONDITIONS, default="root_reply")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-train-pairs", type=int, default=500,
                   help="Paired mode: training pairs. Stratified, exactly 50/50.")
    p.add_argument("--n-train-units", type=int, default=300,
                   help="Pointwise mode: training path-units.")
    p.add_argument("--n-eval-pairs", type=int, default=0, help="0 = all 807.")
    p.add_argument("--no-swap-eval", dest="swap_eval", action="store_false",
                   help="Paired: skip scoring every heldout pair in both orders. "
                        "Halves predict calls, but the accuracy is then only "
                        "trustworthy if pred_A_rate comes out near 0.5.")
    p.add_argument("--augment-swap", action="store_true",
                   help="Paired: also fit on every training pair with A and B "
                        "exchanged and the label flipped, so the learned "
                        "weights are symmetric. Doubles fit-scoring calls.")
    p.add_argument("--control", action="store_true",
                   help="One generic prompt, no induced rules.")
    p.add_argument("--gen-model", default="gemini-3.5-flash",
                   help="Strong model: writes the policies. For a Pro run use "
                        "gemini-3.1-pro-preview -- there is no 3.5 Pro, and "
                        "2.5 Pro is retired.")
    p.add_argument("--predict-model", nargs="+", default=["gemini-2.5-flash-lite"],
                   help="Cheap model(s): evaluates each rule against each "
                        "sample. Pass more than one to make it a PRIORITY "
                        "LIST -- the library tries the first model and falls "
                        "back to the next on a failed call (e.g. a 503), not "
                        "an ensemble of both. Order matters: put the model "
                        "you want used first.")
    p.add_argument("--max-policies", type=int, default=15)
    p.add_argument("--samples-per-batch", type=int, default=20)
    p.add_argument("--max-gen-batches", type=int, default=10)
    p.add_argument("--policy-batch-size", type=int, default=15)
    p.add_argument("--op-words", type=int, default=300,
                   help="Truncate op_text to N words (0 = full). Controls cost.")
    p.add_argument("--beta", type=float, default=0.5,
                   help="F-beta for threshold/C selection. 1.0 degenerates on a "
                        "50/50 split (always-YES scores F1=0.667).")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
