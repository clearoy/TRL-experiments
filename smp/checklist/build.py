"""Build the SMP content checklist: a small set of YES/NO questions about a video
that tell posts in the top 20% of their account-size group from the rest.

Stages (each cached in data/checklist/, so re-running resumes and skips work):

  folds     5 folds grouped by user (stratified by follower quintile), plus a
            time split (latest 20% of posts as test, their users kept out of train).
  generate  for each fold and the time split: the generator (GEN_MODELS, in
            priority order) writes candidate questions
            from a balanced sample (up to 300 YES + 300 NO) of that split's
            TRAINING posts only, using RRF's question generator.
  embed     Gemini embeddings of every question, to find near-duplicates.
  answer    Jev answers every unique question for every post (P(YES)).
  evaluate  per split: select a few questions on the training posts only
            (stability selection with L1 logistic regression, at most --max-rules),
            fit a logistic model on them, score the held-out posts. TF-IDF logistic
            regression runs on the same splits as the reference.
  final     the same selection on all posts, with questions pooled from every
            split (near-duplicates merged); per question: weight, lift (hit rate
            with vs without) and in how many folds a similar question was chosen.
            Writes smp/analyzer/model.json (what the analyzer loads) and
            data/checklist/report.json (what the dashboard shows).

Labels follow the split recipe: account groups and top-20% cutoffs are fitted on
each split's training posts and applied unchanged to its test posts.

Run:
    python smp/checklist/build.py            # all stages
    python smp/checklist/build.py --stage evaluate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

SMP = Path(__file__).resolve().parents[1]
REPO_ROOT = SMP.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_env() -> None:
    env = REPO_ROOT / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


_load_env()

from think_reason_learn.core.llms import (  # noqa: E402
    GoogleChoice, JevBudget, JevChoice, JevRequest, NoulQuestion, OpenAIChoice, llm,
)
from think_reason_learn.rrf import RRF  # noqa: E402

READY = SMP / "data" / "clean" / "ready_data.parquet"
USERS = SMP / "data" / "raw" / "train" / "SMP-Video_anonymized_users_train.jsonl"
WORK = SMP / "data" / "checklist"
MODEL_OUT = SMP / "analyzer" / "model.json"
DASHBOARD_OUT = SMP.parent / "dashboard" / "smp_checklist.json"

# Question writers, in priority order. gpt-6.1-sol was the first choice but the
# OpenAI account ran out of credits; Gemini is used instead. To switch back, put
# OpenAIChoice(model="gpt-6.1-sol") first and delete data/checklist/gen_*.json.
GEN_MODELS = [GoogleChoice(model="gemini-3.1-pro-preview"), GoogleChoice(model="gemini-3.5-flash")]
GEN_MODEL = " / ".join(c.model for c in GEN_MODELS)
EMBED_MODEL = "gemini-embedding-001"
GROUP_INPUTS = ["user_follower_count", "user_following_count", "user_video_count"]
N_GROUPS, TOP_FRAC = 5, 0.20
SEED = 0
DUP_SIM = 0.90      # embeddings this similar are the same question (a paraphrase scores ~0.90)
MATCH_SIM = 0.85    # a fold's question this similar counts as "the same rule" (unrelated ~0.70)
JEV_BATCH = 50      # Jev questions per request

TASK = (
    "Predict whether a short-form social media video post was one of the most "
    "popular posts compared with other posts from accounts of a similar size. "
    "Each sample is one post from a US, English-language account, shown as text: "
    "the creator's caption, when it was posted (UTC), the music or sound used, "
    "the video's length, orientation and resolution, and a detailed description "
    "of the video itself (what happens, the opening seconds, people, setting, "
    "filming, editing, audio, tone, a scene-by-scene timeline, the spoken words "
    "and the on-screen text). The size of the posting account is not shown and "
    "should not be guessed: the question is which content does better than other "
    "content from similar accounts. About 20% of posts are YES. Write concrete, "
    "checkable YES/NO questions about the content of a post (what is shown, said, "
    "heard or written, and how the video is built) whose answers help tell the "
    "top 20% of its account-size group (YES) from the rest (NO). Avoid vague "
    "quality judgements. Each question must be general enough to apply across "
    "topics: roughly 10% to 60% of all posts should get a YES. Never name a "
    "specific object, product, brand or person taken from one example; describe "
    "the general pattern instead (e.g. 'Does the video show a step-by-step "
    "process with a visible finished result?')."
)


# ------------------------------------------------------------------ data

def load() -> pd.DataFrame:
    d = pd.read_parquet(READY)
    u = pd.read_json(USERS, lines=True)[["uid", *GROUP_INPUTS]]
    d = d.merge(u, on="uid", how="left", validate="m:1").reset_index(drop=True)
    assert not d[GROUP_INPUTS].isna().any().any()
    return d


def label_split(d: pd.DataFrame, train: np.ndarray) -> dict:
    """Account groups and top-20% labels fitted on the training rows only."""
    X = np.log10(d[GROUP_INPUTS].clip(lower=1))
    m = LinearRegression().fit(X[train], d.popularity[train])
    score = m.predict(X)
    edges = np.quantile(score[train], np.linspace(0, 1, N_GROUPS + 1))
    group = np.searchsorted(edges[1:-1], score, side="right") + 1
    cut = {g: float(np.quantile(d.popularity[train & (group == g)], 1 - TOP_FRAC))
           for g in range(1, N_GROUPS + 1)}
    y = np.array([d.popularity[i] >= cut[group[i]] for i in range(len(d))], dtype=int)
    return {"y": y, "group": group, "expected": score,
            "model": {"intercept": float(m.intercept_), "coef": dict(zip(GROUP_INPUTS, map(float, m.coef_)))},
            "edges": [float(e) for e in edges], "cutoffs": cut}


def make_splits(d: pd.DataFrame) -> dict:
    strata = pd.qcut(np.log10(d.user_follower_count.clip(lower=1)), 5, labels=False)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = {}
    for k, (tr, te) in enumerate(sgkf.split(d, strata, groups=d.uid)):
        mask = np.zeros(len(d), bool); mask[tr] = True
        splits[f"fold{k}"] = mask
    t = pd.to_datetime(d.post_time if "post_time" in d else d.posted_at_utc.str.replace(" UTC", ""))
    late = t >= t.quantile(0.8)
    late_users = set(d.uid[late])
    train = (~late) & (~d.uid.isin(late_users))
    splits["time"] = train.to_numpy()
    # rows neither in train nor test for the time split: early posts of late users
    splits["time_test"] = late.to_numpy()
    return splits


# ------------------------------------------------------------- generation

class RetryLLM:
    """The library's LLM with patient retries on respond(): Gemini often answers
    "high demand" (503), and one failed call would otherwise lose a whole split."""

    def __init__(self, inner, tries: int = 12):
        self.inner, self.tries = inner, tries

    async def respond(self, *a, **k):
        import random
        for i in range(1, self.tries + 1):
            try:
                return await self.inner.respond(*a, **k)
            except Exception:
                if i == self.tries:
                    raise
                await asyncio.sleep(min(90, 2 ** i) + random.random() * 5)

    def __getattr__(self, name):
        return getattr(self.inner, name)


GEN_SLOTS = asyncio.Semaphore(4)  # at most four splits generate at once

async def generate(d: pd.DataFrame, split: str, train: np.ndarray, y: np.ndarray) -> list[str]:
    path = WORK / f"gen_{split}.json"
    if path.exists():
        return json.loads(path.read_text())
    rng = np.random.default_rng(SEED)
    yes = np.where(train & (y == 1))[0]; no = np.where(train & (y == 0))[0]
    pick = np.concatenate([rng.choice(yes, min(300, len(yes)), replace=False),
                           rng.choice(no, min(300, len(no)), replace=False)])
    X = d.loc[pick, ["model_text"]].rename(columns={"model_text": "post"}).reset_index(drop=True)
    labels = ["YES" if v else "NO" for v in y[pick]]
    rrf = RRF(qgen_llmc=GEN_MODELS, max_generated_questions=60,
              max_samples_as_context=30, class_ratio=(1.0, 1.0), random_state=SEED,
              save_path=WORK / f"rrf_{split}", name=f"gen_{split}")
    # Set after construction: RRF deep-copies its arguments, and the client holds locks.
    rrf._llm_instance = RetryLLM(llm)
    async with GEN_SLOTS:
        await rrf.set_task(task_description=TASK)
        rrf._set_data(X, labels, copy_data=True)
        await rrf._generate_questions()
    qs = [str(q).strip() for q in rrf._questions.question]
    path.write_text(json.dumps(qs, indent=1))
    print(f"  {split}: {len(qs)} questions")
    return qs


# ------------------------------------------------------------- embeddings

def embed(texts: list[str]) -> np.ndarray:
    path = WORK / f"embeddings_{EMBED_MODEL}.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    todo = [t for t in dict.fromkeys(texts) if t not in cache]
    if todo:
        from google import genai
        client = genai.Client(api_key=os.environ["GOOGLE_AI_API_KEY"])
        import time
        # Free tier: 100 texts per minute. Send 40 at a time, wait out 429s.
        for i in range(0, len(todo), 40):
            batch = todo[i:i + 40]
            for attempt in range(10):
                try:
                    r = client.models.embed_content(model=EMBED_MODEL, contents=batch)
                    break
                except Exception as e:
                    if "429" not in str(e) or attempt == 9:
                        raise
                    time.sleep(30)
            for t, e in zip(batch, r.embeddings):
                cache[t] = e.values
            path.write_text(json.dumps(cache))
            if i + 40 < len(todo):
                time.sleep(25)
    E = np.array([cache[t] for t in texts], dtype=float)
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def dedupe(texts: list[str]) -> list[str]:
    """Keep the first of every group of near-duplicate questions."""
    if not texts:
        return []
    E = embed(texts)
    kept: list[int] = []
    for i in range(len(texts)):
        if not kept or (E[kept] @ E[i]).max() < DUP_SIM:
            kept.append(i)
    return [texts[i] for i in kept]


# ----------------------------------------------------------------- answers

async def answer(d: pd.DataFrame, questions: list[str]) -> pd.DataFrame:
    """Jev's P(YES) for every (post, question). Cached by question text."""
    path = WORK / "answers.parquet"
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame(index=d.pid)
    todo = [q for q in dict.fromkeys(questions) if q not in have.columns]
    if todo:
        print(f"  Jev: {len(todo)} new questions x {len(d)} posts")
        budget = JevBudget(40.0)
        choice = [JevChoice(model="jev-latest", max_cost_usd=40.0)]
        new = {q: np.full(len(d), np.nan) for q in todo}
        for i in range(0, len(todo), JEV_BATCH):
            chunk = todo[i:i + JEV_BATCH]
            qmap = {f"q{j:03d}": q for j, q in enumerate(chunk)}
            reqs = [JevRequest(state=f"post: {t}",
                               questions={k: NoulQuestion(instructions=q) for k, q in qmap.items()})
                    for t in d.model_text]
            res = await llm.answer_many(choice, reqs, budget=budget, caller="checklist")
            for r_i, r in enumerate(res):
                for k, v in r.answers.items():
                    if isinstance(v, float):
                        new[qmap[k]][r_i] = v
            print(f"    answered {min(i + JEV_BATCH, len(todo))}/{len(todo)} questions")
        have = pd.concat([have, pd.DataFrame(new, index=d.pid)], axis=1)
        have.to_parquet(path)
        rel = getattr(budget, "_releases", [])
        (WORK / "jev_release.json").write_text(json.dumps({"releases": rel}))
    return have.loc[d.pid, list(dict.fromkeys(questions))]


# --------------------------------------------------------------- selection

def select(F: np.ndarray, y: np.ndarray, max_rules: int, n_boot: int = 100, pool: int = 20) -> dict:
    """Stability selection, then a short list.

    An L1 logistic regression with a penalty that keeps about `pool` questions is
    refitted on 100 random halves of the training posts; each question's
    selection frequency says how reliably it is picked. The `max_rules` most
    frequently picked questions (each picked in at least half the resamples) form
    the checklist, and a plain logistic regression is fitted on them.

    On the two folds, AUC was flat from 1 to 4 rules (0.615-0.618) and fell with
    more (0.604 at 6, 0.590 at 12), so a short list loses nothing."""
    mu, sd = np.nanmean(F, 0), np.nanstd(F, 0) + 1e-9
    Z = np.nan_to_num((F - mu) / sd)
    C = 0.5
    for c in np.logspace(-3, 1, 40):
        m = LogisticRegression(penalty="l1", C=c, solver="liblinear", random_state=SEED).fit(Z, y)
        if np.count_nonzero(m.coef_) > pool:
            break
        C = c
    rng = np.random.default_rng(SEED)
    freq = np.zeros(F.shape[1])
    for _ in range(n_boot):
        idx = rng.choice(len(y), len(y) // 2, replace=False)
        if y[idx].min() == y[idx].max():
            continue
        m = LogisticRegression(penalty="l1", C=C, solver="liblinear", random_state=SEED).fit(Z[idx], y[idx])
        freq += (m.coef_[0] != 0)
    freq /= n_boot
    order = np.argsort(-freq)
    chosen = [int(j) for j in order[:max_rules] if freq[j] >= 0.5] or [int(order[0])]
    fit = LogisticRegression(C=1.0, max_iter=5000).fit(Z[:, chosen], y)
    return {"chosen": chosen, "freq": freq, "mu": mu, "sd": sd, "model": fit, "C": float(C)}


def score(sel: dict, F: np.ndarray) -> np.ndarray:
    Z = np.nan_to_num((F - sel["mu"]) / sel["sd"])
    return sel["model"].predict_proba(Z[:, sel["chosen"]])[:, 1]


def tfidf_auc(d, tr, te, y) -> float:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")
    Xtr, Xte = vec.fit_transform(d.model_text[tr]), vec.transform(d.model_text[te])
    p = LogisticRegression(class_weight="balanced", max_iter=5000).fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
    return float(roc_auc_score(y[te], p))


# ------------------------------------------------------------------ stages

async def run(args) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    d = load()
    splits = make_splits(d)
    names = [f"fold{k}" for k in range(5)] + ["time"]
    labels = {s: label_split(d, splits[s]) for s in names}

    if args.stage in ("all", "generate") and not args.use_available:
        print("generate")
        res = await asyncio.gather(*(generate(d, s, splits[s], labels[s]["y"]) for s in names),
                                   return_exceptions=True)
        failed = [s for s, r in zip(names, res) if isinstance(r, Exception)]
        if failed:
            raise SystemExit(f"generation failed for {failed}; re-run to retry them (finished splits are kept)")
    # Splits whose questions exist. With --use-available the build goes ahead with
    # only those (e.g. when an API quota stops generation part-way); re-running
    # without it later generates the missing splits and rebuilds everything.
    names = [s for s in names if (WORK / f"gen_{s}.json").exists()]
    if not names:
        raise SystemExit("no generated questions yet")
    print(f"splits with questions: {names}")
    gens = {s: dedupe(json.loads((WORK / f"gen_{s}.json").read_text())) for s in names}
    folds = [s for s in names if s.startswith("fold")]
    pooled = dedupe([q for s in folds for q in gens[s]])

    if args.stage in ("all", "generate", "answer"):
        print("answer")
        await answer(d, [q for s in names for q in gens[s]] + pooled)
    A = await answer(d, [q for s in names for q in gens[s]] + pooled)

    if args.stage not in ("all", "generate", "answer", "evaluate", "final"):
        return

    # ---- evaluate: each split uses only questions generated from its own training posts
    print("evaluate")
    ev = {}
    oof = np.full(len(d), np.nan)
    for s in names:
        tr = splits[s]
        te = splits["time_test"] if s == "time" else ~tr
        y = labels[s]["y"]
        Q = gens[s]
        F = A[Q].to_numpy(float)
        sel = select(F[tr], y[tr], args.max_rules)
        p = score(sel, F[te])
        if s != "time":
            oof[te] = p
        all_q = LogisticRegression(penalty="l1", C=sel["C"] * 3, solver="liblinear", random_state=SEED)
        Zall = np.nan_to_num((F - sel["mu"]) / sel["sd"])
        p_all = all_q.fit(Zall[tr], y[tr]).predict_proba(Zall[te])[:, 1]
        ev[s] = {
            "n_train": int(tr.sum()), "n_test": int(te.sum()), "n_yes_test": int(y[te].sum()),
            "n_candidates": len(Q), "n_rules": len(sel["chosen"]),
            "rules": [Q[j] for j in sel["chosen"]],
            "rule_signs": [float(np.sign(c)) for c in sel["model"].coef_[0]],
            "auc_checklist": float(roc_auc_score(y[te], p)),
            "auc_all_questions_l1": float(roc_auc_score(y[te], p_all)),
            "auc_tfidf": tfidf_auc(d, tr, te, y),
        }
        print(f"  {s}: {ev[s]['n_rules']} rules | checklist {ev[s]['auc_checklist']:.3f} | "
              f"all questions {ev[s]['auc_all_questions_l1']:.3f} | tfidf {ev[s]['auc_tfidf']:.3f}")
    summary = {k: [float(np.mean([ev[s][k] for s in folds])), float(np.std([ev[s][k] for s in folds]))]
               for k in ("auc_checklist", "auc_all_questions_l1", "auc_tfidf", "n_rules")}

    # calibration of out-of-fold checklist scores (labels from each post's own fold)
    y_oof = np.zeros(len(d), int)
    for s in folds:
        y_oof[~splits[s]] = labels[s]["y"][~splits[s]]
    seen = ~np.isnan(oof)
    bins = pd.qcut(oof[seen], 5, labels=False, duplicates="drop")
    calib = [{"pred": float(oof[seen][bins == b].mean()), "obs": float(y_oof[seen][bins == b].mean()),
              "n": int((bins == b).sum())} for b in sorted(set(bins))]
    seen = ~np.isnan(oof)
    summary["oof_auc_checklist"] = float(roc_auc_score(y_oof[seen], oof[seen]))
    summary["n_folds"] = len(folds)

    # ---- final: all posts, pooled questions
    print("final")
    lab = label_split(d, np.ones(len(d), bool))
    y = lab["y"]
    F = A[pooled].to_numpy(float)
    sel = select(F, y, args.max_rules)
    E_pool = embed(pooled)
    rules = []
    for rank, j in enumerate(sel["chosen"]):
        q = pooled[j]
        w = float(sel["model"].coef_[0][rank])
        p = F[:, j]
        yes = p >= 0.5
        support = 0
        for s in folds:
            fq = ev[s]["rules"]
            if fq:
                sims = embed(fq) @ E_pool[j]
                k = int(np.argmax(sims))
                if sims[k] >= MATCH_SIM and np.sign(ev[s]["rule_signs"][k]) == np.sign(w):
                    support += 1
        rules.append({
            "id": f"r{rank + 1:02d}", "question": q, "weight": w,
            "direction": "helps" if w > 0 else "hurts",
            "share_yes": float(yes.mean()),
            "hit_rate_yes": float(y[yes].mean()) if yes.any() else None,
            "hit_rate_no": float(y[~yes].mean()) if (~yes).any() else None,
            "selection_freq": float(sel["freq"][j]),
            "fold_support": support,
            "mean": float(sel["mu"][j]), "sd": float(sel["sd"][j]),
        })
    rules.sort(key=lambda r: -abs(r["weight"]))
    rel = json.loads((WORK / "jev_release.json").read_text())["releases"] if (WORK / "jev_release.json").exists() else []
    model = {
        "created_with": {"generator": GEN_MODEL, "answerer": "jev", "jev_release": rel[0] if rel else "jev-latest",
                         "description_model": "gemini-3.5-flash", "description_prompt_version": "8d3a7a981f0c"},
        "label": "top 20% of popularity within the post's account-size group",
        "base_rate": float(y.mean()),
        "group_model": lab["model"], "group_edges": lab["edges"],
        "group_cutoffs": {str(k): v for k, v in lab["cutoffs"].items()},
        "intercept": float(sel["model"].intercept_[0]),
        "rules": rules,
        "evaluation": {"folds": summary, "time_split": ev.get("time"), "calibration": calib},
        "n_posts": int(len(d)), "n_candidates": len(pooled),
    }
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.write_text(json.dumps(model, indent=2))
    # The dashboard's copy: the rules and evaluation, without the scaling numbers.
    public = {k: v for k, v in model.items() if k not in ("group_model", "group_edges", "group_cutoffs", "intercept")}
    public["rules"] = [{k: v for k, v in r.items() if k not in ("mean", "sd")} for r in rules]
    DASHBOARD_OUT.write_text(json.dumps(public, indent=2))
    (WORK / "report.json").write_text(json.dumps({"splits": ev, "summary": summary, "calibration": calib,
                                                   "final_rules": rules, "n_candidates_pooled": len(pooled)}, indent=2))
    print(f"\nfolds: checklist AUC {summary['auc_checklist'][0]:.3f} ± {summary['auc_checklist'][1]:.3f} "
          f"({summary['n_rules'][0]:.1f} rules) | all questions {summary['auc_all_questions_l1'][0]:.3f} | "
          f"tfidf {summary['auc_tfidf'][0]:.3f}")
    if "time" in ev:
        print(f"time split: checklist {ev['time']['auc_checklist']:.3f} | tfidf {ev['time']['auc_tfidf']:.3f}")
    print(f"final: {len(rules)} rules from {len(pooled)} pooled questions -> {MODEL_OUT}")
    for r in rules:
        print(f"  {r['weight']:+.2f} [{r['fold_support']}/{len(folds)} folds] {r['question'][:110]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="all", choices=["all", "generate", "answer", "evaluate", "final"])
    p.add_argument("--max-rules", type=int, default=5)
    p.add_argument("--use-available", action="store_true",
                   help="Skip generation and build from the splits whose questions already exist.")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
