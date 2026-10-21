"""Compare video-to-text extraction designs on a sample of SMP videos.

Stages (each cached in data/research/video_extraction/, so re-running resumes):
  sample     pick N videos, stratified by user group (from data/split/)
  extract    run every variant in prompts.VARIANTS on every sampled video
  questions  the judge watches each video and writes 12 factual Q&A
  answer     a text-only model answers the questions from each variant's text
  grade      a text-only grader marks each answer against the reference
  verify     the judge watches each video again and lists unsupported claims
  report     aggregate everything into report.json / printed tables

Run:
    python smp/research/video_extraction/run.py all
    python smp/research/video_extraction/run.py report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types

import prompts as P

HERE = Path(__file__).resolve().parent
SMP = HERE.parents[1]
REPO_ROOT = SMP.parents[1]
VIDEOS = SMP / "data" / "clean" / "videos"
SPLIT = SMP / "data" / "split"
OUT = SMP / "data" / "research" / "video_extraction"
# The judge falls back to the next model after two overloaded attempts.
JUDGE = ["gemini-3.1-pro-preview", "gemini-2.5-pro"]
TEXT_MODEL = "gemini-3.5-flash"
INLINE_LIMIT = 18 * 1024 * 1024
CALL_TIMEOUT = 300  # seconds; a stuck request is retried instead of hanging
RETRIES = 6
MAX_WAIT = 90  # longest pause between retries, in seconds


# ---------------------------------------------------------------- utilities

def load_env() -> None:
    env = REPO_ROOT / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


class Store:
    """Append-only JSONL keyed by a tuple of fields."""

    def __init__(self, name: str, key: tuple[str, ...]):
        self.path = OUT / f"{name}.jsonl"
        self.key = key
        self.lock = asyncio.Lock()
        self.rows = {tuple(r[k] for k in key): r for r in read_jsonl(self.path)}

    def has(self, *k) -> bool:
        return tuple(k) in self.rows

    async def add(self, row: dict) -> None:
        async with self.lock:
            self.rows[tuple(row[k] for k in self.key)] = row
            with self.path.open("a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


class Gemini:
    def __init__(self, concurrency: int):
        load_env()
        self.client = genai.Client(api_key=os.environ["GOOGLE_AI_API_KEY"])
        self.sem = asyncio.Semaphore(concurrency)

    async def _video_part(self, path: Path, fps: float | None):
        meta = types.VideoMetadata(fps=fps) if fps else None
        if path.stat().st_size <= INLINE_LIMIT:
            return types.Part(
                inline_data=types.Blob(data=path.read_bytes(), mime_type="video/mp4"),
                video_metadata=meta,
            ), None
        f = await self.client.aio.files.upload(
            file=str(path), config=types.UploadFileConfig(mime_type="video/mp4")
        )
        while f.state and f.state.name == "PROCESSING":
            await asyncio.sleep(3)
            f = await self.client.aio.files.get(name=f.name)
        part = types.Part(
            file_data=types.FileData(file_uri=f.uri, mime_type="video/mp4"),
            video_metadata=meta,
        )
        return part, f.name

    async def call(self, model, prompt, schema, video: Path | None = None,
                   fps: float | None = None, retries: int | None = None):
        retries = retries or RETRIES
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.2,
            max_output_tokens=16384,
        )
        async with self.sem:
            for attempt in range(1, retries + 1):
                uploaded = None
                try:
                    parts = []
                    if video is not None:
                        vp, uploaded = await self._video_part(video, fps)
                        parts.append(vp)
                    parts.append(types.Part.from_text(text=prompt))
                    models = model if isinstance(model, list) else [model]
                    use = models[min((attempt - 1) // 2, len(models) - 1)]
                    t0 = time.time()
                    resp = await asyncio.wait_for(
                        self.client.aio.models.generate_content(
                            model=use,
                            contents=[types.Content(role="user", parts=parts)],
                            config=config,
                        ),
                        timeout=CALL_TIMEOUT,
                    )
                    obj = schema.model_validate_json(resp.text)
                    u = resp.usage_metadata
                    return obj, {
                        "model": use,
                        "in_tokens": getattr(u, "prompt_token_count", None),
                        "out_tokens": getattr(u, "candidates_token_count", None),
                        "seconds": round(time.time() - t0, 1),
                    }
                except Exception as e:
                    msg = f"{type(e).__name__}: {str(e)[:100]}"
                    if attempt == retries:
                        print(f"    gave up: {msg}")
                        return None, {"error": msg}
                    await asyncio.sleep(min(MAX_WAIT, 2 ** attempt) + random.random() * 5)
                finally:
                    if uploaded:
                        try:
                            await self.client.aio.files.delete(name=uploaded)
                        except Exception:
                            pass


def render(obj: dict) -> str:
    """The text a downstream reader would get for one extraction."""
    if set(obj) == {"description"}:
        return obj["description"]
    return json.dumps(obj, indent=1, ensure_ascii=False)


# ------------------------------------------------------------------- stages

def stage_sample(n_per_group: int, seed: int) -> pd.DataFrame:
    path = OUT / "sample.csv"
    if path.exists():
        return pd.read_csv(path)
    frames = []
    for g in range(1, 6):
        for part in ("train", "test"):
            d = pd.read_parquet(SPLIT / f"group_{g}" / f"{part}.parquet")
            frames.append(d[["pid", "vid", "user_group", "video_duration", "video_path"]])
    allp = pd.concat(frames)
    sample = allp.groupby("user_group", group_keys=False).sample(n=n_per_group, random_state=seed)
    OUT.mkdir(parents=True, exist_ok=True)
    sample.to_csv(path, index=False)
    return sample


async def stage_extract(g: Gemini, sample: pd.DataFrame, variants: list[str]):
    store = Store("extract", ("variant", "pid"))
    jobs = []
    for v in variants:
        prompt, schema, model, fps = P.VARIANTS[v]
        for r in sample.itertuples():
            if store.has(v, r.pid):
                continue

            async def job(v=v, r=r, prompt=prompt, schema=schema, model=model, fps=fps):
                obj, meta = await g.call(model, prompt, schema, VIDEOS / r.video_path, fps)
                if obj is not None:
                    await store.add({"variant": v, "pid": r.pid, "output": obj.model_dump(), **meta})
            jobs.append(job())
    print(f"extract: {len(jobs)} calls")
    await asyncio.gather(*jobs)


async def stage_questions(g: Gemini, sample: pd.DataFrame):
    store = Store("questions", ("pid",))
    jobs = []
    for r in sample.itertuples():
        if store.has(r.pid):
            continue

        async def job(r=r):
            obj, meta = await g.call(JUDGE, P.QUESTIONS_PROMPT, P.QuestionSet, VIDEOS / r.video_path, fps=2.0)
            if obj is not None:
                await store.add({"pid": r.pid, "questions": [q.model_dump() for q in obj.questions], **meta})
        jobs.append(job())
    print(f"questions: {len(jobs)} calls")
    await asyncio.gather(*jobs)


async def stage_answer(g: Gemini):
    ext = Store("extract", ("variant", "pid")).rows
    qs = Store("questions", ("pid",)).rows
    store = Store("answers", ("variant", "pid"))
    jobs = []
    for (v, pid), e in ext.items():
        if (pid,) not in qs or store.has(v, pid):
            continue
        qlist = qs[(pid,)]["questions"]
        qtext = "\n".join(f"{i}. {q['question']}" for i, q in enumerate(qlist))
        prompt = P.ANSWER_PROMPT.format(record=render(e["output"]), questions=qtext)

        async def job(v=v, pid=pid, prompt=prompt):
            obj, meta = await g.call(TEXT_MODEL, prompt, P.AnswerSet)
            if obj is not None:
                await store.add({"variant": v, "pid": pid, "answers": [a.model_dump() for a in obj.answers], **meta})
        jobs.append(job())
    print(f"answer: {len(jobs)} calls")
    await asyncio.gather(*jobs)


async def stage_grade(g: Gemini):
    ans = Store("answers", ("variant", "pid")).rows
    qs = Store("questions", ("pid",)).rows
    store = Store("grades", ("variant", "pid"))
    jobs = []
    for (v, pid), a in ans.items():
        if store.has(v, pid):
            continue
        qlist = qs[(pid,)]["questions"]
        pred = {x["id"]: x["answer"] for x in a["answers"]}
        items = "\n\n".join(
            f"id {i}\nquestion: {q['question']}\nreference: {q['answer']}\npredicted: {pred.get(i, 'NOT IN TEXT')}"
            for i, q in enumerate(qlist)
        )

        async def job(v=v, pid=pid, items=items):
            obj, meta = await g.call(TEXT_MODEL, P.GRADE_PROMPT.format(items=items), P.GradeSet)
            if obj is not None:
                await store.add({"variant": v, "pid": pid, "grades": [x.model_dump() for x in obj.grades], **meta})
        jobs.append(job())
    print(f"grade: {len(jobs)} calls")
    await asyncio.gather(*jobs)


async def stage_verify(g: Gemini, sample: pd.DataFrame):
    ext = Store("extract", ("variant", "pid")).rows
    paths = dict(zip(sample.pid, sample.video_path))
    store = Store("verify", ("variant", "pid"))
    jobs = []
    for (v, pid), e in ext.items():
        if store.has(v, pid):
            continue
        prompt = P.VERIFY_PROMPT.format(record=render(e["output"]))

        async def job(v=v, pid=pid, prompt=prompt):
            obj, meta = await g.call(JUDGE, prompt, P.Verification, VIDEOS / paths[pid], fps=2.0)
            if obj is not None:
                await store.add({"variant": v, "pid": pid, **obj.model_dump(), **meta})
        jobs.append(job())
    print(f"verify: {len(jobs)} calls")
    await asyncio.gather(*jobs)


VIOLATIONS = {
    "popularity": r"\b(trending|viral|popular|famous|iconic|well[- ]known|meme)\b",
    "demographic": r"\b(black|white|asian|latina|latino|hispanic|african|caucasian|arab|indian|muslim|christian|jewish|gay|lesbian)\s+(man|woman|men|women|person|people|girl|boy|guy|couple|family)\b",
    "handle": r"@\w{3,}",
}


def stage_report(sample: pd.DataFrame) -> None:
    ext = Store("extract", ("variant", "pid")).rows
    qs = Store("questions", ("pid",)).rows
    grades = Store("grades", ("variant", "pid")).rows
    ver = Store("verify", ("variant", "pid")).rows
    score = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0, "not_in_text": 0.0}

    rows = []
    for (v, pid), e in ext.items():
        text = render(e["output"])
        r = {"variant": v, "pid": pid, "words": len(text.split()),
             "in_tokens": e.get("in_tokens"), "out_tokens": e.get("out_tokens")}
        for name, pat in VIOLATIONS.items():
            r[f"viol_{name}"] = len(re.findall(pat, text, flags=re.I))
        gr = grades.get((v, pid))
        if gr and (pid,) in qs:
            cats = [q["category"] for q in qs[(pid,)]["questions"]]
            for x in gr["grades"]:
                if 0 <= x["id"] < len(cats):
                    rows.append({**r, "category": cats[x["id"]], "grade": x["grade"],
                                 "qa": score.get(x["grade"], 0.0)})
        vr = ver.get((v, pid))
        r["major"] = sum(i["severity"] == "major" for i in vr["issues"]) if vr else None
        r["minor"] = sum(i["severity"] == "minor" for i in vr["issues"]) if vr else None
        r["accuracy"] = vr["accuracy"] if vr else None
        rows.append({**r, "category": None, "grade": None, "qa": None})

    if not rows:
        return print("nothing to report yet")
    df = pd.DataFrame(rows)
    per = df[df.category.isna()].drop(columns=["category", "grade", "qa"])
    qa = df[df.category.notna()]

    order = list(P.VARIANTS)
    summary = per.groupby("variant").agg(
        videos=("pid", "nunique"), words=("words", "median"), out_tokens=("out_tokens", "mean"),
        in_tokens=("in_tokens", "mean"), major=("major", "mean"), minor=("minor", "mean"),
        accuracy=("accuracy", "mean"), viol_popularity=("viol_popularity", "sum"),
        viol_demographic=("viol_demographic", "sum"), viol_handle=("viol_handle", "sum"),
    )
    q = qa.groupby("variant").agg(
        qa_score=("qa", "mean"),
        correct=("grade", lambda s: (s == "correct").mean()),
        not_in_text=("grade", lambda s: (s == "not_in_text").mean()),
        incorrect=("grade", lambda s: (s == "incorrect").mean()),
    )
    summary = summary.join(q).reindex([v for v in order if v in summary.index])
    by_cat = qa.pivot_table(index="variant", columns="category", values="qa", aggfunc="mean").reindex(summary.index)

    pd.set_option("display.width", 200)
    print(summary.round(3).to_string())
    print("\nQA score by question category")
    print(by_cat.round(3).to_string())
    (OUT / "report.json").write_text(json.dumps({
        "summary": json.loads(summary.round(4).to_json(orient="index")),
        "by_category": json.loads(by_cat.round(4).to_json(orient="index")),
        "n_videos": int(sample.pid.nunique()),
    }, indent=2))


async def main_async(args):
    global RETRIES, MAX_WAIT
    RETRIES, MAX_WAIT = args.retries, args.max_wait
    sample = stage_sample(args.per_group, args.seed)
    if args.stage == "sample":
        return print(f"sample: {len(sample)} videos")
    if args.stage == "report":
        return stage_report(sample)
    g = Gemini(args.concurrency)
    variants = args.variants.split(",") if args.variants else list(P.VARIANTS)
    stages = ["extract", "questions", "answer", "grade", "verify"] if args.stage == "all" else [args.stage]
    for s in stages:
        t0 = time.time()
        if s == "extract":
            await stage_extract(g, sample, variants)
        elif s == "questions":
            await stage_questions(g, sample)
        elif s == "answer":
            await stage_answer(g)
        elif s == "grade":
            await stage_grade(g)
        elif s == "verify":
            await stage_verify(g, sample)
        print(f"  {s} finished in {time.time() - t0:.0f}s")
    stage_report(sample)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["all", "sample", "extract", "questions", "answer", "grade", "verify", "report"])
    p.add_argument("--per-group", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variants", default="", help="Comma-separated subset of variants.")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--retries", type=int, default=6)
    p.add_argument("--max-wait", type=int, default=90, help="Longest pause between retries (s).")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
