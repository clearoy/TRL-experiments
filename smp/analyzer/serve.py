"""Local server for the SMP video analyzer (internal use).

Serves the dashboard folder and a small API the analyzer page calls:

    GET  /api/health              {"ok": true}
    POST /api/analyze?caption=&music=&posted_at=
                                  body: the raw .mp4 bytes -> {"job": "<id>"}
    GET  /api/jobs/<id>           {"status": queued|running|done|error, "step", "result"|"error"}

Standard library only. Analyses run one at a time on a single event loop in a
background thread (the library's LLM clients are bound to one loop).

Run:
    python smp/analyzer/serve.py            # then open http://localhost:8777/analyzer.html
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import threading
import traceback
import uuid
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
DASHBOARD = HERE.parents[1] / "dashboard"
sys.path.insert(0, str(HERE))

import analyze as az  # noqa: E402

MAX_BYTES = 300 * 1024 * 1024
JOBS: dict[str, dict] = {}
LOOP = asyncio.new_event_loop()
_one_at_a_time: asyncio.Lock | None = None


def _loop_thread() -> None:
    global _one_at_a_time
    asyncio.set_event_loop(LOOP)
    _one_at_a_time = asyncio.Lock()
    LOOP.run_forever()


async def _run_job(job_id: str, video: Path, params: dict) -> None:
    job = JOBS[job_id]
    assert _one_at_a_time is not None
    async with _one_at_a_time:
        job["status"] = "running"
        try:
            res = await az.analyze(video, progress=lambda s: job.__setitem__("step", s), **params)
            job.update(status="done", step="Done", result=res)
        except Exception as e:
            traceback.print_exc()
            job.update(status="error", error=f"{type(e).__name__}: {e}")
        finally:
            video.unlink(missing_ok=True)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(DASHBOARD), **k)

    def end_headers(self):
        # Internal tool that changes often: never let the browser serve a stale page.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):  # keep the console quiet except API calls
        if "/api/" in (args[0] if args else ""):
            super().log_message(fmt, *args)

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            return self._json(200, {"ok": True})
        if path.startswith("/api/jobs/"):
            job = JOBS.get(path.rsplit("/", 1)[-1])
            return self._json(200, job) if job else self._json(404, {"error": "no such job"})
        if path == "/":
            self.path = "/analyzer.html"
        return super().do_GET()

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/api/analyze":
            return self._json(404, {"error": "not found"})
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            params = {"caption": q.get("caption", ""), "music": q.get("music", "")}
            if q.get("posted_at"):
                params["posted_at"] = datetime.fromisoformat(q["posted_at"]).replace(tzinfo=timezone.utc)
        except ValueError:
            return self._json(400, {"error": "posted_at must be a date and time like 2026-11-04T18:30"})
        n = int(self.headers.get("Content-Length", 0))
        if not n:
            return self._json(400, {"error": "no video in the request"})
        if n > MAX_BYTES:
            return self._json(413, {"error": f"video is larger than {MAX_BYTES // 2**20} MB"})
        tmp = Path(tempfile.mkstemp(suffix=".mp4")[1])
        with tmp.open("wb") as fh:
            left = n
            while left:
                chunk = self.rfile.read(min(left, 1 << 20))
                if not chunk:
                    break
                fh.write(chunk); left -= len(chunk)
        job_id = uuid.uuid4().hex[:12]
        JOBS[job_id] = {"status": "queued", "step": "Waiting to start"}
        asyncio.run_coroutine_threadsafe(_run_job(job_id, tmp, params), LOOP)
        return self._json(202, {"job": job_id})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8777)))
    a = p.parse_args()
    threading.Thread(target=_loop_thread, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"SMP analyzer on http://localhost:{a.port}/analyzer.html  (dashboard: {DASHBOARD})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
