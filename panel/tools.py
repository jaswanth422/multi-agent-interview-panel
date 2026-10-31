"""Real tools the panel can call. No mocks on the happy path.

  code_exec       - executes candidate-written Python in a sandboxed subprocess
                    against hidden unit tests (real code execution)
  resume_db       - SQL query against a real SQLite resume/claims database
  github_verify   - live GitHub REST API call to verify a claimed account/repo
  web_fact_check  - live HTTP call (Wikipedia REST + DuckDuckGo) to check a
                    technical assertion
  speech_say      - real text-to-speech via the macOS `say` binary
  export_report   - writes the signed interview artifact to disk

Every invocation is timed, logged to SQLite, and degrades to a structured
failure object instead of raising, so one dead network call can never kill a
live demo. `ToolBelt.failure_handled` records that we exercised that path.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx

from .config import ARTIFACTS, SETTINGS
from .memory import Memory

SAFE_IMPORTS = {"math", "re", "itertools", "collections", "functools", "heapq",
                "bisect", "string", "typing", "dataclasses", "json", "random"}

BANNED_CODE = re.compile(
    r"\b(import\s+(os|sys|subprocess|socket|shutil|pathlib|requests|httpx|ctypes|"
    r"multiprocessing|threading)|__import__|open\s*\(|eval\s*\(|exec\s*\(|"
    r"compile\s*\(|globals\s*\(|locals\s*\(|getattr\s*\(|setattr\s*\()", re.I)


@dataclass
class ToolResult:
    tool: str
    ok: bool
    data: Any = None
    error: str = ""
    latency_ms: float = 0.0
    degraded: bool = False

    def brief(self, limit: int = 420) -> str:
        if not self.ok:
            return f"[{self.tool} FAILED: {self.error}] (panel continued without it)"
        return f"[{self.tool}] " + json.dumps(self.data, default=str)[:limit]


class ToolBelt:
    def __init__(self, memory: Memory, session_id: str):
        self.m = memory
        self.sid = session_id
        self.round = 0
        self.agent = "panel"
        self.failure_handled: list[dict[str, Any]] = []
        self._http = httpx.Client(
            timeout=SETTINGS.http_timeout_s,
            headers={"User-Agent": "multi-agent-interview-panel/1.0"},
            follow_redirects=True,
        )
        self._resume_db: str | None = None

    # ------------------------------------------------------------ registry
    def registry(self) -> dict[str, Callable[..., ToolResult]]:
        r = {
            "code_exec": self.code_exec,
            "resume_db": self.resume_db,
            "export_report": self.export_report,
        }
        if SETTINGS.enable_github:
            r["github_verify"] = self.github_verify
        if SETTINGS.enable_web:
            r["web_fact_check"] = self.web_fact_check
        if SETTINGS.enable_speech:
            r["speech_say"] = self.speech_say
        return r

    def catalog(self) -> list[dict[str, str]]:
        return [
            {"name": "code_exec", "desc": "Run candidate Python against hidden tests. "
                                          "args: code(str), task_id(str)"},
            {"name": "resume_db", "desc": "Query the candidate's structured resume. "
                                          "args: field(str) one of projects|skills|roles|all"},
            {"name": "github_verify", "desc": "Verify a claimed GitHub user/repo exists. "
                                              "args: user(str), repo(str optional)"},
            {"name": "web_fact_check", "desc": "Look up a technical term to check an "
                                               "assertion. args: query(str)"},
        ]

    def _run(self, name: str, args: dict[str, Any], fn: Callable[[], Any]) -> ToolResult:
        t0 = time.time()
        ok, data, err, degraded = True, None, "", False
        try:
            data = fn()
            if isinstance(data, dict) and data.get("_degraded"):
                degraded = True
        except Exception as e:  # noqa: BLE001 - tools must never crash the panel
            ok, err = False, f"{type(e).__name__}: {e}"[:400]
            self.failure_handled.append({"tool": name, "error": err, "round": self.round})
        dt = (time.time() - t0) * 1000
        self.m.log_tool(self.sid, self.round, self.agent, name, args,
                        data, ok, err, dt)
        return ToolResult(tool=name, ok=ok, data=data, error=err,
                          latency_ms=round(dt, 1), degraded=degraded)

    # ------------------------------------------------------------ 1. code
    def code_exec(self, code: str, task_id: str = "two_sum") -> ToolResult:
        """Execute untrusted candidate code in a subprocess with a hard timeout."""
        def inner():
            task = CODE_TASKS.get(task_id) or CODE_TASKS["two_sum"]
            if not code or not code.strip():
                return {"passed": 0, "total": len(task["tests"]), "verdict": "no_code_submitted",
                        "cases": [], "static_check": "empty"}
            banned = BANNED_CODE.search(code)
            if banned:
                # Guardrail: refuse to execute code that reaches outside the sandbox.
                return {"passed": 0, "total": len(task["tests"]), "verdict": "refused_unsafe_code",
                        "static_check": f"blocked pattern: {banned.group(0)!r}", "cases": []}

            harness = _HARNESS.format(
                entry=task["entry"],
                tests=json.dumps(task["tests"]),
            )
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / "sol.py"
                p.write_text(code + "\n\n" + harness, encoding="utf-8")
                proc = subprocess.run(
                    [sys.executable, str(p)],
                    capture_output=True, text=True,
                    timeout=SETTINGS.code_exec_timeout_s,
                    cwd=td,
                    env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1",
                         "HOME": td},
                )
            tail = (proc.stdout or "").strip().splitlines()
            if not tail:
                return {"passed": 0, "total": len(task["tests"]), "verdict": "crashed",
                        "stderr": (proc.stderr or "")[-500:], "cases": []}
            try:
                res = json.loads(tail[-1])
            except json.JSONDecodeError:
                return {"passed": 0, "total": len(task["tests"]), "verdict": "bad_output",
                        "stdout": "\n".join(tail)[-500:], "cases": []}
            res["verdict"] = ("pass" if res.get("passed") == res.get("total")
                              else "partial" if res.get("passed") else "fail")
            res["task"] = task_id
            res["prompt"] = task["prompt"]
            return res
        return self._run("code_exec", {"task_id": task_id, "code_len": len(code or "")}, inner)

    def code_task(self, task_id: str = "two_sum") -> dict[str, Any]:
        t = CODE_TASKS.get(task_id) or CODE_TASKS["two_sum"]
        return {"task_id": task_id, "prompt": t["prompt"], "entry": t["entry"]}

    # ------------------------------------------------------------ 2. DB
    def seed_resume_db(self, resume: dict[str, Any]) -> str:
        """Materialize the resume into a real relational DB the agents must query."""
        path = str(ARTIFACTS / f"resume_{self.sid}.sqlite3")
        con = sqlite3.connect(path)
        con.executescript("""
          CREATE TABLE IF NOT EXISTS profile(k TEXT PRIMARY KEY, v TEXT);
          CREATE TABLE IF NOT EXISTS projects(name TEXT, role TEXT, team_size INTEGER,
                                              stack TEXT, impact TEXT, months INTEGER);
          CREATE TABLE IF NOT EXISTS skills(name TEXT, years REAL, level TEXT);
          CREATE TABLE IF NOT EXISTS roles(company TEXT, title TEXT, years REAL);
        """)
        con.execute("DELETE FROM projects"); con.execute("DELETE FROM skills")
        con.execute("DELETE FROM roles"); con.execute("DELETE FROM profile")
        for k in ("name", "headline", "total_years", "location"):
            if k in resume:
                con.execute("INSERT OR REPLACE INTO profile VALUES(?,?)", (k, str(resume[k])))
        for p in resume.get("projects", []):
            con.execute("INSERT INTO projects VALUES(?,?,?,?,?,?)",
                        (p.get("name", ""), p.get("role", ""), p.get("team_size") or 0,
                         ", ".join(p.get("stack", [])), p.get("impact", ""),
                         p.get("months") or 0))
        for s in resume.get("skills", []):
            con.execute("INSERT INTO skills VALUES(?,?,?)",
                        (s.get("name", ""), s.get("years") or 0, s.get("level", "")))
        for r in resume.get("roles", []):
            con.execute("INSERT INTO roles VALUES(?,?,?)",
                        (r.get("company", ""), r.get("title", ""), r.get("years") or 0))
        con.commit(); con.close()
        self._resume_db = path
        return path

    def resume_db(self, field: str = "all") -> ToolResult:
        def inner():
            if not self._resume_db or not os.path.exists(self._resume_db):
                raise FileNotFoundError("resume database not provisioned for this session")
            con = sqlite3.connect(self._resume_db)
            con.row_factory = sqlite3.Row
            q = {
                "projects": "SELECT name,role,team_size,stack,impact,months FROM projects",
                "skills": "SELECT name,years,level FROM skills",
                "roles": "SELECT company,title,years FROM roles",
                "profile": "SELECT k,v FROM profile",
            }
            if field == "all":
                out = {k: [dict(r) for r in con.execute(v)] for k, v in q.items()}
            elif field in q:
                out = {field: [dict(r) for r in con.execute(q[field])]}
            else:
                con.close()
                raise ValueError(f"unknown field {field!r}; use projects|skills|roles|profile|all")
            con.close()
            out["_sql_executed"] = q.get(field, "all four tables")
            return out
        return self._run("resume_db", {"field": field}, inner)

    # ------------------------------------------------------------ 3. GitHub
    def github_verify(self, user: str, repo: str = "") -> ToolResult:
        def inner():
            user_clean = re.sub(r"[^A-Za-z0-9_.-]", "", (user or "").strip())
            if not user_clean:
                raise ValueError("no github user supplied")
            headers = {"Accept": "application/vnd.github+json"}
            if os.environ.get("GITHUB_TOKEN"):
                headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
            r = self._http.get(f"https://api.github.com/users/{user_clean}", headers=headers)
            if r.status_code == 404:
                return {"user": user_clean, "exists": False,
                        "note": "claimed GitHub account does not exist"}
            if r.status_code == 403:
                return {"_degraded": True, "user": user_clean,
                        "note": "GitHub rate limit hit; verification skipped"}
            r.raise_for_status()
            u = r.json()
            out = {"user": user_clean, "exists": True,
                   "public_repos": u.get("public_repos"),
                   "followers": u.get("followers"),
                   "created_at": u.get("created_at"),
                   "name": u.get("name")}
            if repo:
                repo_clean = re.sub(r"[^A-Za-z0-9_.-]", "", repo.strip())
                rr = self._http.get(
                    f"https://api.github.com/repos/{user_clean}/{repo_clean}", headers=headers)
                if rr.status_code == 200:
                    j = rr.json()
                    out["repo"] = {"name": repo_clean, "exists": True,
                                   "stars": j.get("stargazers_count"),
                                   "language": j.get("language"),
                                   "pushed_at": j.get("pushed_at")}
                else:
                    out["repo"] = {"name": repo_clean, "exists": False,
                                   "status": rr.status_code}
            return out
        return self._run("github_verify", {"user": user, "repo": repo}, inner)

    # ------------------------------------------------------------ 4. web
    def web_fact_check(self, query: str) -> ToolResult:
        def inner():
            q = (query or "").strip()
            if not q:
                raise ValueError("empty query")
            # Wikipedia REST summary - stable, no key, good for technical terms.
            title = re.sub(r"\s+", "_", q)[:120]
            r = self._http.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}")
            if r.status_code == 200:
                j = r.json()
                if j.get("extract"):
                    return {"source": "wikipedia", "query": q,
                            "title": j.get("title"),
                            "extract": j["extract"][:700],
                            "url": (j.get("content_urls", {})
                                      .get("desktop", {}).get("page", ""))}
            # Fallback: Wikipedia search API.
            r2 = self._http.get("https://en.wikipedia.org/w/api.php", params={
                "action": "query", "list": "search", "srsearch": q,
                "format": "json", "srlimit": 3})
            r2.raise_for_status()
            hits = r2.json().get("query", {}).get("search", [])
            if hits:
                return {"source": "wikipedia_search", "query": q,
                        "results": [{"title": h["title"],
                                     "snippet": re.sub(r"<[^>]+>", "", h.get("snippet", ""))}
                                    for h in hits]}
            # Last resort: DuckDuckGo instant answer.
            r3 = self._http.get("https://api.duckduckgo.com/",
                                params={"q": q, "format": "json", "no_html": 1})
            r3.raise_for_status()
            j3 = r3.json()
            return {"source": "duckduckgo", "query": q,
                    "abstract": (j3.get("AbstractText") or "")[:600],
                    "heading": j3.get("Heading", ""),
                    "_degraded": not (j3.get("AbstractText"))}
        return self._run("web_fact_check", {"query": query}, inner)

    # ------------------------------------------------------------ 5. speech
    def speech_say(self, text: str, voice: str = "Samantha") -> ToolResult:
        def inner():
            binary = shutil.which("say")
            if not binary:
                return {"_degraded": True, "note": "no TTS binary on this host"}
            wav = str(ARTIFACTS / f"q_{self.sid}_{int(time.time())}.aiff")
            subprocess.run([binary, "-v", voice, "-o", wav, text[:600]],
                           capture_output=True, timeout=25, check=False)
            return {"spoken_chars": len(text[:600]), "audio_file": wav,
                    "exists": os.path.exists(wav)}
        return self._run("speech_say", {"chars": len(text or ""), "voice": voice}, inner)

    # ------------------------------------------------------------ 6. export
    def export_report(self, payload: dict[str, Any], name: str = "") -> ToolResult:
        def inner():
            fn = name or f"interview_{self.sid}.json"
            p = ARTIFACTS / fn
            p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            return {"path": str(p), "bytes": p.stat().st_size}
        return self._run("export_report", {"name": name}, inner)

    def stats(self) -> dict[str, Any]:
        calls = self.m.tool_calls(self.sid)
        by: dict[str, dict[str, Any]] = {}
        for c in calls:
            b = by.setdefault(c["tool"], {"calls": 0, "ok": 0, "mean_ms": 0.0})
            b["calls"] += 1
            b["ok"] += int(c["ok"])
            b["mean_ms"] += c["latency_ms"]
        for b in by.values():
            b["mean_ms"] = round(b["mean_ms"] / max(b["calls"], 1), 1)
        return {"total_calls": len(calls),
                "distinct_tools": len(by),
                "success_rate": round(sum(int(c["ok"]) for c in calls) / len(calls), 3)
                                if calls else 0.0,
                "handled_failures": len(self.failure_handled),
                "by_tool": by}


# --------------------------------------------------------------- code tasks
CODE_TASKS: dict[str, dict[str, Any]] = {
    "two_sum": {
        "entry": "two_sum",
        "prompt": ("Write `def two_sum(nums: list[int], target: int) -> list[int]` returning the "
                   "indices of the two numbers adding to target. Assume exactly one solution. "
                   "Aim for O(n)."),
        "tests": [
            {"args": [[2, 7, 11, 15], 9], "expect": [0, 1]},
            {"args": [[3, 2, 4], 6], "expect": [1, 2]},
            {"args": [[3, 3], 6], "expect": [0, 1]},
            {"args": [[-1, -2, -3, -4], -7], "expect": [2, 3]},
        ],
    },
    "dedupe_stable": {
        "entry": "dedupe",
        "prompt": ("Write `def dedupe(items: list) -> list` that removes duplicates while "
                   "preserving first-seen order."),
        "tests": [
            {"args": [[1, 1, 2, 3, 2]], "expect": [1, 2, 3]},
            {"args": [["a", "b", "a"]], "expect": ["a", "b"]},
            {"args": [[]], "expect": []},
        ],
    },
    "rate_limiter": {
        "entry": "allowed_count",
        "prompt": ("Write `def allowed_count(timestamps: list[int], window: int, limit: int) -> int` "
                   "returning how many requests a sliding-window rate limiter would ALLOW."),
        "tests": [
            {"args": [[1, 2, 3, 4, 5], 3, 2], "expect": 2},
            {"args": [[1, 10, 20], 3, 2], "expect": 3},
            {"args": [[], 5, 1], "expect": 0},
        ],
    },
}

_HARNESS = '''
if __name__ == "__main__":
    import json as _j
    _tests = _j.loads(r"""{tests}""")
    _passed, _cases = 0, []
    _fn = globals().get("{entry}")
    if _fn is None:
        print(_j.dumps({{"passed": 0, "total": len(_tests),
                         "error": "missing function {entry}", "cases": []}}))
    else:
        for _t in _tests:
            try:
                _got = _fn(*_t["args"])
                _ok = _got == _t["expect"]
            except Exception as _e:
                _got, _ok = f"{{type(_e).__name__}}: {{_e}}", False
            _passed += int(_ok)
            _cases.append({{"args": _t["args"], "expect": _t["expect"],
                            "got": _got, "ok": _ok}})
        print(_j.dumps({{"passed": _passed, "total": len(_tests), "cases": _cases}}))
'''
