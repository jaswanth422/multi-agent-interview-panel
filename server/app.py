"""FastAPI demo server: streams the whole agentic loop to the browser.

  GET  /                     the demo UI
  GET  /api/cases            labelled evaluation cases
  POST /api/start            start an interview (returns session_id)
  WS   /ws/{session_id}      live event stream
  POST /api/answer           supply a human candidate's answer
  POST /api/approve          the human approval checkpoint
  GET  /api/report/{sid}     full JSON artifact
  GET  /api/sessions         past sessions (memory persists across runs)
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit.cases import CASES, get_case  # noqa: E402
from panel.candidate import HumanCandidate, ScriptedCandidate  # noqa: E402
from panel.config import ARTIFACTS, SETTINGS  # noqa: E402
from panel.llm import LLM  # noqa: E402
from panel.memory import Memory  # noqa: E402
from panel.orchestrator import InterviewConfig, Orchestrator  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="Multi-Agent Interview Panel")

MEM = Memory()
RUNS: dict[str, dict[str, Any]] = {}


class StartReq(BaseModel):
    case_id: str = "C01"
    rounds: int = 0
    mode: str = "simulated"          # simulated | human
    candidate_name: str = ""
    role: str = ""
    speech: bool = False


class AnswerReq(BaseModel):
    session_id: str
    text: str


class ApproveReq(BaseModel):
    session_id: str
    decision: str
    by: str = "human-reviewer"
    note: str = ""


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/cases")
def cases():
    return [{"id": c["id"], "name": c["name"], "role": c["role"],
             "seniority": c["seniority"], "expected": c["expected_decision"],
             "planted": len(c["ground_truth"]), "persona": c["persona"],
             "notes": c.get("notes", ""),
             "ground_truth": c["ground_truth"]}
            for c in CASES]


@app.get("/api/sessions")
def sessions():
    return MEM.list_sessions(30)


@app.get("/api/config")
def config():
    llm = LLM()
    return {"llm": llm.describe(), "rounds": SETTINGS.rounds,
            "hire_threshold": SETTINGS.hire_threshold,
            "require_approval": SETTINGS.require_human_approval,
            "debate_rounds": SETTINGS.debate_rounds}


@app.post("/api/start")
def start(req: StartReq):
    case = get_case(req.case_id)
    llm = LLM()
    q: queue.Queue = queue.Queue()

    orch = Orchestrator(
        llm=llm, memory=MEM, settings=SETTINGS,
        emit_fn=lambda phase, kind, payload: q.put(
            {"phase": phase, "kind": kind, "data": payload}))

    cfg = InterviewConfig(
        candidate_name=req.candidate_name or case["name"],
        role=req.role or case["role"], seniority=case["seniority"],
        rounds=req.rounds or SETTINGS.rounds, enable_speech=req.speech)

    sid = orch.setup(cfg, case["resume"])

    if req.mode == "human":
        cand: Any = HumanCandidate()
        answer_gate = threading.Event()
        holder = {"text": ""}

        class Gated:
            def answer(self, question, *, round_=0, asked_by="", history=None):
                q.put({"phase": "interview", "kind": "awaiting_human_answer",
                       "data": {"question": question, "round": round_,
                                "asked_by": asked_by}})
                answer_gate.wait()
                answer_gate.clear()
                return holder["text"] or "(no answer)"

        cand = Gated()
        RUNS[sid] = {"orch": orch, "queue": q, "gate": answer_gate,
                     "holder": holder, "case": case, "done": False}
    else:
        cand = ScriptedCandidate(
            name=cfg.candidate_name, persona=case["persona"],
            resume=case["resume"], bank=case["bank"], llm=llm,
            code_answer=case.get("code_answer", ""))
        RUNS[sid] = {"orch": orch, "queue": q, "case": case, "done": False}

    def runner():
        try:
            orch.run_interview(cand)
        except Exception as e:  # noqa: BLE001
            q.put({"phase": "error", "kind": "run_failed",
                   "data": {"error": f"{type(e).__name__}: {e}"}})
        finally:
            RUNS[sid]["done"] = True
            q.put({"phase": "interview", "kind": "run_complete",
                   "data": {"session_id": sid}})

    t = threading.Thread(target=runner, daemon=True)
    RUNS[sid]["thread"] = t
    t.start()

    return {"session_id": sid, "case": case["id"],
            "ground_truth": case["ground_truth"],
            "expected_decision": case["expected_decision"],
            "llm": llm.describe()}


@app.post("/api/answer")
def answer(req: AnswerReq):
    run = RUNS.get(req.session_id)
    if not run or "gate" not in run:
        raise HTTPException(404, "no interactive run for that session")
    run["holder"]["text"] = req.text
    run["gate"].set()
    return {"ok": True}


@app.post("/api/approve")
def approve(req: ApproveReq):
    run = RUNS.get(req.session_id)
    if req.decision not in ("approve", "override", "reject"):
        raise HTTPException(400, "decision must be approve|override|reject")
    if not run:
        ok = MEM.record_approval(req.session_id, req.decision, req.by, req.note)
        if not ok:
            raise HTTPException(404, "unknown session")
        return {"ok": True, "persisted_only": True}
    report = run["orch"].record_approval(req.decision, req.by, req.note)
    return {"ok": True, "decision": req.decision,
            "approval_status": report["verdict"]["approval_status"]}


@app.get("/compare")
def compare():
    return FileResponse(STATIC / "board.html")


@app.get("/api/board")
def board():
    """Hiring-manager view: every completed screen, newest per candidate.

    Reads real sessions straight out of SQLite - nothing precomputed and
    nothing staged. The gate label is re-derived from the stored evidence
    using the same precedence as the consensus rules, so what you see is
    why the decision actually happened.
    """
    db = MEM.db
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    rows = MEM._q(
        "SELECT s.id sid, s.candidate_name n, s.role, s.seniority, "
        "v.decision, v.confidence, v.rationale, v.approval_status, "
        "v.approved_by, v.per_agent_json, v.ts "
        "FROM verdicts v JOIN sessions s ON s.id = v.session_id "
        "ORDER BY v.ts DESC")
    for r in rows:
        if r["n"] in seen:
            continue
        seen.add(r["n"])
        sid = r["sid"]

        contras = MEM.contradictions(sid)
        cm = MEM.claim_map(sid)
        high = [x for x in contras if x.severity == "high" and x.status != "resolved"]
        guard = MEM.guardrail_events(sid)
        inj = [g for g in guard if g["kind"] == "prompt_injection"]
        refus = [g for g in guard if g["kind"] == "prohibited_question"]

        code_verdict = ""
        for tc in MEM.tool_calls(sid):
            if tc["tool"] == "code_exec" and isinstance(tc.get("result"), dict):
                code_verdict = str(tc["result"].get("verdict", ""))

        # Same precedence as VerdictEngine.consensus()
        if inj:
            gate, why = "G1", f"{len(inj)} attempt(s) to manipulate the evaluators"
        elif high:
            types = ", ".join(sorted({x.ctype for x in high}))
            gate, why = "G2", f"{len(high)} unresolved high-severity contradiction(s): {types}"
        elif code_verdict in ("refused_unsafe_code", "fail"):
            gate, why = "G3", ("submitted code refused by the sandbox as unsafe"
                               if code_verdict == "refused_unsafe_code"
                               else "submitted code failed every hidden test")
        elif r["decision"] == "hire":
            gate, why = "S1", "clean record, panel scores cleared the bar"
        else:
            gate, why = "S2", "clean record but below the seniority bar"

        headline = None
        if contras:
            top = max(contras, key=lambda x: x.score)
            a, b = cm.get(top.claim_a), cm.get(top.claim_b)
            headline = {
                "type": top.ctype, "severity": top.severity,
                "rationale": top.rationale,
                "said_first": (b.raw_span if b else ""), "first_round": (b.round if b else None),
                "said_later": (a.raw_span if a else ""), "later_round": (a.round if a else None),
            }

        per_agent = json.loads(r["per_agent_json"] or "{}")
        out.append({
            "session_id": sid, "candidate": r["n"], "role": r["role"],
            "seniority": r["seniority"], "decision": r["decision"],
            "confidence": r["confidence"], "rationale": r["rationale"],
            "approval_status": r["approval_status"], "approved_by": r["approved_by"],
            "gate": gate, "gate_reason": why,
            "contradictions": len(contras), "high_severity": len(high),
            "confronted_live": len([x for x in contras if x.status == "addressed"]),
            "injections": len(inj), "refusals": len(refus),
            "code_verdict": code_verdict,
            "claims": len(cm), "turns": len(MEM.transcript(sid)),
            "distinct_tools": len({t["tool"] for t in MEM.tool_calls(sid)}),
            "headline_contradiction": headline,
            "panel": {k: {"overall": v.get("overall"),
                          "recommendation": v.get("recommendation"),
                          "label_overridden": v.get("label_overridden", False)}
                      for k, v in per_agent.items()},
        })
    return out


@app.get("/api/report/{sid}")
def report(sid: str):
    run = RUNS.get(sid)
    if run:
        return JSONResponse(run["orch"].build_report())
    data = MEM.export(sid)
    if not data["session"]:
        raise HTTPException(404, "unknown session")
    return JSONResponse(data)


@app.websocket("/ws/{sid}")
async def ws(sock: WebSocket, sid: str):
    await sock.accept()
    run = RUNS.get(sid)
    if not run:
        await sock.send_json({"phase": "error", "kind": "unknown_session",
                              "data": {"session_id": sid}})
        await sock.close()
        return

    q: queue.Queue = run["queue"]
    loop = asyncio.get_running_loop()
    try:
        while True:
            try:
                ev = await loop.run_in_executor(None, q.get, True, 0.5)
            except queue.Empty:
                if run.get("done") and q.empty():
                    await asyncio.sleep(0.3)
                    if q.empty():
                        break
                continue
            await sock.send_json(ev)
            if ev["kind"] == "run_complete":
                break
    except WebSocketDisconnect:
        return
    except Exception:  # noqa: BLE001
        pass
    finally:
        try:
            await sock.close()
        except Exception:  # noqa: BLE001
            pass


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
