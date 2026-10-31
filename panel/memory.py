"""Durable interview memory: SQLite-backed claim ledger + full audit trail.

This is the system's spine. Every question, answer, atomic claim, tool call,
guardrail event and verdict is persisted, so state survives across steps AND
across sessions (resume an interview by id), and every decision is traceable
back to the claim that justified it.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import SETTINGS

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  candidate_name TEXT, role TEXT, seniority TEXT,
  status TEXT, created_at REAL, updated_at REAL,
  config_json TEXT, resume_json TEXT
);
CREATE TABLE IF NOT EXISTS turns (
  id TEXT PRIMARY KEY, session_id TEXT, round INTEGER, seq INTEGER,
  agent TEXT, kind TEXT, text TEXT, ts REAL, meta_json TEXT
);
CREATE TABLE IF NOT EXISTS claims (
  id TEXT PRIMARY KEY, session_id TEXT, turn_id TEXT, round INTEGER,
  source TEXT,                 -- 'resume' | 'interview'
  subject TEXT, predicate TEXT, object TEXT,
  polarity INTEGER,            -- 1 asserted, -1 negated
  value_num REAL, unit TEXT,
  ownership INTEGER,           -- 0..4 ladder, NULL if n/a
  topic TEXT, confidence REAL, raw_span TEXT, ts REAL
);
CREATE TABLE IF NOT EXISTS contradictions (
  id TEXT PRIMARY KEY, session_id TEXT, claim_a TEXT, claim_b TEXT,
  ctype TEXT, severity TEXT, score REAL, rationale TEXT,
  detected_round INTEGER, detector TEXT, status TEXT,
  resolution TEXT, evidence_json TEXT, ts REAL
);
CREATE TABLE IF NOT EXISTS tool_calls (
  id TEXT PRIMARY KEY, session_id TEXT, round INTEGER, agent TEXT,
  tool TEXT, args_json TEXT, result_json TEXT,
  ok INTEGER, error TEXT, latency_ms REAL, ts REAL
);
CREATE TABLE IF NOT EXISTS guardrail_events (
  id TEXT PRIMARY KEY, session_id TEXT, round INTEGER, kind TEXT,
  severity TEXT, trigger_text TEXT, action TEXT, detail TEXT, ts REAL
);
CREATE TABLE IF NOT EXISTS verdicts (
  id TEXT PRIMARY KEY, session_id TEXT, decision TEXT, confidence REAL,
  rationale TEXT, per_agent_json TEXT, evidence_json TEXT, dissent_json TEXT,
  debate_json TEXT, approval_status TEXT, approved_by TEXT,
  approval_note TEXT, approved_at REAL, ts REAL
);
CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY, session_id TEXT, phase TEXT, kind TEXT,
  payload_json TEXT, ts REAL
);
CREATE INDEX IF NOT EXISTS idx_turns_s ON turns(session_id, seq);
CREATE INDEX IF NOT EXISTS idx_claims_s ON claims(session_id);
CREATE INDEX IF NOT EXISTS idx_contra_s ON contradictions(session_id);
CREATE INDEX IF NOT EXISTS idx_tools_s ON tool_calls(session_id);
CREATE INDEX IF NOT EXISTS idx_events_s ON events(session_id, ts);
"""


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


# ---------------------------------------------------------------- models
@dataclass
class Claim:
    """An atomic, comparable assertion extracted from one answer."""
    subject: str                  # 'self' | 'team' | project/system name
    predicate: str                # normalized: role_on, team_size, years_exp, used_tech, metric, outcome
    object: str = ""
    polarity: int = 1
    value_num: float | None = None
    unit: str = ""
    ownership: int | None = None  # 0 observed .. 4 led
    topic: str = ""
    confidence: float = 0.8
    raw_span: str = ""
    source: str = "interview"
    id: str = field(default_factory=lambda: _id("c"))
    turn_id: str = ""
    round: int = 0

    def key(self) -> tuple[str, str, str]:
        return (self.subject.lower().strip(), self.predicate.lower().strip(),
                self.object.lower().strip())

    def short(self) -> str:
        bits = [self.subject, self.predicate]
        if self.object:
            bits.append(self.object)
        if self.value_num is not None:
            bits.append(f"{self.value_num:g}{self.unit}")
        s = " / ".join(b for b in bits if b)
        return ("NOT " if self.polarity < 0 else "") + s


@dataclass
class Contradiction:
    claim_a: str
    claim_b: str
    ctype: str
    severity: str = "medium"
    score: float = 0.5
    rationale: str = ""
    detected_round: int = 0
    detector: str = "rules+llm"
    status: str = "open"
    resolution: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: _id("x"))


# ---------------------------------------------------------------- store
class Memory:
    def __init__(self, db_path: str | None = None):
        self.path = db_path or SETTINGS.db_path
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        with self._lock:
            self.db.executescript(SCHEMA)
            self.db.commit()

    # -- helpers
    def _w(self, sql: str, params: Iterable[Any] = ()):
        with self._lock:
            cur = self.db.execute(sql, tuple(params))
            self.db.commit()
            return cur

    def _q(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self.db.execute(sql, tuple(params)))

    # -- sessions
    def create_session(self, candidate_name: str, role: str, seniority: str,
                       config: dict[str, Any], resume: dict[str, Any]) -> str:
        sid = _id("s")
        now = time.time()
        self._w(
            "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?)",
            (sid, candidate_name, role, seniority, "running", now, now,
             json.dumps(config), json.dumps(resume)),
        )
        return sid

    def set_status(self, sid: str, status: str):
        self._w("UPDATE sessions SET status=?, updated_at=? WHERE id=?",
                (status, time.time(), sid))

    def get_session(self, sid: str) -> dict[str, Any] | None:
        rows = self._q("SELECT * FROM sessions WHERE id=?", (sid,))
        if not rows:
            return None
        d = dict(rows[0])
        d["config"] = json.loads(d.pop("config_json") or "{}")
        d["resume"] = json.loads(d.pop("resume_json") or "{}")
        return d

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        return [dict(r) for r in self._q(
            "SELECT id,candidate_name,role,status,created_at FROM sessions "
            "ORDER BY created_at DESC LIMIT ?", (limit,))]

    # -- turns
    def add_turn(self, sid: str, round_: int, agent: str, kind: str, text: str,
                 meta: dict[str, Any] | None = None) -> str:
        tid = _id("t")
        seq = len(self._q("SELECT id FROM turns WHERE session_id=?", (sid,))) + 1
        self._w("INSERT INTO turns VALUES (?,?,?,?,?,?,?,?,?)",
                (tid, sid, round_, seq, agent, kind, text, time.time(),
                 json.dumps(meta or {})))
        return tid

    def transcript(self, sid: str) -> list[dict[str, Any]]:
        out = []
        for r in self._q("SELECT * FROM turns WHERE session_id=? ORDER BY seq", (sid,)):
            d = dict(r)
            d["meta"] = json.loads(d.pop("meta_json") or "{}")
            out.append(d)
        return out

    # -- claims
    def add_claims(self, sid: str, claims: list[Claim]) -> list[Claim]:
        for c in claims:
            self._w(
                "INSERT INTO claims VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (c.id, sid, c.turn_id, c.round, c.source, c.subject, c.predicate,
                 c.object, c.polarity, c.value_num, c.unit, c.ownership, c.topic,
                 c.confidence, c.raw_span, time.time()),
            )
        return claims

    def claims(self, sid: str) -> list[Claim]:
        out = []
        for r in self._q("SELECT * FROM claims WHERE session_id=? ORDER BY round, ts", (sid,)):
            out.append(Claim(
                id=r["id"], turn_id=r["turn_id"], round=r["round"], source=r["source"],
                subject=r["subject"], predicate=r["predicate"], object=r["object"],
                polarity=r["polarity"], value_num=r["value_num"], unit=r["unit"] or "",
                ownership=r["ownership"], topic=r["topic"] or "",
                confidence=r["confidence"] or 0.0, raw_span=r["raw_span"] or "",
            ))
        return out

    def claim_map(self, sid: str) -> dict[str, Claim]:
        return {c.id: c for c in self.claims(sid)}

    # -- contradictions
    def add_contradiction(self, sid: str, x: Contradiction) -> Contradiction:
        self._w("INSERT INTO contradictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (x.id, sid, x.claim_a, x.claim_b, x.ctype, x.severity, x.score,
                 x.rationale, x.detected_round, x.detector, x.status, x.resolution,
                 json.dumps(x.evidence), time.time()))
        return x

    def contradictions(self, sid: str) -> list[Contradiction]:
        out = []
        for r in self._q("SELECT * FROM contradictions WHERE session_id=? ORDER BY ts", (sid,)):
            out.append(Contradiction(
                id=r["id"], claim_a=r["claim_a"], claim_b=r["claim_b"], ctype=r["ctype"],
                severity=r["severity"], score=r["score"] or 0.0, rationale=r["rationale"] or "",
                detected_round=r["detected_round"] or 0, detector=r["detector"] or "",
                status=r["status"] or "open", resolution=r["resolution"] or "",
                evidence=json.loads(r["evidence_json"] or "{}"),
            ))
        return out

    def resolve_contradiction(self, xid: str, status: str, resolution: str):
        self._w("UPDATE contradictions SET status=?, resolution=? WHERE id=?",
                (status, resolution, xid))

    def has_pair(self, sid: str, a: str, b: str) -> bool:
        return bool(self._q(
            "SELECT id FROM contradictions WHERE session_id=? AND "
            "((claim_a=? AND claim_b=?) OR (claim_a=? AND claim_b=?))", (sid, a, b, b, a)))

    # -- tool calls
    def log_tool(self, sid: str, round_: int, agent: str, tool: str,
                 args: dict[str, Any], result: Any, ok: bool,
                 error: str, latency_ms: float) -> str:
        tid = _id("tc")
        self._w("INSERT INTO tool_calls VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (tid, sid, round_, agent, tool, json.dumps(args, default=str),
                 json.dumps(result, default=str)[:20000], 1 if ok else 0,
                 error, latency_ms, time.time()))
        return tid

    def tool_calls(self, sid: str) -> list[dict[str, Any]]:
        out = []
        for r in self._q("SELECT * FROM tool_calls WHERE session_id=? ORDER BY ts", (sid,)):
            d = dict(r)
            d["args"] = json.loads(d.pop("args_json") or "{}")
            try:
                d["result"] = json.loads(d.pop("result_json") or "null")
            except json.JSONDecodeError:
                d["result"] = None
            out.append(d)
        return out

    # -- guardrails
    def log_guardrail(self, sid: str, round_: int, kind: str, severity: str,
                      trigger_text: str, action: str, detail: str) -> str:
        gid = _id("g")
        self._w("INSERT INTO guardrail_events VALUES (?,?,?,?,?,?,?,?,?)",
                (gid, sid, round_, kind, severity, trigger_text[:600], action,
                 detail, time.time()))
        return gid

    def guardrail_events(self, sid: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self._q(
            "SELECT * FROM guardrail_events WHERE session_id=? ORDER BY ts", (sid,))]

    # -- verdict
    def save_verdict(self, sid: str, v: dict[str, Any]) -> str:
        vid = _id("v")
        approval = "pending" if SETTINGS.require_human_approval else "auto_approved"
        self._w("INSERT INTO verdicts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (vid, sid, v["decision"], v.get("confidence", 0.0),
                 v.get("rationale", ""), json.dumps(v.get("per_agent", {})),
                 json.dumps(v.get("evidence", [])), json.dumps(v.get("dissent", [])),
                 json.dumps(v.get("debate", [])), approval, "", "", None, time.time()))
        return vid

    def verdict(self, sid: str) -> dict[str, Any] | None:
        rows = self._q("SELECT * FROM verdicts WHERE session_id=? ORDER BY ts DESC LIMIT 1", (sid,))
        if not rows:
            return None
        d = dict(rows[0])
        for src, dst in (("per_agent_json", "per_agent"), ("evidence_json", "evidence"),
                         ("dissent_json", "dissent"), ("debate_json", "debate")):
            d[dst] = json.loads(d.pop(src) or "null")
        return d

    def record_approval(self, sid: str, decision: str, by: str, note: str) -> bool:
        rows = self._q("SELECT id FROM verdicts WHERE session_id=? ORDER BY ts DESC LIMIT 1", (sid,))
        if not rows:
            return False
        self._w("UPDATE verdicts SET approval_status=?, approved_by=?, approval_note=?, "
                "approved_at=? WHERE id=?",
                (decision, by, note, time.time(), rows[0]["id"]))
        return True

    # -- event trace (the agentic loop is fully replayable from this)
    def log_event(self, sid: str, phase: str, kind: str, payload: dict[str, Any]) -> str:
        eid = _id("e")
        self._w("INSERT INTO events VALUES (?,?,?,?,?,?)",
                (eid, sid, phase, kind, json.dumps(payload, default=str)[:20000], time.time()))
        return eid

    def events(self, sid: str) -> list[dict[str, Any]]:
        out = []
        for r in self._q("SELECT * FROM events WHERE session_id=? ORDER BY ts", (sid,)):
            d = dict(r)
            d["payload"] = json.loads(d.pop("payload_json") or "{}")
            out.append(d)
        return out

    # -- export
    def export(self, sid: str) -> dict[str, Any]:
        cm = self.claim_map(sid)

        def decorate(x: Contradiction) -> dict[str, Any]:
            d = asdict(x)
            a, b = cm.get(x.claim_a), cm.get(x.claim_b)
            d["claim_a_text"] = a.short() if a else x.claim_a
            d["claim_b_text"] = b.short() if b else x.claim_b
            d["claim_a_span"] = a.raw_span if a else ""
            d["claim_b_span"] = b.raw_span if b else ""
            d["claim_a_round"] = a.round if a else None
            d["claim_b_round"] = b.round if b else None
            return d

        return {
            "session": self.get_session(sid),
            "transcript": self.transcript(sid),
            "claims": [asdict(c) for c in self.claims(sid)],
            "contradictions": [decorate(x) for x in self.contradictions(sid)],
            "tool_calls": self.tool_calls(sid),
            "guardrail_events": self.guardrail_events(sid),
            "verdict": self.verdict(sid),
            "events": self.events(sid),
        }
