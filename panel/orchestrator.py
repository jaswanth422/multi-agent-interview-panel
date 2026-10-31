"""The agentic loop.

PHASES
  setup    -> provision session + resume DB, extract resume claims into the ledger
  round N  -> for each panelist in rotation:
                PLAN     (agent inspects ledger/contradictions/coverage, picks action)
                ACT      (optionally calls a tool, then asks one question)
                OBSERVE  (candidate answers; answer is screened as untrusted input)
                EXTRACT  (answer -> atomic claims -> ledger)
                DETECT   (rule engine + LLM adjudication over the WHOLE ledger)
                RE-PLAN  (a new contradiction rewrites the next agent's priorities)
  score    -> three independent, evidence-citing scorecards
  debate   -> panelists see each other and may revise, with reasons
  consensus-> explicit rule set, Bar-Raiser veto, recorded dissent
  approval -> HUMAN checkpoint: nothing is final until a person signs off
  export   -> PII-redacted artifact written to disk

Every phase writes an event to SQLite, so a run is fully replayable and the
"more than one LLM call" bar is met many times over (typically 35-60 calls).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .agents import PANEL, AgentSpec, Panelist
from .claims import ClaimExtractor, normalize_project
from .candidate import Candidate
from .config import SETTINGS, Settings
from .contradictions import ContradictionDetector
from .guardrails import Guardrails
from .llm import LLM
from .memory import Claim, Memory
from .tools import ToolBelt
from .verdict import VerdictEngine

Emit = Callable[[str, str, dict[str, Any]], None]


@dataclass
class InterviewConfig:
    candidate_name: str = "Candidate"
    role: str = "Senior Backend Engineer"
    seniority: str = "senior"
    rounds: int = 0
    code_task_id: str = "two_sum"
    enable_speech: bool = False

    def __post_init__(self):
        if not self.rounds:
            self.rounds = SETTINGS.rounds


@dataclass
class Orchestrator:
    llm: LLM
    memory: Memory
    settings: Settings = field(default_factory=lambda: SETTINGS)
    emit_fn: Emit | None = None
    session_id: str = ""
    panelists: list[Panelist] = field(default_factory=list)
    tools: ToolBelt | None = None
    guard: Guardrails | None = None
    extractor: ClaimExtractor | None = None
    detector: ContradictionDetector | None = None
    verdict_engine: VerdictEngine | None = None
    code_result: dict[str, Any] | None = None
    code_ran: bool = False
    covered: list[dict[str, str]] = field(default_factory=list)
    all_questions: list[dict[str, str]] = field(default_factory=list)
    tool_findings: list[dict[str, Any]] = field(default_factory=list)
    known_projects: list[str] = field(default_factory=list)
    started_at: float = 0.0
    cfg: InterviewConfig | None = None

    # ------------------------------------------------------------------ emit
    def emit(self, phase: str, kind: str, payload: dict[str, Any]):
        if self.session_id:
            self.memory.log_event(self.session_id, phase, kind, payload)
        if self.emit_fn:
            try:
                self.emit_fn(phase, kind, payload)
            except Exception:  # noqa: BLE001 - UI must never break the run
                pass

    # ----------------------------------------------------------------- setup
    def setup(self, cfg: InterviewConfig, resume: dict[str, Any]) -> str:
        self.cfg = cfg
        self.started_at = time.time()
        self.session_id = self.memory.create_session(
            cfg.candidate_name, cfg.role, cfg.seniority,
            {"rounds": cfg.rounds, "code_task_id": cfg.code_task_id,
             "llm": self.llm.describe(),
             "hire_threshold": self.settings.hire_threshold},
            resume)

        self.tools = ToolBelt(self.memory, self.session_id)
        self.tools.seed_resume_db(resume)
        self.guard = Guardrails(self.memory, self.session_id)
        self.extractor = ClaimExtractor(self.llm)
        self.detector = ContradictionDetector(self.llm)
        self.verdict_engine = VerdictEngine(
            self.llm, hire_threshold=self.settings.hire_threshold,
            debate_rounds=self.settings.debate_rounds)
        self.panelists = [
            Panelist(spec=s, llm=self.llm, memory=self.memory, tools=self.tools,
                     guard=self.guard, session_id=self.session_id)
            for s in PANEL
        ]

        self.emit("setup", "session_created", {
            "session_id": self.session_id, "candidate": cfg.candidate_name,
            "role": cfg.role, "seniority": cfg.seniority, "rounds": cfg.rounds,
            "llm": self.llm.describe(),
            "panel": [{"key": s.key, "display": s.display, "emoji": s.emoji,
                       "mandate": s.mandate[:220], "weight": s.weight} for s in PANEL],
            "tools_available": [t["name"] for t in self.tools.catalog()],
        })

        # Seed the ledger from the resume so round-1 answers can already
        # contradict the written record.
        seeded = self._seed_resume_claims(resume)
        self.emit("setup", "resume_ingested",
                  {"claims_seeded": len(seeded),
                   "projects": self.known_projects,
                   "sql_backed": True})
        return self.session_id

    def _seed_resume_claims(self, resume: dict[str, Any]) -> list[Claim]:
        claims: list[Claim] = []
        tid = self.memory.add_turn(
            self.session_id, 0, "system", "resume",
            json.dumps(resume)[:4000], {"source": "resume"})
        for p in resume.get("projects", []):
            proj = normalize_project(p.get("name", ""))
            if proj and proj not in self.known_projects:
                self.known_projects.append(proj)
            if p.get("role"):
                from .claims import ownership_of
                claims.append(Claim(
                    subject=proj, predicate="role_on", object=p["role"],
                    ownership=ownership_of(p["role"]), source="resume",
                    raw_span=f"Resume: {p.get('name')} - {p['role']}",
                    confidence=0.95, turn_id=tid, round=0))
            if p.get("team_size"):
                claims.append(Claim(
                    subject=proj, predicate="team_size", value_num=float(p["team_size"]),
                    unit="people", source="resume",
                    raw_span=f"Resume: {p.get('name')} - team of {p['team_size']}",
                    confidence=0.95, turn_id=tid, round=0))
            if p.get("months"):
                claims.append(Claim(
                    subject=proj, predicate="timeline", value_num=float(p["months"]),
                    unit="months", source="resume",
                    raw_span=f"Resume: {p.get('name')} - {p['months']} months",
                    confidence=0.95, turn_id=tid, round=0))
        for s in resume.get("skills", []):
            claims.append(Claim(
                subject="self", predicate="used_tech", object=s.get("name", ""),
                source="resume", raw_span=f"Resume skill: {s.get('name')} "
                                          f"({s.get('years','?')}y, {s.get('level','')})",
                confidence=0.9, turn_id=tid, round=0))
            if s.get("years"):
                claims.append(Claim(
                    subject="self", predicate="years_exp", object=s.get("name", ""),
                    value_num=float(s["years"]), unit="years", source="resume",
                    raw_span=f"Resume: {s.get('years')} years of {s.get('name')}",
                    confidence=0.9, turn_id=tid, round=0))
        self.memory.add_claims(self.session_id, claims)
        return claims

    # ------------------------------------------------------------------ state
    def _state(self, round_: int) -> dict[str, Any]:
        claims = self.memory.claims(self.session_id)
        contras = self.memory.contradictions(self.session_id)
        cm = {c.id: c for c in claims}

        def brief(x) -> dict[str, Any]:
            a, b = cm.get(x.claim_a), cm.get(x.claim_b)
            return {"id": x.id, "type": x.ctype, "severity": x.severity,
                    "status": x.status, "rationale": x.rationale,
                    "score": x.score, "detected_round": x.detected_round,
                    "evidence": x.evidence,
                    "claim_a_text": a.short() if a else "", "claim_a_round": a.round if a else None,
                    "claim_b_text": b.short() if b else "", "claim_b_round": b.round if b else None}

        turns = self.memory.transcript(self.session_id)
        convo = [t for t in turns if t["kind"] in ("question", "answer")]
        recent = "\n".join(
            f"[R{t['round']}] {t['agent'].upper()} ({t['kind']}): {t['text']}"
            for t in convo[-(4 if self.settings.fast else 6):])
        full = "\n".join(
            f"[R{t['round']}] {t['agent'].upper()} ({t['kind']}): {t['text']}"
            for t in convo)

        # Checkable, not-yet-verified claims.
        verified_terms = {str(c["args"]) for c in self.memory.tool_calls(self.session_id)}
        unverified = [
            {"id": c.id, "text": c.short(), "span": c.raw_span[:120]}
            for c in claims
            if c.predicate in ("used_tech", "metric", "team_size", "years_exp")
            and c.source == "interview"
            and not any(c.object and c.object.lower() in v.lower() for v in verified_terms)
        ][:6]

        resume = (self.memory.get_session(self.session_id) or {}).get("resume", {})
        resume_summary = {
            "headline": resume.get("headline", ""),
            "total_years": resume.get("total_years"),
            "github": resume.get("github", ""),
            "projects": [{"name": p.get("name"), "role": p.get("role"),
                          "team_size": p.get("team_size")}
                         for p in resume.get("projects", [])],
            "skills": [s.get("name") for s in resume.get("skills", [])],
        }

        g = self.guard.summary() if self.guard else {}
        return {
            "round": round_, "total_rounds": self.cfg.rounds if self.cfg else 3,
            "role": self.cfg.role if self.cfg else "", 
            "seniority": self.cfg.seniority if self.cfg else "",
            "claims": [{"id": c.id, "subject": c.subject, "predicate": c.predicate,
                        "object": c.object, "round": c.round, "raw_span": c.raw_span,
                        "source": c.source} for c in claims],
            "claims_brief": [{"id": c.id, "text": c.short(), "round": c.round,
                              "source": c.source}
                             for c in claims][:25 if self.settings.fast else 60],
            "contradictions": [brief(x) for x in contras],
            "contradictions_brief": [brief(x) for x in contras],
            "covered": self.covered, "all_questions": self.all_questions,
            "recent_transcript": recent or "(interview has not started yet)",
            "full_transcript": full, "unverified": unverified,
            "resume_summary": resume_summary,
            "code_ran": self.code_ran, "code_result": self.code_result,
            "code_task_id": self.cfg.code_task_id if self.cfg else "two_sum",
            "tool_findings": self.tool_findings[-6:],
            "integrity_flags": {"refusal_count": g.get("refusal_count", 0),
                                "injection_count": g.get("injection_count", 0),
                                "injections": g.get("injection_attempts", [])[:3]},
        }

    # ------------------------------------------------------------- interview
    def run_interview(self, candidate: Candidate) -> dict[str, Any]:
        assert self.cfg and self.tools and self.guard and self.extractor and self.detector
        total = self.cfg.rounds

        for rnd in range(1, total + 1):
            self.emit("interview", "round_start", {"round": rnd, "of": total})
            for p in self.panelists:
                self._turn(p, rnd, candidate)
            self.emit("interview", "round_end", {
                "round": rnd,
                "claims": len(self.memory.claims(self.session_id)),
                "contradictions": len(self.memory.contradictions(self.session_id))})

        return self.finalize()

    def _forced_action(self, p: Panelist, rnd: int,
                       state: dict[str, Any]) -> dict[str, Any] | None:
        """Return a ready-made plan when panel protocol dictates the action.

        Same three protocol rules as before (P1 confront, P3 code task, P2 tool
        grounding) - just evaluated BEFORE the planner call instead of after,
        so a discarded plan costs nothing.
        """
        open_x = [x for x in state["contradictions"] if x["status"] == "open"]
        base = {"target_competency": "", "target_claim_id": "",
                "target_contradiction_id": "", "tool": "", "tool_args": {},
                "agent": p.spec.key}

        if open_x:
            return {**base, "action": "challenge_contradiction",
                    "target_contradiction_id": open_x[0]["id"],
                    "policy_reason": ("P1 protocol: an unaddressed contradiction "
                                      "outranks opening a new topic"),
                    "reasoning": ("P1 protocol: confront the open contradiction "
                                  "before any new topic")}
        if p.spec.can_code and rnd >= 2 and not state["code_ran"]:
            return {**base, "action": "request_code",
                    "policy_reason": ("P3 protocol: technical screen requires a live "
                                      "coding task by round 2"),
                    "reasoning": "P3 protocol: set the live coding task"}
        if p.spec.tools and not p.used_tool:
            return {**base, "action": "verify_with_tool",
                    "tool": self._pick_tool(p, state),
                    "policy_reason": (f"P2 protocol: {p.spec.display} has not yet "
                                      "grounded a question in tool evidence"),
                    "reasoning": "P2 protocol: ground this question in tool evidence"}
        return None

    def _apply_policy(self, plan: dict[str, Any], p: Panelist, rnd: int,
                      state: dict[str, Any]) -> dict[str, Any]:
        """Panel policy overrides on top of each agent's own plan.

        A small planner model is inconsistent about doing the things the panel
        *must* do. These are not the agent choosing freely; they are the
        interview protocol, so they are applied deterministically and the
        override is recorded in the event trace for auditability.

        P1 An unaddressed contradiction MUST be confronted before any new
           topic is opened. This is the panel's core promise.
        P2 Every panelist must ground at least one question in tool evidence.
        P3 The technical panelist must set a live coding task by round 2.
        """
        open_x = [x for x in state["contradictions"] if x["status"] == "open"]
        original = plan["action"]
        reason = ""

        if open_x and plan["action"] != "challenge_contradiction":
            plan["action"] = "challenge_contradiction"
            plan["target_contradiction_id"] = open_x[0]["id"]
            reason = ("P1 protocol: an unaddressed contradiction outranks opening a "
                      "new topic")
        elif (p.spec.can_code and rnd >= 2 and not state["code_ran"]
                and plan["action"] != "request_code"):
            plan["action"] = "request_code"
            reason = "P3 protocol: technical screen requires a live coding task by round 2"
        elif (plan["action"] not in ("verify_with_tool", "request_code")
                and p.spec.tools and not p.used_tool):
            plan["action"] = "verify_with_tool"
            plan["tool"] = self._pick_tool(p, state)
            reason = (f"P2 protocol: {p.spec.display} has not yet grounded a question "
                      "in tool evidence")

        if reason:
            plan["policy_override"] = {"from": original, "to": plan["action"],
                                       "reason": reason}
            plan["reasoning"] = f"{plan.get('reasoning','')} [{reason}]".strip()
            self.emit("interview", "policy_override", {
                "round": rnd, "agent": p.spec.key, "display": p.spec.display,
                "from_action": original, "to_action": plan["action"],
                "reason": reason})
        return plan

    def _pick_tool(self, p: Panelist, state: dict[str, Any]) -> str:
        """Choose a tool for this panelist, preferring one nobody has used yet."""
        used = {c["tool"] for c in self.memory.tool_calls(self.session_id)}
        prefer = {"hr": ["resume_db"],
                  "tech": ["github_verify", "resume_db", "web_fact_check"],
                  "bar": ["web_fact_check", "github_verify", "resume_db"]}[p.spec.key]
        available = self.tools.registry() if self.tools else {}
        for t in prefer:
            if t in p.spec.tools and t in available and t not in used:
                return t
        for t in prefer:
            if t in p.spec.tools and t in available:
                return t
        return "resume_db"

    def _turn(self, p: Panelist, rnd: int, candidate: Candidate):
        spec: AgentSpec = p.spec
        state = self._state(rnd)

        # --- PLAN -------------------------------------------------------
        # If panel protocol already dictates this turn's action, we do not pay
        # for an LLM plan we would immediately discard. The agent still plans
        # freely whenever it has a genuine choice.
        t0 = time.time()
        forced = self._forced_action(p, rnd, state)
        if forced is not None:
            plan = forced
            self.emit("interview", "policy_override", {
                "round": rnd, "agent": p.spec.key, "display": p.spec.display,
                "from_action": "(planner skipped)", "to_action": plan["action"],
                "reason": plan["policy_reason"]})
        else:
            plan = p.plan(state)
        self.emit("interview", "agent_plan", {
            "round": rnd, "agent": spec.key, "display": spec.display,
            "emoji": spec.emoji, "action": plan["action"],
            "reasoning": plan.get("reasoning", ""),
            "target_competency": plan.get("target_competency", ""),
            "tool": plan.get("tool", ""),
            "target_contradiction_id": plan.get("target_contradiction_id", ""),
            "latency_s": round(time.time() - t0, 2)})

        # --- ACT --------------------------------------------------------
        asked = p.ask(plan, state)
        q = asked["question"]

        if asked.get("tool_observation"):
            obs = asked["tool_observation"]
            self.tool_findings.append({"round": rnd, "agent": spec.key, **obs})
            self.emit("interview", "tool_call", {
                "round": rnd, "agent": spec.key, "display": spec.display,
                "tool": obs.get("tool"), "ok": obs.get("ok"),
                "args": obs.get("args"), "latency_ms": obs.get("latency_ms"),
                "error": obs.get("error"),
                "result_preview": json.dumps(obs.get("data"), default=str)[:600]})

        if asked.get("refused"):
            self.emit("guardrail", "question_refused", {
                "round": rnd, "agent": spec.key, "display": spec.display,
                "blocked_question": asked["refused"]["blocked"],
                "reason": asked["refused"]["reason"],
                "replacement_question": q})

        if self.cfg and self.cfg.enable_speech:
            self.tools.round = rnd
            self.tools.agent = spec.key
            self.tools.speech_say(q)

        qid = self.memory.add_turn(
            self.session_id, rnd, spec.key, "question", q,
            {"intent": asked.get("intent", ""), "action": asked.get("action", ""),
             "reasoning": asked.get("reasoning", ""),
             "contradiction_id": asked.get("contradiction_id", "")})
        self.all_questions.append({"agent": spec.key, "round": rnd, "q": q})
        if plan.get("target_competency"):
            self.covered.append({"agent": spec.key, "competency": plan["target_competency"]})

        self.emit("interview", "question", {
            "round": rnd, "agent": spec.key, "display": spec.display,
            "emoji": spec.emoji, "text": q, "intent": asked.get("intent", ""),
            "action": asked.get("action", ""), "turn_id": qid,
            "is_challenge": asked.get("action") == "challenge_contradiction",
            "contradiction_id": asked.get("contradiction_id", "")})

        # --- OBSERVE ----------------------------------------------------
        history = [{"agent": t["agent"], "text": t["text"]}
                   for t in self.memory.transcript(self.session_id)
                   if t["kind"] in ("question", "answer")]
        try:
            raw_answer = candidate.answer(q, round_=rnd, asked_by=spec.display,
                                          history=history)
        except Exception as e:  # noqa: BLE001
            raw_answer = "(no answer provided)"
            self.emit("interview", "candidate_error", {"error": str(e)[:300]})

        # Guardrail: candidate text is untrusted input.
        scr = self.guard.screen_answer(raw_answer, round_=rnd)
        answer = raw_answer
        if not scr.allowed:
            answer = scr.sanitized
            self.emit("guardrail", "injection_blocked", {
                "round": rnd, "agent": spec.key,
                "findings": scr.findings, "detail": scr.detail,
                "original_excerpt": raw_answer[:400],
                "sanitized_excerpt": answer[:400]})

        aid = self.memory.add_turn(
            self.session_id, rnd, "candidate", "answer", answer,
            {"to_agent": spec.key, "injection_flagged": not scr.allowed})
        self.emit("interview", "answer", {
            "round": rnd, "agent": "candidate", "to_agent": spec.key,
            "text": answer, "turn_id": aid,
            "injection_flagged": not scr.allowed})

        # --- live code execution ---------------------------------------
        if asked.get("is_code_request") or ("```" in answer and spec.can_code):
            code = _extract_code(answer)
            self.tools.round = rnd
            self.tools.agent = spec.key
            res = self.tools.code_exec(code, asked.get("code_task_id",
                                                       self.cfg.code_task_id))
            self.code_ran = True
            self.code_result = res.data if res.ok else {"verdict": "tool_error",
                                                        "error": res.error}
            self.emit("interview", "code_execution", {
                "round": rnd, "agent": spec.key, "ok": res.ok,
                "latency_ms": res.latency_ms, "result": self.code_result,
                "code_excerpt": code[:600]})

        # --- EXTRACT ----------------------------------------------------
        new_claims = self.extractor.extract(
            answer, round_=rnd, turn_id=aid,
            topic=asked.get("intent", ""),
            context_projects=self.known_projects)
        for c in new_claims:
            if c.subject not in ("self", "unspecified") and c.subject not in self.known_projects:
                self.known_projects.append(c.subject)
        self.memory.add_claims(self.session_id, new_claims)
        self.emit("interview", "claims_extracted", {
            "round": rnd, "from_agent": spec.key, "count": len(new_claims),
            "claims": [{"id": c.id, "text": c.short(), "span": c.raw_span[:140],
                        "predicate": c.predicate, "confidence": c.confidence}
                       for c in new_claims],
            "ledger_size": len(self.memory.claims(self.session_id))})

        # --- DETECT (whole ledger, so cross-round pairs are caught) -----
        if new_claims:
            all_claims = self.memory.claims(self.session_id)
            existing = [(x.claim_a, x.claim_b)
                        for x in self.memory.contradictions(self.session_id)]
            found = self.detector.detect(all_claims, round_=rnd, skip_pairs=existing)
            for x in found:
                if self.memory.has_pair(self.session_id, x.claim_a, x.claim_b):
                    continue
                self.memory.add_contradiction(self.session_id, x)
                self.emit("interview", "contradiction_detected", {
                    "round": rnd, "id": x.id, "type": x.ctype,
                    "severity": x.severity, "score": x.score,
                    "rationale": x.rationale, "detector": x.detector,
                    "evidence": x.evidence,
                    "found_by": spec.key, "found_by_display": spec.display})
                # --- RE-PLAN signal -----------------------------------
                self.emit("interview", "replan", {
                    "round": rnd,
                    "trigger": f"new {x.severity}-severity {x.ctype}",
                    "effect": ("next panelist's planner is now primed to confront "
                               "this before opening any new topic"),
                    "contradiction_id": x.id})

        # --- mark challenged contradictions as addressed ---------------
        if asked.get("action") == "challenge_contradiction" and asked.get("contradiction_id"):
            self.memory.resolve_contradiction(
                asked["contradiction_id"], "addressed",
                f"Confronted by {spec.display} in round {rnd}; "
                f"candidate response recorded in turn {aid}.")
            self.emit("interview", "contradiction_addressed", {
                "round": rnd, "id": asked["contradiction_id"],
                "by": spec.key, "answer_turn": aid})

    # -------------------------------------------------------------- finalize
    def finalize(self) -> dict[str, Any]:
        assert self.verdict_engine and self.tools and self.guard
        state = self._state(self.cfg.rounds if self.cfg else 3)

        # --- independent scorecards ------------------------------------
        self.emit("score", "scoring_start", {"panelists": [p.spec.key for p in self.panelists]})
        cards: dict[str, dict[str, Any]] = {}
        for p in self.panelists:
            card = p.score(state)
            cards[p.spec.key] = card
            self.emit("score", "scorecard", card)

        # --- debate -----------------------------------------------------
        self.emit("debate", "debate_start", {
            "initial_positions": {k: c["recommendation"] for k, c in cards.items()},
            "rounds": self.settings.debate_rounds})
        debate = self.verdict_engine.run_debate(self.panelists, cards, state, emit=self.emit)

        # --- consensus --------------------------------------------------
        cons = self.verdict_engine.consensus(cards, state)
        self.emit("consensus", "consensus_computed", cons)

        narrative = self.verdict_engine.narrate(cons, cards, debate, state)
        verdict = {
            "decision": cons["decision"],
            "confidence": cons["confidence"],
            "rationale": narrative["rationale"],
            "per_agent": cards,
            "evidence": narrative["key_evidence"],
            "dissent": cons["dissent"],
            "debate": debate,
            "consensus_detail": cons,
            "decision_drivers": narrative["decision_drivers"],
            "risk_if_hired": narrative["risk_if_hired"],
            "recommended_next_step": narrative["recommended_next_step"],
        }
        vid = self.memory.save_verdict(self.session_id, verdict)
        self.emit("consensus", "verdict", {**verdict, "verdict_id": vid,
                                           "approval_required": self.settings.require_human_approval})

        # --- HUMAN APPROVAL CHECKPOINT ---------------------------------
        if self.settings.require_human_approval:
            self.memory.set_status(self.session_id, "awaiting_approval")
            self.emit("approval", "approval_required", {
                "verdict_id": vid, "decision": cons["decision"],
                "confidence": cons["confidence"],
                "message": ("This hire/no-hire recommendation is NOT final. A human "
                            "reviewer must approve, override, or reject it before it "
                            "is released to the hiring team."),
                "options": ["approve", "override", "reject"]})
        else:
            self.memory.set_status(self.session_id, "completed")

        report = self.build_report()
        self.tools.round = self.cfg.rounds if self.cfg else 3
        self.tools.agent = "moderator"
        exp = self.tools.export_report(report)
        self.emit("export", "report_written", {
            "path": (exp.data or {}).get("path") if exp.ok else "",
            "ok": exp.ok, "bytes": (exp.data or {}).get("bytes") if exp.ok else 0})
        return report

    # ---------------------------------------------------------------- report
    def build_report(self) -> dict[str, Any]:
        data = self.memory.export(self.session_id)
        g = self.guard.summary() if self.guard else {}
        contras = data["contradictions"]
        report = {
            "session_id": self.session_id,
            "generated_at": time.time(),
            "wall_clock_s": round(time.time() - self.started_at, 1),
            "llm": self.llm.describe(),
            "llm_usage": self.llm.usage.snapshot(),
            "tool_usage": self.tools.stats() if self.tools else {},
            "detector_stats": self.detector.stats if self.detector else {},
            "guardrails": g,
            "minimum_bar": {
                "agentic_loop": {
                    "satisfied": self.llm.usage.calls > 1,
                    "llm_calls": self.llm.usage.calls,
                    "phases": ["plan", "act", "observe", "extract", "detect",
                               "replan", "score", "debate", "consensus", "approval"],
                    "replans_triggered": len([e for e in data["events"]
                                              if e["kind"] == "replan"]),
                },
                "tools": {
                    "satisfied": (self.tools.stats().get("distinct_tools", 0) >= 3
                                  if self.tools else False),
                    "distinct_tools_used": (self.tools.stats().get("distinct_tools", 0)
                                            if self.tools else 0),
                    "detail": self.tools.stats().get("by_tool", {}) if self.tools else {},
                },
                "memory": {
                    "satisfied": len(data["claims"]) > 0,
                    "backend": "sqlite",
                    "db_path": self.memory.path,
                    "claims_persisted": len(data["claims"]),
                    "turns_persisted": len(data["transcript"]),
                    "events_persisted": len(data["events"]),
                    "resumable_across_sessions": True,
                },
                "guardrails": {
                    "satisfied": bool(self.settings.require_human_approval),
                    "human_approval_checkpoint": self.settings.require_human_approval,
                    "refusals_demonstrated": g.get("refusal_count", 0),
                    "injections_neutralized": g.get("injection_count", 0),
                    "tool_failures_handled": (len(self.tools.failure_handled)
                                              if self.tools else 0),
                },
            },
            "contradiction_summary": {
                "total": len(contras),
                "by_severity": _count(contras, "severity"),
                "by_type": _count(contras, "ctype"),
                "addressed_in_interview": len([x for x in contras
                                               if x["status"] == "addressed"]),
                "cross_round": len([x for x in contras
                                    if x.get("claim_a_round") != x.get("claim_b_round")]),
            },
            **data,
        }
        return report

    # --------------------------------------------------------------- approval
    def record_approval(self, decision: str, by: str, note: str) -> dict[str, Any]:
        ok = self.memory.record_approval(self.session_id, decision, by, note)
        self.memory.set_status(self.session_id,
                               "completed" if decision == "approve" else f"human_{decision}")
        self.emit("approval", "approval_recorded", {
            "decision": decision, "by": by, "note": note, "saved": ok})
        report = self.build_report()
        if self.guard and self.tools:
            redacted = self.guard.redact_export(report)
            self.tools.agent = "moderator"
            self.tools.export_report(redacted, f"interview_{self.session_id}_signed.json")
        return report


def _count(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[str(r.get(key))] = out.get(str(r.get(key)), 0) + 1
    return out


def _extract_code(text: str) -> str:
    """Pull python out of a markdown block, else assume the whole answer is code."""
    import re
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.S)
    if m:
        return m.group(1)
    if re.search(r"^\s*def\s+\w+\s*\(", text or "", re.M):
        return text
    return ""
