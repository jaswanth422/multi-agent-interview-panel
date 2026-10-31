"""The three panelists plus the moderator.

Each panelist runs its own mini agentic loop every turn:

    PLAN   - look at the claim ledger, open contradictions and what the other
             panelists already covered, then choose an ACTION
    ACT    - execute the action (ask / probe / challenge / run a tool)
    OBSERVE- read the answer, which the orchestrator turns into new claims

Actions available to a panelist:
    ask_new               - open a new competency area
    probe_prior           - drill into a specific earlier answer (must cite it)
    challenge_contradiction - confront an open contradiction
    verify_with_tool      - call resume_db / github_verify / web_fact_check first,
                            then ask a question informed by the result
    request_code          - hand over a live coding task (Technical only)

The agents genuinely differ: different system prompts, different competency
maps, different scoring rubrics, and the Bar-Raiser is the only one allowed to
escalate difficulty and is biased toward finding disconfirming evidence.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

from .claims import Claim
from .guardrails import Guardrails
from .llm import LLM
from .memory import Contradiction, Memory
from .tools import ToolBelt

ACTIONS = ["ask_new", "probe_prior", "challenge_contradiction",
           "verify_with_tool", "request_code"]


@dataclass
class AgentSpec:
    key: str
    display: str
    emoji: str
    mandate: str
    competencies: list[str]
    rubric: list[str]
    tools: list[str]
    can_code: bool = False
    escalates: bool = False
    weight: float = 1.0


HR = AgentSpec(
    key="hr", display="HR Panelist", emoji="🧑‍💼",
    mandate=(
        "You are a senior HR interviewer. You assess collaboration, ownership, "
        "communication, conflict handling and motivation using the STAR method "
        "(Situation, Task, Action, Result). You listen for WHO did WHAT: candidates "
        "often say 'we' to hide that they personally did very little. Push politely "
        "but persistently for first-person specifics. You never ask about protected "
        "attributes (age, family, religion, nationality, health, salary history)."),
    competencies=["ownership & accountability", "collaboration & conflict",
                  "communication clarity", "motivation & role fit",
                  "handling failure and feedback"],
    rubric=["Did they describe their OWN actions, not the team's?",
            "Is the story specific and falsifiable, with a concrete result?",
            "Do they take accountability for failures?",
            "Is their account stable across the interview?"],
    tools=["resume_db"], weight=0.9,
)

TECH = AgentSpec(
    key="tech", display="Technical Panelist", emoji="🛠️",
    mandate=(
        "You are a staff-level engineer running the technical screen. You verify "
        "depth, not vocabulary. Ask for mechanisms, trade-offs and failure modes: "
        "'what happens when', 'why that and not X', 'how did you measure it'. "
        "If an answer is buzzword-shaped, immediately ask the one question someone "
        "who had actually done it could answer. You may set a live coding task."),
    competencies=["system design & trade-offs", "debugging & root-cause depth",
                  "data modelling / correctness", "coding ability",
                  "operational maturity (testing, monitoring, rollout)"],
    rubric=["Can they explain the mechanism, not just the name?",
            "Do they reason about trade-offs and failure modes?",
            "Does their claimed depth survive one follow-up?",
            "Did their code actually pass the tests?"],
    tools=["resume_db", "github_verify", "web_fact_check", "code_exec"],
    can_code=True, weight=1.2,
)

BAR = AgentSpec(
    key="bar", display="Bar-Raiser", emoji="🎯",
    mandate=(
        "You are the Bar-Raiser: an independent senior interviewer whose job is to "
        "protect the hiring bar, NOT to be agreeable. You deliberately seek "
        "DISCONFIRMING evidence for the emerging consensus. You target the weakest "
        "or least-verified claim, escalate difficulty each round, and you are the "
        "panelist who confronts inconsistencies directly but respectfully. You do "
        "not repeat ground the other two already covered. A candidate who only "
        "sounds good must not pass you."),
    competencies=["depth under pressure", "claim consistency & integrity",
                  "scope and impact calibration", "ambiguity handling",
                  "raising the hiring bar"],
    rubric=["Does the strongest claim hold up under the hardest question?",
            "Did they reconcile inconsistencies honestly or deflect?",
            "Is their self-reported scope calibrated to reality?",
            "Would this person raise the average of the team?"],
    tools=["web_fact_check", "github_verify", "resume_db"],
    escalates=True, weight=1.3,
)

PANEL: list[AgentSpec] = [HR, TECH, BAR]
BY_KEY = {a.key: a for a in PANEL}


PLAN_SYSTEM = """You are the planning module of one interview panelist.

Choose the single most valuable next action given the interview state. You are
one of three panelists; do not duplicate what others already covered.

Priority order (strong guidance, use judgement):
1. If an OPEN CONTRADICTION exists and is unresolved -> "challenge_contradiction".
2. If a prior answer was vague, evasive, or used "we" to hide personal ownership
   -> "probe_prior" on that specific answer.
3. If the candidate made a checkable factual/resume claim you haven't verified
   -> "verify_with_tool".
4. If you are the technical panelist, it is round >= 2, and no code has been run
   -> "request_code".
5. Otherwise -> "ask_new" on an uncovered competency.

Return JSON ONLY:
{"action": "<one of ask_new|probe_prior|challenge_contradiction|verify_with_tool|request_code>",
 "target_competency": "<competency or ''>",
 "target_claim_id": "<claim id you are drilling into, or ''>",
 "target_contradiction_id": "<contradiction id, or ''>",
 "tool": "<resume_db|github_verify|web_fact_check or ''>",
 "tool_args": {"...": "..."},
 "reasoning": "<<=30 words on why this action now>"}"""


QUESTION_SYSTEM = """You are {display} ({emoji}) on a 3-person interview panel.

YOUR MANDATE
{mandate}

HARD RULES
- Ask exactly ONE question. No preamble, no multi-part lists.
- 10-45 words. Conversational, as if spoken aloud.
- BUILD ON the interview so far: when drilling into something the candidate said,
  quote their words briefly ("You said X ...") so it is visibly a follow-up.
- Never ask about age, family, marital status, religion, caste, nationality,
  visa, health, disability, or salary history.
- Never repeat a question already asked by any panelist.
- Do not evaluate or comment, just ask.

Return JSON ONLY: {{"question": "...", "intent": "<what you are testing, <=12 words>"}}"""


@dataclass
class Panelist:
    spec: AgentSpec
    llm: LLM
    memory: Memory
    tools: ToolBelt
    guard: Guardrails
    session_id: str
    asked: list[str] = field(default_factory=list)
    notes: list[dict[str, Any]] = field(default_factory=list)
    escalation: int = 0
    used_tool: bool = False

    # ------------------------------------------------------------- PLAN
    def plan(self, state: dict[str, Any]) -> dict[str, Any]:
        open_x = [x for x in state["contradictions"] if x["status"] == "open"]
        user = f"""YOU ARE: {self.spec.display}
YOUR COMPETENCIES: {', '.join(self.spec.competencies)}
TOOLS YOU MAY USE: {', '.join(self.spec.tools) or 'none'}
ROUND: {state['round']} of {state['total_rounds']}
CODE TASK ALREADY RUN: {state['code_ran']}

COMPETENCIES ALREADY COVERED BY THE PANEL:
{json.dumps(state['covered'], indent=1)}

OPEN CONTRADICTIONS ({len(open_x)}):
{json.dumps(open_x, indent=1)[:1500] if open_x else 'none'}

UNVERIFIED CHECKABLE CLAIMS:
{json.dumps(state['unverified'], indent=1)[:1200] if state['unverified'] else 'none'}

RECENT TRANSCRIPT (last 6 turns):
{state['recent_transcript'][:2200]}

Choose your next action."""

        fallback = {"action": "challenge_contradiction" if open_x else "ask_new",
                    "target_competency": random.choice(self.spec.competencies),
                    "target_claim_id": "", "tool": "", "tool_args": {},
                    "target_contradiction_id": open_x[0]["id"] if open_x else "",
                    "reasoning": "heuristic fallback (planner JSON unavailable)"}

        d = self.llm.json(PLAN_SYSTEM, user, label=f"plan_{self.spec.key}",
                          max_tokens=320, temperature=0.3, fallback=fallback)
        if not isinstance(d, dict):
            d = fallback

        action = str(d.get("action", "")).strip()
        if action not in ACTIONS:
            action = fallback["action"]
        # Enforce capability boundaries the LLM might ignore.
        if action == "request_code" and not self.spec.can_code:
            action = "ask_new"
        if action == "request_code" and state["code_ran"]:
            action = "probe_prior"
        if action == "challenge_contradiction" and not open_x:
            action = "ask_new"
        if action == "verify_with_tool":
            t = str(d.get("tool", "")).strip()
            if t not in self.spec.tools or t == "code_exec":
                t = self.spec.tools[0] if self.spec.tools else ""
            if not t:
                action = "ask_new"
            d["tool"] = t

        d["action"] = action
        d.setdefault("tool_args", {})
        if not isinstance(d.get("tool_args"), dict):
            d["tool_args"] = {}
        d["agent"] = self.spec.key
        return d

    # -------------------------------------------------------------- ACT
    def ask(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        """Produce the question text for a planned action. Returns dict with
        question, intent, tool_observation (optional)."""
        tool_obs: dict[str, Any] | None = None
        context_block = ""

        action = plan["action"]

        if action == "verify_with_tool":
            tool_obs = self._run_tool(plan, state)
            context_block = (
                "\nTOOL EVIDENCE you just gathered (use it to ask a sharper question; "
                f"if it contradicts the candidate, say so):\n{json.dumps(tool_obs, default=str)[:1100]}")

        elif action == "challenge_contradiction":
            x = next((c for c in state["contradictions"]
                      if c["id"] == plan.get("target_contradiction_id")), None)
            if x is None:
                x = next((c for c in state["contradictions"] if c["status"] == "open"), None)
            if x:
                plan["target_contradiction_id"] = x["id"]
                ev = x.get("evidence", {})
                a, b = ev.get("claim_a", {}), ev.get("claim_b", {})
                context_block = f"""
THE INCONSISTENCY YOU MUST CONFRONT (this is your question's whole purpose):
  In round {b.get('round')} the candidate said: "{b.get('span','')}"
  In round {a.get('round')} the candidate said: "{a.get('span','')}"
  Why these conflict: {x.get('rationale','')}
  Suggested framing: {ev.get('challenge_question','')}

Ask ONE question that quotes BOTH statements back and asks them to reconcile it.
Be respectful and non-accusatory, but do not let them off the hook."""

        elif action == "probe_prior":
            cid = plan.get("target_claim_id") or ""
            c = next((c for c in state["claims"] if c["id"] == cid), None)
            if c:
                context_block = (f"\nDRILL INTO THIS SPECIFIC CLAIM the candidate made: "
                                 f"\"{c.get('raw_span','')}\" (structured: {c.get('subject')} / "
                                 f"{c.get('predicate')} / {c.get('object')}).\n"
                                 "Quote their words and ask for the detail only someone who "
                                 "really did it would know.")
            else:
                context_block = ("\nThe candidate's last answer was thin. Quote part of it and "
                                 "ask for the specific detail that would prove it.")

        elif action == "request_code":
            task = self.tools.code_task(state["code_task_id"])
            return {"question": (f"Let's write some code. {task['prompt']} "
                                 "Share your implementation when ready."),
                    "intent": "live coding ability",
                    "action": action, "is_code_request": True,
                    "code_task_id": task["task_id"], "tool_observation": None,
                    "reasoning": plan.get("reasoning", "")}

        if self.spec.escalates:
            self.escalation += 1
            context_block += (f"\nESCALATION LEVEL {self.escalation}: make this question harder "
                              "than the panel's previous ones. Target the weakest claim.")

        user = f"""INTERVIEW SO FAR (most recent last):
{state['recent_transcript'][:2600]}

CANDIDATE RESUME SUMMARY:
{json.dumps(state['resume_summary'], indent=1)[:700]}

QUESTIONS ALREADY ASKED BY THE WHOLE PANEL (never repeat these):
{json.dumps(state['all_questions'], indent=1)[:1400]}

YOUR PLANNED ACTION: {action}
TARGET COMPETENCY: {plan.get('target_competency') or 'your choice'}
YOUR PLANNING NOTE: {plan.get('reasoning','')}
{context_block}

Ask your single question now."""

        sys_prompt = QUESTION_SYSTEM.format(
            display=self.spec.display, emoji=self.spec.emoji, mandate=self.spec.mandate)

        d = self.llm.json(sys_prompt, user, label=f"ask_{self.spec.key}",
                          max_tokens=260, temperature=0.55,
                          fallback={"question": "", "intent": plan.get("target_competency", "")})
        q = ""
        intent = plan.get("target_competency", "")
        if isinstance(d, dict):
            q = str(d.get("question") or "").strip()
            intent = str(d.get("intent") or intent).strip()
        if not q:
            q = self._template_question(action, state)

        # Guardrail: screen our own outbound question.
        screen = self.guard.screen_question(q, agent=self.spec.key, round_=state["round"])
        refused = None
        if not screen.allowed:
            refused = {"blocked": q, "reason": screen.detail}
            q = screen.replacement

        self.asked.append(q)
        return {"question": q, "intent": intent, "action": action,
                "tool_observation": tool_obs, "refused": refused,
                "is_code_request": False,
                "reasoning": plan.get("reasoning", ""),
                "contradiction_id": plan.get("target_contradiction_id", "")}

    def _run_tool(self, plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        tool = plan.get("tool", "")
        args = dict(plan.get("tool_args") or {})
        self.tools.agent = self.spec.key
        self.tools.round = state["round"]
        reg = self.tools.registry()
        if tool not in reg:
            tool = "resume_db"
            args = {"field": "projects"}
        # Repair commonly-malformed args.
        if tool == "resume_db":
            f = str(args.get("field", "all"))
            args = {"field": f if f in ("projects", "skills", "roles", "profile", "all") else "all"}
        elif tool == "github_verify":
            u = args.get("user") or state["resume_summary"].get("github") or ""
            args = {"user": str(u), "repo": str(args.get("repo", "") or "")}
            if not args["user"]:
                tool, args = "resume_db", {"field": "skills"}
        elif tool == "web_fact_check":
            q = args.get("query") or plan.get("target_competency") or "software engineering"
            args = {"query": str(q)[:120]}

        res = reg[tool](**args)
        self.used_tool = True
        return {"tool": res.tool, "ok": res.ok, "latency_ms": res.latency_ms,
                "data": res.data, "error": res.error, "args": args}

    def _template_question(self, action: str, state: dict[str, Any]) -> str:
        comp = random.choice(self.spec.competencies)
        if action == "challenge_contradiction":
            x = next((c for c in state["contradictions"] if c["status"] == "open"), None)
            if x:
                return (x.get("evidence", {}).get("challenge_question")
                        or "Two things you told us don't line up. Can you reconcile them?")
        return {
            "hr": f"Tell me about a specific time you personally drove {comp}. What did YOU do?",
            "tech": f"Walk me through the mechanism behind your approach to {comp}. What breaks first under load?",
            "bar": f"What is the hardest trade-off you have personally made around {comp}, and what did you get wrong?",
        }[self.spec.key]

    # ---------------------------------------------------------- SCORE
    def score(self, state: dict[str, Any]) -> dict[str, Any]:
        """Independent scorecard. Must cite claim ids as evidence."""
        system = f"""You are {self.spec.display} writing your INDEPENDENT scorecard after a
panel interview. You have not seen the other panelists' scores. Be decisive and
evidence-bound: every score must be justified by something the candidate
actually said.

YOUR MANDATE: {self.spec.mandate}

YOUR RUBRIC:
{chr(10).join('- ' + r for r in self.spec.rubric)}

Scoring scale (be strict, 3 is the bar):
 1 = strong no   2 = no   3 = meets the bar   4 = above bar   5 = exceptional

Return JSON ONLY:
{{"scores": {{"<competency>": <1-5>, ...}},
  "overall": <1-5 float>,
  "recommendation": "strong_hire"|"hire"|"no_hire"|"strong_no_hire",
  "confidence": <0-1>,
  "strengths": ["<evidence-backed, <=20 words>", ...],
  "concerns": ["<evidence-backed, <=20 words>", ...],
  "evidence_claim_ids": ["<claim id>", ...],
  "integrity_note": "<how the candidate handled inconsistencies, <=30 words>",
  "summary": "<<=60 words>"}}"""

        user = f"""ROLE: {state['role']} ({state['seniority']})
COMPETENCIES YOU MUST SCORE: {', '.join(self.spec.competencies)}

FULL TRANSCRIPT:
{state['full_transcript'][:6000]}

CONTRADICTIONS THE PANEL DETECTED:
{json.dumps(state['contradictions_brief'], indent=1)[:1600] if state['contradictions_brief'] else 'none detected'}

CODE EXECUTION RESULT:
{json.dumps(state['code_result'], default=str)[:700] if state['code_result'] else 'no coding task run'}

TOOL-VERIFIED FACTS:
{json.dumps(state['tool_findings'], default=str)[:1200] if state['tool_findings'] else 'none'}

CLAIM LEDGER (cite these ids as evidence):
{json.dumps(state['claims_brief'], indent=1)[:2200]}

INTEGRITY FLAGS: {json.dumps(state['integrity_flags'])[:400]}

Write your scorecard."""

        fb = {"scores": {c: 3 for c in self.spec.competencies}, "overall": 3.0,
              "recommendation": "no_hire", "confidence": 0.3,
              "strengths": [], "concerns": ["scorecard generation failed; defaulted conservatively"],
              "evidence_claim_ids": [], "integrity_note": "",
              "summary": "Scorecard unavailable; conservative default applied."}

        d = self.llm.json(system, user, label=f"score_{self.spec.key}",
                          max_tokens=900, temperature=0.2, fallback=fb)
        if not isinstance(d, dict):
            d = fb
        return self._clean_score(d, fb)

    def _clean_score(self, d: dict[str, Any], fb: dict[str, Any]) -> dict[str, Any]:
        scores = d.get("scores") if isinstance(d.get("scores"), dict) else {}
        clean_scores: dict[str, float] = {}
        for k, v in scores.items():
            try:
                clean_scores[str(k)] = max(1.0, min(5.0, float(v)))
            except (TypeError, ValueError):
                continue
        if not clean_scores:
            clean_scores = {c: 3.0 for c in self.spec.competencies}
        try:
            overall = float(d.get("overall"))
        except (TypeError, ValueError):
            overall = sum(clean_scores.values()) / len(clean_scores)
        overall = max(1.0, min(5.0, overall))
        rec = str(d.get("recommendation", "")).lower().replace("-", "_").replace(" ", "_")
        if rec not in ("strong_hire", "hire", "no_hire", "strong_no_hire"):
            rec = ("strong_hire" if overall >= 4.5 else "hire" if overall >= 3.2
                   else "no_hire" if overall >= 2.0 else "strong_no_hire")
        try:
            conf = max(0.0, min(1.0, float(d.get("confidence", 0.6))))
        except (TypeError, ValueError):
            conf = 0.6

        def strlist(key: str) -> list[str]:
            v = d.get(key)
            return [str(x)[:200] for x in v][:5] if isinstance(v, list) else []

        return {"agent": self.spec.key, "display": self.spec.display,
                "emoji": self.spec.emoji, "weight": self.spec.weight,
                "scores": clean_scores, "overall": round(overall, 2),
                "recommendation": rec, "confidence": round(conf, 2),
                "strengths": strlist("strengths"), "concerns": strlist("concerns"),
                "evidence_claim_ids": strlist("evidence_claim_ids"),
                "integrity_note": str(d.get("integrity_note", ""))[:300],
                "summary": str(d.get("summary", ""))[:600]}
