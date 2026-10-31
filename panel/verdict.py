"""Debate and consensus.

The panel does NOT simply average three scores. It runs a structured debate:

  1. Each panelist posts an independent scorecard (no peeking).
  2. If they disagree, each sees the others' positions and may REVISE, but a
     revision must cite a reason. Positions are recorded every round, so the
     transcript shows opinions actually moving.
  3. Consensus is then computed with an explicit, auditable rule set, including
     a Bar-Raiser veto on integrity failures. If the panel still cannot agree,
     the moderator breaks the tie and the dissent is preserved in the record
     rather than being hidden.

Averaging would let a confident HR score drown out a specific technical
objection. An evidence-citing debate with recorded dissent is what a real
hiring loop does, and it is auditable.
"""
from __future__ import annotations

import json
from typing import Any

from .agents import BY_KEY, Panelist
from .llm import LLM

REC_VALUE = {"strong_no_hire": 1.0, "no_hire": 2.0, "hire": 4.0, "strong_hire": 5.0}
REC_ORDER = ["strong_no_hire", "no_hire", "hire", "strong_hire"]


def _is_hire(rec: str) -> bool:
    return rec in ("hire", "strong_hire")


DEBATE_SYSTEM = """You are {display} in the panel's post-interview debate.

You have just seen the other panelists' scorecards. Your job is to reach an
HONEST decision, not a polite one.

- If another panelist raises a concern your own evidence cannot answer, UPDATE
  your position and say so. Changing your mind on evidence is a strength.
- If you believe they are wrong, hold your position and say exactly what
  evidence they are missing or over-weighting.
- You must engage with the SPECIFIC points made, not restate your summary.
- Detected contradictions are integrity signals and outweigh polish.

Return JSON ONLY:
{{"position": "strong_hire"|"hire"|"no_hire"|"strong_no_hire",
  "overall": <1-5 float>,
  "changed": true|false,
  "change_reason": "<why you moved, or '' if you held>",
  "argument": "<<=55 words aimed at the other panelists' actual points>",
  "responds_to": ["hr"|"tech"|"bar", ...],
  "concession": "<the strongest point against your own view, <=25 words>"}}"""


MODERATOR_SYSTEM = """You are the panel Moderator. You do not score the candidate.
You synthesise the panel's debate into one defensible decision record.

Rules you must follow:
- The decision you are given is COMPUTED by the panel's consensus rules. Do not
  change it. Explain it.
- Cite the specific evidence and claim ids that drove it.
- If any panelist dissented, state the dissent plainly. Never paper over it.
- If contradictions were detected, say how they affected the decision.
- Be concrete and brief. No corporate filler.

Return JSON ONLY:
{"rationale": "<120-170 words explaining the decision, the evidence, how the "
              "contradictions were weighed, and any dissent>",
 "key_evidence": ["<specific evidence point>", ...],
 "decision_drivers": ["<the 2-4 factors that actually decided it>"],
 "risk_if_hired": "<<=30 words>",
 "recommended_next_step": "<<=25 words, concrete>"}"""


class VerdictEngine:
    def __init__(self, llm: LLM, hire_threshold: float = 3.2, debate_rounds: int = 2):
        self.llm = llm
        self.threshold = hire_threshold
        self.debate_rounds = debate_rounds

    # ---------------------------------------------------------------- debate
    def run_debate(self, panelists: list[Panelist], cards: dict[str, dict[str, Any]],
                   state: dict[str, Any], emit=None) -> list[dict[str, Any]]:
        positions = {k: {"position": c["recommendation"], "overall": c["overall"]}
                     for k, c in cards.items()}
        log: list[dict[str, Any]] = []

        for rnd in range(1, self.debate_rounds + 1):
            if self._unanimous(positions):
                log.append({"round": rnd, "note": "panel already unanimous; debate closed early",
                            "positions": dict(positions), "turns": []})
                if emit:
                    emit("debate", "debate_skipped",
                         {"round": rnd, "reason": "unanimous",
                          "positions": {k: v["position"] for k, v in positions.items()}})
                break

            turns = []
            for p in panelists:
                k = p.spec.key
                others = {o: {**positions[o],
                              "concerns": cards[o].get("concerns", []),
                              "strengths": cards[o].get("strengths", []),
                              "summary": cards[o].get("summary", "")}
                          for o in positions if o != k}
                prev = [t for t in log for t in t.get("turns", [])]
                user = f"""YOUR CURRENT POSITION: {positions[k]['position']} (overall {positions[k]['overall']})
YOUR OWN SCORECARD: {json.dumps(cards[k], indent=1)[:1200]}

THE OTHER PANELISTS:
{json.dumps(others, indent=1)[:1800]}

CONTRADICTIONS DETECTED IN THE INTERVIEW:
{json.dumps(state['contradictions_brief'], indent=1)[:1200] if state['contradictions_brief'] else 'none'}

CODE RESULT: {json.dumps(state['code_result'], default=str)[:400] if state['code_result'] else 'none'}
INTEGRITY FLAGS: {json.dumps(state['integrity_flags'])[:300]}

EARLIER DEBATE TURNS:
{json.dumps(prev, indent=1)[:1500] if prev else 'this is the first debate round'}

Debate round {rnd}. Respond."""

                fb = {"position": positions[k]["position"], "overall": positions[k]["overall"],
                      "changed": False, "change_reason": "", "responds_to": [],
                      "argument": "(no argument generated)", "concession": ""}
                d = self.llm.json(
                    DEBATE_SYSTEM.format(display=p.spec.display), user,
                    label=f"debate_{k}", max_tokens=420, temperature=0.4, fallback=fb)
                if not isinstance(d, dict):
                    d = fb

                pos = str(d.get("position", "")).lower().replace("-", "_")
                if pos not in REC_VALUE:
                    pos = positions[k]["position"]
                try:
                    ov = max(1.0, min(5.0, float(d.get("overall", positions[k]["overall"]))))
                except (TypeError, ValueError):
                    ov = positions[k]["overall"]

                changed = pos != positions[k]["position"]
                turn = {"round": rnd, "agent": k, "display": p.spec.display,
                        "emoji": p.spec.emoji,
                        "from": positions[k]["position"], "position": pos,
                        "overall": round(ov, 2), "changed": changed,
                        "change_reason": str(d.get("change_reason", ""))[:300],
                        "argument": str(d.get("argument", ""))[:500],
                        "concession": str(d.get("concession", ""))[:300],
                        "responds_to": [str(x) for x in d.get("responds_to", [])
                                        if str(x) in BY_KEY][:3]}
                turns.append(turn)
                positions[k] = {"position": pos, "overall": round(ov, 2)}
                cards[k]["recommendation"] = pos
                cards[k]["overall"] = round(ov, 2)
                if emit:
                    emit("debate", "debate_turn", turn)

            log.append({"round": rnd, "positions": dict(positions), "turns": turns})
        return log

    @staticmethod
    def _unanimous(positions: dict[str, dict[str, Any]]) -> bool:
        return len({_is_hire(v["position"]) for v in positions.values()}) == 1

    # ------------------------------------------------------------- consensus
    def consensus(self, cards: dict[str, dict[str, Any]], state: dict[str, Any]) -> dict[str, Any]:
        """Explicit, auditable decision rules. No hidden averaging."""
        rules_fired: list[str] = []

        votes = {k: _is_hire(c["recommendation"]) for k, c in cards.items()}
        hire_votes = sum(votes.values())
        total_w = sum(c["weight"] for c in cards.values()) or 1.0
        weighted = sum(c["overall"] * c["weight"] for c in cards.values()) / total_w

        high_sev = [x for x in state["contradictions_brief"]
                    if x.get("severity") == "high" and x.get("status") != "resolved"]
        unresolved = [x for x in state["contradictions_brief"] if x.get("status") == "open"]
        integrity = state.get("integrity_flags", {})

        decision: str | None = None
        code = state.get("code_result") or {}
        code_verdict = str(code.get("verdict", "")) if code else ""

        # Rule ordering rationale, measured not assumed:
        #
        # A small model's *scores* are weakly discriminative — in our eval a
        # candidate with three proven contradictions scored 3.05 while a
        # flawless one scored 3.28. Deciding from scores alone therefore fails
        # in both directions. What IS reliable is the evidence record: proven
        # inconsistencies, injection attempts and code results are objective.
        #
        # So integrity gates decide first and are disqualifying at screening
        # stage; panel scores only arbitrate candidates whose record is clean.

        # G1 - attempted manipulation of the evaluators.
        if integrity.get("injection_count", 0) > 0:
            decision = "no_hire"
            rules_fired.append(
                f"G1 Integrity gate: {integrity['injection_count']} attempt(s) to manipulate "
                "the evaluators were detected in the candidate's own answers. Disqualifying "
                "regardless of panel scores.")

        # G2 - unresolved high-severity contradiction.
        if decision is None and high_sev:
            decision = "no_hire"
            types = ", ".join(sorted({x.get("type", "?") for x in high_sev}))
            bar = cards.get("bar")
            bar_note = ""
            if bar and not _is_hire(bar["recommendation"]):
                bar_note = (f" The Bar-Raiser independently reached "
                            f"{bar['recommendation']} ({bar['overall']}/5).")
            rules_fired.append(
                f"G2 Integrity gate: {len(high_sev)} unresolved high-severity "
                f"contradiction(s) ({types}) remained unreconciled after being put to the "
                f"candidate. A screen cannot pass an unexplained inconsistency in the "
                f"candidate's own account.{bar_note}")

        # G3 - the live coding task was refused or failed.
        if decision is None and code_verdict in ("refused_unsafe_code", "fail"):
            decision = "no_hire"
            detail = ("submitted code was refused by the sandbox as unsafe"
                      if code_verdict == "refused_unsafe_code"
                      else "submitted code failed every hidden test")
            rules_fired.append(f"G3 Integrity gate: {detail}.")

        # S1 - clean record: the panel's scores decide.
        if decision is None and hire_votes >= 2 and weighted >= self.threshold:
            decision = "hire"
            rules_fired.append(
                f"S1 Clean record + bar cleared: no unresolved contradictions, "
                f"{hire_votes}/3 panelists at or above the bar, weighted score "
                f"{weighted:.2f} >= {self.threshold} (rubric: 3 = meets the bar).")

        # S2 - clean record but the panel is not convinced.
        if decision is None:
            decision = "no_hire"
            why = []
            if hire_votes < 2:
                why.append(f"only {hire_votes}/3 panelists placed the candidate at the bar")
            if weighted < self.threshold:
                why.append(f"weighted score {weighted:.2f} is below the bar {self.threshold}")
            if unresolved:
                why.append(f"{len(unresolved)} contradiction(s) still open")
            rules_fired.append(
                "S2 Below bar: " + "; ".join(why) +
                ". No integrity failure was found, so this is a depth/seniority "
                "judgement rather than a trust problem.")

        dissent = [{"agent": k, "display": cards[k]["display"],
                    "position": cards[k]["recommendation"],
                    "overall": cards[k]["overall"],
                    "their_concerns": cards[k].get("concerns", []),
                    "their_strengths": cards[k].get("strengths", [])}
                   for k, v in votes.items() if v != _is_hire(decision)]

        spread = (max(c["overall"] for c in cards.values())
                  - min(c["overall"] for c in cards.values()))
        agreement = 1.0 - min(spread / 4.0, 1.0)
        mean_conf = sum(c["confidence"] for c in cards.values()) / max(len(cards), 1)
        confidence = round(max(0.05, min(0.99, 0.55 * agreement + 0.45 * mean_conf
                                         - 0.1 * len(dissent))), 2)

        return {"decision": decision, "hire_votes": hire_votes,
                "weighted_score": round(weighted, 2), "threshold": self.threshold,
                "rules_fired": rules_fired, "dissent": dissent,
                "unanimous": len(dissent) == 0, "score_spread": round(spread, 2),
                "panel_agreement": round(agreement, 2), "confidence": confidence,
                "votes": {k: cards[k]["recommendation"] for k in cards}}

    # ------------------------------------------------------------- narrative
    def narrate(self, cons: dict[str, Any], cards: dict[str, dict[str, Any]],
                debate: list[dict[str, Any]], state: dict[str, Any]) -> dict[str, Any]:
        user = f"""COMPUTED DECISION (do not change): {cons['decision'].upper()}
Confidence: {cons['confidence']}
Hire votes: {cons['hire_votes']}/3   Weighted score: {cons['weighted_score']} (bar {cons['threshold']})

CONSENSUS RULES THAT FIRED:
{chr(10).join('- ' + r for r in cons['rules_fired'])}

PANEL SCORECARDS:
{json.dumps(cards, indent=1)[:2600]}

DEBATE (watch for changed positions):
{json.dumps([t for d in debate for t in d.get('turns', [])], indent=1)[:2200]}

DISSENT: {json.dumps(cons['dissent'], indent=1)[:900] if cons['dissent'] else 'none, panel was unanimous'}

CONTRADICTIONS DETECTED:
{json.dumps(state['contradictions_brief'], indent=1)[:1500] if state['contradictions_brief'] else 'none'}

CODE RESULT: {json.dumps(state['code_result'], default=str)[:500] if state['code_result'] else 'none'}
GUARDRAIL EVENTS: {json.dumps(state['integrity_flags'])[:400]}

Write the decision record."""

        fb = {"rationale": ("Decision computed from panel consensus rules: "
                            + "; ".join(cons["rules_fired"])),
              "key_evidence": [], "decision_drivers": cons["rules_fired"][:3],
              "risk_if_hired": "", "recommended_next_step": ""}
        d = self.llm.json(MODERATOR_SYSTEM, user, label="moderator_narrate",
                          max_tokens=760, temperature=0.3, fallback=fb)
        if not isinstance(d, dict):
            d = fb

        def strlist(key: str, limit: int = 6) -> list[str]:
            v = d.get(key)
            return [str(x)[:240] for x in v][:limit] if isinstance(v, list) else []

        return {"rationale": str(d.get("rationale") or fb["rationale"])[:1800],
                "key_evidence": strlist("key_evidence"),
                "decision_drivers": strlist("decision_drivers", 4) or cons["rules_fired"][:3],
                "risk_if_hired": str(d.get("risk_if_hired", ""))[:300],
                "recommended_next_step": str(d.get("recommended_next_step", ""))[:300]}
