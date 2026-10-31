"""Candidate simulator.

Two reasons this exists:

1. The live demo must be autonomous and repeatable on stage. A human typing
   answers is slow and can't reproduce a planted contradiction on cue.
2. The evaluation set needs GROUND TRUTH. Each eval case plants specific
   contradictions, so we can measure the detector's precision and recall
   instead of asserting "it looked right".

`ScriptedCandidate` answers from a topic-matched bank (deterministic, carries
ground truth). Unmatched questions fall through to an LLM persona so the
candidate can respond to anything the panel improvises - this keeps the
interview genuinely open-ended while the planted facts stay stable.

A real human candidate uses `HumanCandidate` via the web UI instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .llm import LLM


class Candidate(Protocol):
    def answer(self, question: str, *, round_: int, asked_by: str,
               history: list[dict[str, Any]]) -> str: ...


@dataclass
class ScriptedAnswer:
    """One scripted answer plus the keywords that trigger it.

    `plant_round` makes the answer deterministic: if set, the candidate WILL
    say this in that round regardless of what was asked (first unused slot).
    That is what guarantees the planted ground-truth contradiction actually
    enters the transcript, so precision/recall is measured against a known
    answer key rather than against whatever the model happened to improvise.
    """
    triggers: list[str]
    text: str
    tag: str = ""
    priority: int = 0
    plant_round: int = 0


@dataclass
class ScriptedCandidate:
    name: str
    persona: str
    resume: dict[str, Any]
    bank: list[ScriptedAnswer]
    llm: LLM | None = None
    code_answer: str = ""
    used: set[str] = field(default_factory=set)
    fallback_style: str = ""

    def _score(self, ans: ScriptedAnswer, q: str) -> float:
        ql = q.lower()
        hits = sum(1 for t in ans.triggers if t.lower() in ql)
        if not hits:
            return 0.0
        score = hits / max(len(ans.triggers), 1) + 0.35 * hits + ans.priority
        if ans.tag in self.used:
            score -= 1.6          # strongly prefer unused answers
        return score

    def answer(self, question: str, *, round_: int = 0, asked_by: str = "",
               history: list[dict[str, Any]] | None = None) -> str:
        q = question or ""

        # Planted answers fire on schedule so ground truth is deterministic.
        # A challenge question is exempt: when the panel confronts the candidate
        # we want the persona's real reaction, not a scripted fact drop.
        is_challenge = bool(re.search(
            r"\b(reconcile|earlier you said|you just said|inconsisten|contradic|"
            r"squar(e|ing) (that|those)|clarify the difference|both be true)\b", q, re.I))
        if not is_challenge:
            for a in self.bank:
                tag = a.tag or a.text[:20]
                if a.plant_round and a.plant_round == round_ and tag not in self.used:
                    self.used.add(tag)
                    return a.text

        # Live coding request?
        if self.code_answer and re.search(
                r"\b(write|implement|code|function|def )\b", q, re.I) and "```" not in q:
            if re.search(r"\b(two_sum|dedupe|allowed_count|implementation|write some code)\b", q, re.I):
                self.used.add("code")
                return self.code_answer

        best, best_score = None, 0.0
        for a in self.bank:
            s = self._score(a, q)
            if s > best_score:
                best, best_score = a, s

        if best is not None and best_score > 0.30:
            self.used.add(best.tag or best.text[:20])
            return best.text

        return self._improvise(q, round_, asked_by, history or [])

    def _improvise(self, q: str, round_: int, asked_by: str,
                   history: list[dict[str, Any]]) -> str:
        if self.llm is None:
            return ("I'd have to check my notes on the specifics, but broadly I was "
                    "involved in that area and picked up a lot from it.")
        facts = "\n".join(f"- {a.text}" for a in self.bank[:8])
        hist = "\n".join(f"{h.get('agent','?')}: {h.get('text','')[:160]}"
                         for h in history[-6:])
        system = f"""You are roleplaying a job candidate in an interview. Stay in character.

CHARACTER: {self.name}
PERSONA: {self.persona}
ANSWERING STYLE: {self.fallback_style or 'natural, conversational, 40-90 words'}

THINGS YOU HAVE ALREADY TOLD THIS PANEL (stay consistent with these, do not
contradict them unless your persona says you exaggerate):
{facts}

Rules:
- Answer in first person, 40-90 words, conversational as if spoken.
- Never break character. Never mention that you are an AI.
- Do not volunteer brand-new quantified achievements; keep improvised detail soft.
- If challenged on an inconsistency, respond the way your PERSONA would."""
        return self.llm.text(
            system,
            f"Recent conversation:\n{hist}\n\nInterviewer ({asked_by}) asks: \"{q}\"\n\nYour answer:",
            label="candidate_improvise", max_tokens=260, temperature=0.65).strip()


@dataclass
class HumanCandidate:
    """Answers are supplied externally (web UI / CLI prompt)."""
    pending: list[str] = field(default_factory=list)

    def answer(self, question: str, *, round_: int = 0, asked_by: str = "",
               history: list[dict[str, Any]] | None = None) -> str:
        if self.pending:
            return self.pending.pop(0)
        raise RuntimeError("no human answer supplied")
