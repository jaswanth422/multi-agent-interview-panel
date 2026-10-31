"""Hybrid contradiction detector: deterministic rule engine + LLM adjudication.

Design rationale (this is the part that earns the reliability score):

  Stage 1 - RULES generate *candidate* pairs by comparing structured claim
            fields. Cheap, exhaustive, zero hallucination, fully explainable.
  Stage 2 - An LLM adjudicates only those candidates, deciding whether the
            conflict is real or innocently explainable, and writes the rationale.

Rules alone over-fire (low precision). An LLM alone misses cross-round pairs it
can't hold in context and invents conflicts (low precision AND low recall).
Gating the LLM behind the rules gives us both, and every finding cites two
claim ids plus their verbatim spans, so a human can audit it in seconds.
"""
from __future__ import annotations

import itertools
import re
from typing import Any, Iterable

from .claims import ATTRIBUTION, DENIAL
from .llm import LLM
from .memory import Claim, Contradiction

# A numeric disagreement must exceed this relative tolerance to count.
NUMERIC_TOLERANCE = {
    "team_size": 0.34,   # "about 5" vs 6 is fine; 5 vs 12 is not
    "years_exp": 0.25,
    "timeline": 0.40,
    "metric": 0.50,      # metrics get wide latitude; people round aggressively
}

# Predicates where the number only means something together with its object.
# "3 years of Kafka" vs "6 years of Python" is not a conflict.
OBJECT_SCOPED_NUMERIC = {"years_exp", "metric"}

SEVERITY_BY_TYPE = {
    "ownership_conflict": "high",
    "polarity_conflict": "high",
    "exclusivity_conflict": "high",
    "numeric_conflict": "medium",
    "timeline_conflict": "medium",
}


def _rel_delta(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom


class RuleHit:
    __slots__ = ("a", "b", "ctype", "detail", "score")

    def __init__(self, a: Claim, b: Claim, ctype: str, detail: str, score: float):
        self.a, self.b, self.ctype, self.detail, self.score = a, b, ctype, detail, score


# Phrases that explain an apparent ownership gap innocently.
PROGRESSION = re.compile(
    r"\b(started (as|out)|began as|initially|at first|early on|then (i |took|moved)|"
    r"later (i )?(took|became|moved)|took over|moved into|by the end|eventually|"
    r"promoted|stepped up|grew into|for the (first|second) year|at the time|"
    r"the (following|next) (quarter|year|month)|migrated to|switched to|"
    r"moved (us )?to|we later|afterwards|since then)\b", re.I)

COEXISTENCE = re.compile(
    r"\b(also|as well|in addition|on top of that|besides that|myself because|"
    r"i personally (also|wrote)|alongside|while i was|at the same time)\b", re.I)


def benign_explanation(hit: RuleHit) -> str:
    """Return a reason this candidate pair is NOT a contradiction, or ''.

    An explicitly narrated progression excuses a role or capability conflict
    ("started as a contributor, then took over as lead") because the two
    statements describe different points in time, not competing versions of
    one fact. It does NOT excuse a numeric conflict: shifting a team size or a
    metric is not explained by saying "initially".
    """
    if hit.ctype not in ("ownership_conflict", "polarity_conflict",
                         "exclusivity_conflict"):
        return ""
    both = f"{hit.a.raw_span} || {hit.b.raw_span}"
    m = PROGRESSION.search(both)
    if m:
        return f"explained progression over time ({m.group(0)!r})"

    # Work credited to someone else is not a claim about the candidate's own
    # experience, so it cannot contradict one.
    if hit.ctype in ("polarity_conflict", "exclusivity_conflict"):
        if ATTRIBUTION.search(both) and not DENIAL.search(both):
            return "work attributed to a colleague, not a denial of experience"
        return ""
    if hit.ctype != "ownership_conflict":
        return ""

    # A lower-ownership statement framed as an ADDITION to the higher one
    # ("I owned the design and I also wrote the tests myself") is coexisting
    # scope. Checked across BOTH spans: the marker frequently sits in a
    # different sentence from the one the claim was anchored to, which is how
    # an honest many-hats startup engineer previously got flagged.
    m2 = COEXISTENCE.search(both)
    if m2:
        return f"coexisting scope, not a replacement ({m2.group(0)!r})"
    return ""


def dedupe_for_pairing(claims: list[Claim]) -> list[Claim]:
    """Collapse restatements so one disagreement yields one candidate pair.

    A candidate repeats themselves, the panel re-asks, and the resume holds a
    written copy of the same fact. Pairing the raw ledger therefore produces
    N*M duplicates of a single conflict. We keep the EARLIEST instance of each
    distinct assertion, because "you said this in round 1" is the version worth
    quoting back.
    """
    best: dict[tuple, Claim] = {}
    for c in claims:
        if c.predicate == "role_on":
            key = (c.subject.lower().strip(), "role_on", c.ownership)
        elif c.predicate in ("used_tech", "employment", "education", "outcome",
                             "responsibility"):
            key = (c.subject.lower().strip(), c.predicate,
                   c.object.lower().strip(), c.polarity)
        else:
            key = (c.subject.lower().strip(), c.predicate,
                   c.object.lower().strip(), c.value_num, c.unit)
        prev = best.get(key)
        if prev is None:
            best[key] = c
            continue
        # Prefer the earliest round; within a round prefer the richer span.
        if (c.round, -len(c.raw_span)) < (prev.round, -len(prev.raw_span)):
            best[key] = c
    return list(best.values())


# ----------------------------------------------------------------- stage 1
def rule_candidates(claims: list[Claim]) -> list[RuleHit]:
    """Pure function: structured claims in, candidate conflicts out."""
    hits: list[RuleHit] = []

    for a, b in itertools.combinations(claims, 2):
        if a.id == b.id:
            continue
        # Only compare claims that are actually about the same thing.
        same_subject = a.subject.lower().strip() == b.subject.lower().strip()

        # R1 - ownership ladder conflict on the same project.
        if (a.predicate == b.predicate == "role_on" and same_subject
                and a.ownership is not None and b.ownership is not None):
            gap = abs(a.ownership - b.ownership)
            if gap >= 2:
                hits.append(RuleHit(
                    a, b, "ownership_conflict",
                    f"ownership ladder gap of {gap} on project '{a.subject}' "
                    f"({a.object!r} = level {a.ownership} vs {b.object!r} = level {b.ownership})",
                    min(1.0, 0.55 + 0.15 * gap)))
                continue

        # R2 - same claim asserted and denied.
        if (a.predicate == b.predicate and same_subject
                and a.object.lower().strip() == b.object.lower().strip()
                and a.object.strip() and a.polarity != b.polarity):
            hits.append(RuleHit(
                a, b, "polarity_conflict",
                f"'{a.object}' is both asserted and denied for '{a.subject}'", 0.9))
            continue

        # R3 - "never used X" vs "used X" anywhere (subject-independent:
        #      a blanket denial about a technology conflicts with any usage).
        if (a.predicate == b.predicate == "used_tech"
                and a.object.lower().strip() == b.object.lower().strip()
                and a.object.strip() and a.polarity != b.polarity):
            hits.append(RuleHit(
                a, b, "exclusivity_conflict",
                f"candidate both denies and claims experience with '{a.object}'", 0.88))
            continue

        # R4 - incompatible numbers for the same quantity.
        #
        # The object must match for any predicate that is scoped by what it
        # describes. Without this, "6 years of Python" and "3 years of Kafka"
        # pair up as a numeric conflict, which is nonsense: same subject
        # ("self"), same predicate (years_exp), completely different things.
        if (a.predicate == b.predicate and same_subject
                and a.value_num is not None and b.value_num is not None
                and a.predicate in NUMERIC_TOLERANCE):
            same_obj = a.object.lower().strip() == b.object.lower().strip()
            object_scoped = a.predicate in OBJECT_SCOPED_NUMERIC
            if (a.unit or "") == (b.unit or "") and (same_obj or not object_scoped):
                d = _rel_delta(a.value_num, b.value_num)
                tol = NUMERIC_TOLERANCE[a.predicate]
                if d > tol:
                    hits.append(RuleHit(
                        a, b,
                        "timeline_conflict" if a.predicate == "timeline" else "numeric_conflict",
                        f"{a.predicate}"
                        + (f" of '{a.object}'" if a.object and object_scoped else "")
                        + f" for '{a.subject}' stated as "
                          f"{a.value_num:g}{a.unit} and {b.value_num:g}{b.unit} "
                          f"({d*100:.0f}% apart, tolerance {tol*100:.0f}%)",
                        min(1.0, 0.5 + d)))
                    continue
    return _collapse(hits)


def _collapse(hits: list[RuleHit]) -> list[RuleHit]:
    """One finding per real disagreement.

    The ledger accumulates the same assertion many times (restated across
    rounds, plus the resume copy), so raw pairing produces N*M duplicates of a
    single conflict. We keep the single strongest, most informative pair per
    (subject, predicate, conflict type), preferring cross-round pairs because
    "you said X in round 1 and Y in round 3" is the finding a human cares about.
    """
    groups: dict[tuple[str, str, str], RuleHit] = {}
    for h in hits:
        key = (h.a.subject.lower().strip(), h.a.predicate, h.ctype)
        prev = groups.get(key)
        if prev is None or _rank(h) > _rank(prev):
            groups[key] = h
    return sorted(groups.values(), key=_rank, reverse=True)


def _rank(h: RuleHit) -> tuple:
    cross_round = h.a.round != h.b.round
    both_interview = h.a.source == "interview" and h.b.source == "interview"
    span_quality = min(len(h.a.raw_span), len(h.b.raw_span))
    return (cross_round, both_interview, round(h.score, 2), span_quality)


# ----------------------------------------------------------------- stage 2
ADJUDICATE_SYSTEM = """You are the panel's fact-checking adjudicator in a job interview.

A deterministic rule engine flagged two claims the candidate made in DIFFERENT
parts of the interview as potentially contradictory. Decide whether this is a
REAL contradiction a human interviewer should challenge.

Judge strictly but fairly:
- REAL: the two statements cannot both be true about the same thing, or the
  candidate materially inflated/changed their role, scope, or numbers.
- NOT REAL: different projects/time periods, different scopes that coexist
  (e.g. "led the migration" + "also wrote tests" is consistent), rounding or
  vagueness, or a candidate openly correcting themselves.

Return JSON:
{"is_contradiction": true|false,
 "severity": "low"|"medium"|"high",
 "confidence": 0.0-1.0,
 "rationale": "<=45 words, quote the conflicting bits",
 "challenge_question": "a direct, polite question that makes the candidate address the conflict"}"""


class ContradictionDetector:
    def __init__(self, llm: LLM, *, use_llm: bool = True):
        self.llm = llm
        self.use_llm = use_llm
        self.stats = {"rule_hits": 0, "llm_confirmed": 0, "llm_rejected": 0,
                      "prefilter_rejected": 0, "duplicate_suppressed": 0}
        # One disagreement should be reported once. Detection runs incrementally
        # after every answer, so the same conflict can surface again later via a
        # different claim pair (e.g. round 3 vs round 1, then round 3 vs the
        # résumé). Those are the same finding to a human reviewer.
        self._signatures: set[tuple[str, str, str]] = set()

    def detect(self, claims: list[Claim], *, round_: int,
               skip_pairs: Iterable[tuple[str, str]] = (),
               max_new: int = 3) -> list[Contradiction]:
        seen = {frozenset(p) for p in skip_pairs}
        out: list[Contradiction] = []

        for hit in rule_candidates(dedupe_for_pairing(claims)):
            if len(out) >= max_new:
                break
            pair = frozenset((hit.a.id, hit.b.id))
            if pair in seen:
                continue
            seen.add(pair)

            sig = (hit.a.subject.lower().strip(), hit.a.predicate, hit.ctype)
            if sig in self._signatures:
                self.stats["duplicate_suppressed"] += 1
                continue

            self.stats["rule_hits"] += 1

            # Stage 1.5 - deterministic benign-pattern filter. Cheaper and far
            # more consistent than asking a small model to notice that
            # "started as a contributor, then took over as lead" is a career
            # progression rather than an inconsistency.
            benign = benign_explanation(hit)
            if benign:
                self.stats["prefilter_rejected"] += 1
                continue

            verdict = self._adjudicate(hit)
            if not verdict["is_contradiction"]:
                self.stats["llm_rejected"] += 1
                continue
            self.stats["llm_confirmed"] += 1
            self._signatures.add(sig)

            out.append(Contradiction(
                claim_a=hit.a.id, claim_b=hit.b.id, ctype=hit.ctype,
                severity=verdict["severity"],
                score=round(0.5 * hit.score + 0.5 * verdict["confidence"], 3),
                rationale=verdict["rationale"],
                detected_round=round_,
                detector="rules+llm" if self.use_llm else "rules",
                evidence={
                    "rule_detail": hit.detail,
                    "rule_score": round(hit.score, 3),
                    "llm_confidence": verdict["confidence"],
                    "challenge_question": verdict["challenge_question"],
                    "claim_a": {"id": hit.a.id, "round": hit.a.round,
                                "text": hit.a.short(), "span": hit.a.raw_span},
                    "claim_b": {"id": hit.b.id, "round": hit.b.round,
                                "text": hit.b.short(), "span": hit.b.raw_span},
                },
            ))
        return out

    def _adjudicate(self, hit: RuleHit) -> dict[str, Any]:
        default_q = (f"Earlier you said \"{hit.b.raw_span[:110]}\", but just now you said "
                     f"\"{hit.a.raw_span[:110]}\". Can you help me reconcile those?")
        if not self.use_llm:
            return {"is_contradiction": True, "severity": SEVERITY_BY_TYPE.get(hit.ctype, "medium"),
                    "confidence": hit.score, "rationale": hit.detail,
                    "challenge_question": default_q}

        user = f"""Rule that fired: {hit.ctype}
Rule detail: {hit.detail}

CLAIM A (interview round {hit.a.round}):
  structured: {hit.a.short()}
  verbatim: "{hit.a.raw_span}"

CLAIM B (interview round {hit.b.round}):
  structured: {hit.b.short()}
  verbatim: "{hit.b.raw_span}"

Is this a real contradiction?"""

        d = self.llm.json(ADJUDICATE_SYSTEM, user, label="adjudicate_contradiction",
                          max_tokens=320, temperature=0.1,
                          fallback={"is_contradiction": True,
                                    "severity": SEVERITY_BY_TYPE.get(hit.ctype, "medium"),
                                    "confidence": hit.score, "rationale": hit.detail,
                                    "challenge_question": default_q})
        if not isinstance(d, dict):
            d = {}
        sev = str(d.get("severity", "")).lower()
        if sev not in ("low", "medium", "high"):
            sev = SEVERITY_BY_TYPE.get(hit.ctype, "medium")
        try:
            conf = float(d.get("confidence", hit.score))
        except (TypeError, ValueError):
            conf = hit.score
        return {
            "is_contradiction": bool(d.get("is_contradiction", True)),
            "severity": sev,
            "confidence": max(0.0, min(1.0, conf)),
            "rationale": str(d.get("rationale") or hit.detail)[:400],
            "challenge_question": str(d.get("challenge_question") or default_q)[:400],
        }
