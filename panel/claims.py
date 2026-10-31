"""Turn free-text answers into atomic, comparable claims.

Contradiction detection on raw prose is unreliable. So we first normalize each
answer into typed claims with a controlled predicate vocabulary and an
*ownership ladder*, then compare structured values. The LLM does extraction
(what it's good at); comparison is deterministic (what it's bad at).
"""
from __future__ import annotations

import re
from typing import Any

from .llm import LLM
from .memory import Claim

# Controlled predicate vocabulary -----------------------------------------
PREDICATES = [
    "role_on",      # subject = project, object = role phrase, ownership set
    "team_size",    # value_num
    "years_exp",    # value_num, unit=years
    "used_tech",    # object = technology, polarity may be -1
    "metric",       # object = metric name, value_num + unit
    "outcome",      # object = result statement
    "responsibility",
    "timeline",     # value_num + unit (months)
    "employment",   # object = company
    "education",
]

# Ownership ladder: higher = stronger claim of personal ownership.
OWNERSHIP_LADDER: list[tuple[int, tuple[str, ...]]] = [
    (4, ("led", "lead", "owned", "own", "headed", "drove", "drove the", "spearheaded",
         "managed", "was responsible for the whole", "architected end-to-end",
         "ran the project", "project lead", "tech lead")),
    (3, ("designed", "architected", "built", "created", "developed", "implemented",
         "authored", "wrote the", "rewrote", "migrated")),
    (2, ("contributed", "helped", "worked on", "part of", "collaborated",
         "assisted", "supported", "pair", "involved in")),
    (1, ("tested", "qa", "wrote tests", "validated", "verified", "reviewed",
         "triaged", "documented", "monitored")),
    (0, ("observed", "shadowed", "watched", "sat in", "was told", "learned about",
         "no involvement", "not involved")),
]

NEGATION = re.compile(
    r"\b(never|not|no|didn'?t|did not|haven'?t|have not|wasn'?t|was not|"
    r"don'?t|do not|none|nor|without)\b", re.I)

# Spelled-out numbers appear constantly in speech ("a team of four", "six months").
WORD_NUM = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "hundred": 100, "couple": 2, "few": 3, "several": 4,
    "dozen": 12, "half": 0.5,
}
_WORD_NUM_RE = re.compile(
    r"\b(" + "|".join(sorted(WORD_NUM, key=len, reverse=True)) + r")\s+"
    r"(%|percent|ms|seconds?|secs?|rps|qps|people|engineers?|members?|devs?|"
    r"developers?|years?|yrs?|months?|mos?|weeks?)\b", re.I)

# "led the billing migration", "owned our payments pipeline"
_OWNER_PHRASE = re.compile(
    r"\b(" + "|".join(
        sorted((w for _, words in OWNERSHIP_LADDER for w in words), key=len, reverse=True)
    ).replace(" ", r"\s+") + r")\b"
    r"(?:\s+(?:on|for|of|in|at))?\s*"
    r"(?:the|our|my|a|an|this|that)?\s*"
    r"([a-z][a-z0-9]*(?:\s+[a-z][a-z0-9]*){0,3})", re.I)

PROJECT_NOUNS = (
    "migration", "service", "platform", "pipeline", "system", "project",
    "rewrite", "refactor", "integration", "api", "gateway", "dashboard",
    "revamp", "app", "application", "module", "cluster", "infrastructure",
    "infra", "checkout", "onboarding", "search", "billing", "payments",
    "ingestion", "etl", "warehouse", "recommender", "feed", "microservice",
    "mesh", "rollout", "gateway", "ledger", "gatewayv2", "gateway2",
)
_PROJECTISH = re.compile(
    r"\b(?:the|our|my|a|an)\s+([a-z][a-z0-9-]*(?:\s+[a-z][a-z0-9-]*){0,2}\s+"
    r"(?:" + "|".join(PROJECT_NOUNS) + r"))\b", re.I)
_PROJECT_NOUN_RE = re.compile(r"\b(?:" + "|".join(PROJECT_NOUNS) + r")\b", re.I)

_TEAM_SIZE_RE = re.compile(
    r"(?:team of|crew of|squad of|group of|with|were|was)\s+"
    r"(?:about|around|roughly|some|like)?\s*"
    r"(?P<n>\d{1,3}|" + "|".join(sorted(WORD_NUM, key=len, reverse=True)) + r")"
    r"\s*(?:other\s+)?(?:engineers?|people|devs?|developers?|members?|folks)",
    re.I)

# Predicates whose subject should be a project/system, not the person.
PROJECT_SCOPED = {"role_on", "team_size", "timeline", "outcome", "responsibility"}

STOP_SUBJECTS = {
    "", "self", "i", "me", "candidate", "unspecified", "it", "this", "that",
    "them", "they", "we", "us", "team", "project", "work", "thing", "role",
    "time", "lot", "whole", "end", "everything", "one",
}

_NUM = re.compile(r"(\d+(?:\.\d+)?)\s*(%|percent|ms|s|sec|seconds|k|m|rps|qps|x|"
                  r"people|engineers|members|years?|yrs?|months?|mos?)?", re.I)

UNIT_CANON = {
    "percent": "%", "sec": "s", "seconds": "s", "yrs": "years", "yr": "years",
    "year": "years", "mos": "months", "mo": "months", "month": "months",
    "people": "people", "engineers": "people", "members": "people",
    "qps": "rps",
}


def ownership_of(text: str) -> int | None:
    """Map a role phrase onto the ownership ladder. Highest match wins."""
    t = f" {text.lower().strip()} "
    best: int | None = None
    for level, words in OWNERSHIP_LADDER:
        for w in words:
            if re.search(rf"\b{re.escape(w)}\b", t):
                if best is None or level > best:
                    best = level
                break
    return best


def canon_unit(u: str | None) -> str:
    if not u:
        return ""
    u = u.lower().strip()
    return UNIT_CANON.get(u, u)


def parse_number(text: str, *, unit_hint: str = "") -> tuple[float | None, str]:
    """Digits first, then spelled-out numbers. Unit-aware when a hint is given."""
    t = text or ""
    # Prefer a match whose unit agrees with the predicate we're filling.
    want = {"people": ("people",), "months": ("months", "years", "weeks"),
            "years": ("years", "months")}.get(unit_hint, ())
    candidates: list[tuple[float, str]] = []
    for m in _NUM.finditer(t):
        try:
            val = float(m.group(1))
        except (TypeError, ValueError):
            continue
        unit = canon_unit(m.group(2))
        if unit == "k":
            val, unit = val * 1_000, ""
        elif unit == "m":
            val, unit = val * 1_000_000, ""
        candidates.append((val, unit))
    for m in _WORD_NUM_RE.finditer(t):
        candidates.append((float(WORD_NUM[m.group(1).lower()]), canon_unit(m.group(2))))
    if not candidates:
        return None, ""
    if want:
        for val, unit in candidates:
            if unit in want:
                return _to_months(val, unit, unit_hint)
    return candidates[0]


def _to_months(val: float, unit: str, hint: str) -> tuple[float, str]:
    """Normalize durations so 6 months and 0.5 years are comparable."""
    if hint == "months":
        if unit == "years":
            return val * 12, "months"
        if unit == "weeks":
            return round(val / 4.345, 2), "months"
    if hint == "years" and unit == "months":
        return round(val / 12.0, 3), "years"
    return val, unit


def detect_projects(text: str, lexicon: list[str] | None = None) -> list[str]:
    """Find project/system names in free text. Lexicon first, then patterns."""
    t = (text or "").lower()
    found: list[str] = []
    for p in lexicon or []:
        if not p or len(p) <= 3:
            continue
        if p in t:
            if p not in found:
                found.append(p)
            continue
        # People shorten project names in speech: a résumé's "Service Mesh
        # Rollout" becomes "our service mesh". Match the leading bigram so the
        # shortened form still resolves to the same ledger subject.
        words = p.split()
        if len(words) >= 3:
            bigram = " ".join(words[:2])
            if len(bigram) > 6 and bigram in t and p not in found:
                found.append(p)
    for m in _PROJECTISH.finditer(t):
        n = normalize_project(m.group(1))
        if n not in STOP_SUBJECTS and n not in found:
            found.append(n)
    return found


def extract_ownership_phrases(text: str, lexicon: list[str] | None = None
                              ) -> list[tuple[str, int, str]]:
    """Deterministically mine '<ownership verb> the <project>' from an answer.

    Returns (project, ownership_level, matched_span). This is the backbone of
    ownership-conflict detection: it does not depend on the LLM getting the
    `subject` field right, which small models frequently do not.
    """
    out: list[tuple[str, int, str]] = []
    t = text or ""
    projects_in_text = detect_projects(t, lexicon)
    for m in _OWNER_PHRASE.finditer(t):
        verb = re.sub(r"\s+", " ", m.group(1).lower())
        level = ownership_of(verb)
        if level is None:
            continue
        tail = normalize_project(m.group(2) or "")
        proj = ""
        # 1. The phrase's own tail, but only if it names a known or
        #    project-shaped thing. Otherwise it is just sentence filler
        #    ("and validated the output") and would poison the subject key.
        for p in projects_in_text:
            if p and (p in tail or tail in p):
                proj = p
                break
        if not proj and tail and tail not in STOP_SUBJECTS and len(tail) > 3:
            if tail in (lexicon or []) or _PROJECT_NOUN_RE.search(tail):
                proj = tail
        # 2. Fall back to the project the sentence is actually about.
        if not proj and projects_in_text:
            proj = _nearest(t, m.start(), projects_in_text)
        if not proj:
            continue
        out.append((canonicalize_project(proj, lexicon), level, m.group(0)[:120]))
    return out


def normalize_project(name: str) -> str:
    """Collapse project aliases so 'the billing service' == 'Billing Service'."""
    n = (name or "").lower().strip()
    n = re.sub(r"^(the|our|my|a|an)\s+", "", n)
    n = re.sub(r"\s+(project|system|service|platform|pipeline|app|application)$", "", n)
    n = re.sub(r"[^a-z0-9 ]+", "", n).strip()
    return re.sub(r"\s+", " ", n) or "unspecified"


EXTRACT_SYSTEM = """You are a precise information-extraction engine for interview transcripts.
You decompose a candidate's answer into ATOMIC, VERIFIABLE claims.

Rules:
- One fact per claim. Never merge two facts.
- `subject`: "self" for claims about the candidate personally, otherwise the
  normalized project/system/company name the claim is about.
- `predicate` MUST be one of: role_on, team_size, years_exp, used_tech, metric,
  outcome, responsibility, timeline, employment, education.
- `object`: the short value (role phrase, technology, metric name, company...).
- `polarity`: 1 if asserted, -1 if the candidate DENIES it ("I never used Kafka").
- `value_num` + `unit`: fill when a number is stated (team_size, metric, timeline, years_exp).
- `raw_span`: the EXACT substring of the answer the claim came from. Copy verbatim.
- `confidence`: 0..1, how clearly the answer states this.
- Extract claims about the candidate's own involvement especially carefully:
  if they say what they did on a project, that is a `role_on` claim.
- Do NOT invent facts. If the answer is vague, extract fewer claims.

Reply with JSON: {"claims": [{...}, ...]}"""


class ClaimExtractor:
    def __init__(self, llm: LLM):
        self.llm = llm

    def extract(self, answer: str, *, round_: int, turn_id: str,
                topic: str = "", source: str = "interview",
                context_projects: list[str] | None = None) -> list[Claim]:
        if not answer or len(answer.strip()) < 8:
            return []
        hint = ""
        if context_projects:
            hint = ("\nKnown project names so far (reuse these exact names when the answer "
                    f"refers to them): {', '.join(sorted(set(context_projects)))}")
        payload = self.llm.json(
            EXTRACT_SYSTEM,
            f"Interview answer to decompose:\n\"\"\"\n{answer.strip()}\n\"\"\"{hint}",
            label="extract_claims",
            max_tokens=900,
            temperature=0.1,
            fallback={"claims": []},
        )
        raw = payload.get("claims") if isinstance(payload, dict) else payload
        if not isinstance(raw, list):
            raw = []

        lex = list(context_projects or [])
        out: list[Claim] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            c = self._build(item, answer, round_, turn_id, topic, source, lex)
            if c is not None:
                out.append(c)

        # Deterministic augmentation: mine ownership claims the LLM missed or
        # mis-attributed. Small models routinely set subject="self" for a claim
        # that is really about a project, which would silently break pairing.
        out = self._augment_ownership(out, answer, round_, turn_id, topic, source, lex)
        out = self._retype_durations(out, lex)

        # Backstop: never let a round produce zero claims.
        if not out:
            out = self._fallback(answer, round_, turn_id, topic, source)
        return self._dedupe(out)[:14]

    @staticmethod
    def _retype_durations(claims: list[Claim], lex: list[str]) -> list[Claim]:
        """"I spent two years on the ingestion pipeline" is a project duration.

        The model emits that as years_exp (how long the PERSON has done
        something) rather than timeline (how long the PROJECT ran). They are
        then never compared against "it was a six month project", so an
        obvious padding of the timeline slips through. Re-type it when the
        object names a project rather than a technology.
        """
        for c in claims:
            if c.predicate != "years_exp" or c.value_num is None:
                continue
            target = canonicalize_project(c.object, lex)
            if c.object and (target in lex or _PROJECT_NOUN_RE.search(
                    normalize_project(c.object))):
                c.predicate = "timeline"
                c.subject = target
                c.object = ""
                c.value_num, c.unit = _to_months(c.value_num, "years", "months")
        return claims

    @staticmethod
    def _dedupe(claims: list[Claim]) -> list[Claim]:
        seen: dict[tuple, Claim] = {}
        for c in claims:
            k = (c.subject.lower(), c.predicate, c.object.lower()[:40],
                 c.value_num, c.polarity, c.ownership)
            prev = seen.get(k)
            if prev is None or c.confidence > prev.confidence:
                seen[k] = c
        return list(seen.values())

    def _augment_ownership(self, claims: list[Claim], answer: str, round_: int,
                           turn_id: str, topic: str, source: str,
                           lex: list[str]) -> list[Claim]:
        phrases = extract_ownership_phrases(answer, lex)
        if not phrases:
            return claims

        # Re-attribute LLM role_on claims that lost their project.
        for c in claims:
            if c.predicate == "role_on" and c.subject in STOP_SUBJECTS:
                best = max(phrases, key=lambda p: p[1])
                c.subject = best[0]
                if c.ownership is None:
                    c.ownership = best[1]
        # Project-scoped numerics stated about an unnamed subject.
        projs = detect_projects(answer, lex)
        if projs:
            for c in claims:
                if c.predicate in PROJECT_SCOPED and c.subject in STOP_SUBJECTS:
                    c.subject = projs[0]

        # Team size is stated conversationally ("a team of four") far more often
        # than the model emits it as a typed claim. Mine it deterministically.
        if projs and not any(c.predicate == "team_size" for c in claims):
            tm = _TEAM_SIZE_RE.search(answer or "")
            if tm:
                tok = tm.group("n").lower()
                n = WORD_NUM.get(tok)
                if n is None:
                    try:
                        n = float(tok)
                    except ValueError:
                        n = None
                if n:
                    claims.append(Claim(
                        subject=projs[0], predicate="team_size", object="",
                        value_num=float(n), unit="people",
                        raw_span=_sentence_around(answer, tm.group(0)),
                        confidence=0.85, topic=topic, source=source,
                        turn_id=turn_id, round=round_))

        existing = {(c.subject.lower(), c.ownership) for c in claims
                    if c.predicate == "role_on"}
        for proj, level, span in phrases:
            if (proj.lower(), level) in existing:
                continue
            existing.add((proj.lower(), level))
            claims.append(Claim(
                subject=proj, predicate="role_on", object=span.strip(),
                ownership=level, raw_span=_sentence_around(answer, span),
                confidence=0.88, topic=topic, source=source,
                turn_id=turn_id, round=round_))
        return claims

    def _build(self, item: dict[str, Any], answer: str, round_: int, turn_id: str,
               topic: str, source: str, lex: list[str] | None = None) -> Claim | None:
        pred = str(item.get("predicate", "")).strip().lower().replace(" ", "_")
        if pred not in PREDICATES:
            pred = {"role": "role_on", "tech": "used_tech", "team": "team_size",
                    "experience": "years_exp", "result": "outcome"}.get(pred, "")
            if not pred:
                return None
        subject = str(item.get("subject", "self")).strip().lower() or "self"
        if subject in ("i", "me", "candidate", "myself"):
            subject = "self"

        obj = str(item.get("object", "") or "").strip()
        span = str(item.get("raw_span", "") or "").strip() or answer.strip()[:200]

        polarity = item.get("polarity", 1)
        try:
            polarity = 1 if int(polarity) >= 0 else -1
        except (TypeError, ValueError):
            polarity = 1
        # Deterministic negation override on the actual span.
        if NEGATION.search(span) and pred == "used_tech":
            polarity = -1

        # Numbers: trust the model's digits, but fall back to parsing the span,
        # and force a canonical unit per predicate so values are comparable.
        hint = {"team_size": "people", "timeline": "months",
                "years_exp": "years"}.get(pred, "")
        val = item.get("value_num")
        try:
            val = float(val) if val is not None and str(val) != "" else None
        except (TypeError, ValueError):
            val = None
        unit = canon_unit(str(item.get("unit", "") or ""))
        if pred in ("team_size", "metric", "timeline", "years_exp"):
            pv, pu = parse_number(span or answer, unit_hint=hint)
            if val is None:
                val, unit = pv, unit or pu
            elif hint and unit and unit != hint:
                val, unit = _to_months(val, unit, hint) if hint == "months" else (val, unit)
        if hint and val is not None:
            if hint == "months" and unit in ("years", "weeks"):
                val, unit = _to_months(val, unit, "months")
            unit = hint

        own = None
        if pred == "role_on":
            own = ownership_of(obj)
            if own is None:
                own = ownership_of(span)

        # Project-scoped claims must be keyed by the project, not the person,
        # otherwise two statements about the same project never get compared.
        if pred in PROJECT_SCOPED:
            subject = coerce_subject(subject, span, answer, lex)
        elif subject not in STOP_SUBJECTS:
            subject = normalize_project(subject)

        try:
            conf = float(item.get("confidence", 0.8))
        except (TypeError, ValueError):
            conf = 0.8

        return Claim(
            subject=subject, predicate=pred, object=obj, polarity=polarity,
            value_num=val, unit=unit, ownership=own,
            topic=topic or str(item.get("topic", "") or ""),
            confidence=max(0.0, min(1.0, conf)),
            raw_span=span[:400], source=source, turn_id=turn_id, round=round_,
        )

    def _fallback(self, answer: str, round_: int, turn_id: str,
                  topic: str, source: str) -> list[Claim]:
        """Regex-only extraction so the ledger is never empty on LLM failure."""
        own = ownership_of(answer)
        claims: list[Claim] = []
        if own is not None:
            claims.append(Claim(
                subject=normalize_project(topic) if topic else "unspecified",
                predicate="role_on", object=answer.strip()[:60], ownership=own,
                raw_span=answer.strip()[:300], confidence=0.45, topic=topic,
                source=source, turn_id=turn_id, round=round_))
        val, unit = parse_number(answer)
        if val is not None:
            claims.append(Claim(
                subject="self", predicate="metric", object=topic or "stated figure",
                value_num=val, unit=unit, raw_span=answer.strip()[:300],
                confidence=0.4, topic=topic, source=source,
                turn_id=turn_id, round=round_))
        return claims


def _sentence_around(text: str, needle: str) -> str:
    """Return the full sentence containing `needle`, so evidence spans read well."""
    t = (text or "").strip()
    if not needle or needle not in t:
        return t[:300]
    i = t.index(needle)
    start = max((t.rfind(ch, 0, i) for ch in ".!?\n"), default=-1)
    end_candidates = [p for p in (t.find(ch, i + len(needle)) for ch in ".!?\n") if p != -1]
    end = min(end_candidates) + 1 if end_candidates else len(t)
    return t[start + 1:end].strip()[:300]


def _nearest(text: str, pos: int, projects: list[str]) -> str:
    """Pick the project mention closest to `pos` in the text."""
    best, best_d = "", 10**9
    low = (text or "").lower()
    for p in projects:
        i = low.find(p)
        while i != -1:
            d = abs(i - pos)
            if d < best_d:
                best, best_d = p, d
            i = low.find(p, i + 1)
    return best or (projects[0] if projects else "")


_ALIAS_STOP = {"the", "our", "my", "a", "an", "whole", "entire", "new", "old",
               "this", "that", "and", "reported", "on", "of", "for", "in", "at",
               "project", "work", "thing"}


def canonicalize_project(name: str, lexicon: list[str] | None = None) -> str:
    """Map a project alias onto its canonical ledger name.

    A single project is referred to many ways in one interview: the résumé says
    "Service Mesh Rollout", round 1 says "our service mesh", round 2 says "the
    rollout". Unless these collapse to one subject, two statements about the
    same project never get compared and the contradiction is silently missed.
    """
    n = normalize_project(name)
    if not n or not lexicon:
        return n
    for p in lexicon:
        if p == n:
            return p
    n_words = {w for w in n.split() if w not in _ALIAS_STOP and len(w) > 3}
    if not n_words:
        return n
    best, best_overlap = n, 0
    for p in lexicon:
        p_words = {w for w in p.split() if w not in _ALIAS_STOP and len(w) > 3}
        overlap = len(n_words & p_words)
        if overlap > best_overlap:
            best, best_overlap = p, overlap
    return best if best_overlap else n


def coerce_subject(subject: str, span: str, answer: str,
                   lexicon: list[str] | None = None) -> str:
    """Accept a subject only if it plausibly names a project; else re-derive it.

    Guards the claim key. A junk subject like "nd validated the output" would
    silently prevent two statements about the same project from ever being
    compared, which is a false negative we cannot see.
    """
    s = normalize_project(subject or "")
    lex = lexicon or []
    if s and s not in STOP_SUBJECTS:
        if s in lex or _PROJECT_NOUN_RE.search(s) or len(s.split()) <= 2:
            return canonicalize_project(s, lex)
    found = detect_projects(span, lex) or detect_projects(answer, lex)
    return canonicalize_project(found[0], lex) if found else "self"
