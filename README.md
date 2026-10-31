# 🎙 Multi-Agent Interview Panel

**ziro.digital · National Level Agentic AI Hackathon 2026 · Grand Finale**
**Problem 02 — Multi-Agent Interview Panel (HARD, 1.1× multiplier)**

> **The problem.** A single AI interviewer is easy to game and gives one-sided scores.

Three independent agents (**HR**, **Technical**, **Bar-Raiser**) interview a candidate in
rotation, each building on earlier answers. Every answer is decomposed into atomic
claims in a persistent ledger, cross-checked against the whole interview *and* the
résumé, contradictions are confronted live, and the panel then **debates** to a
hire / no-hire decision with cited evidence and recorded dissent.

---

**Repository:** https://github.com/jaswanth422/multi-agent-interview-panel

## Run it

```bash
cd interview-panel
uv venv --python 3.11 && uv pip install fastapi "uvicorn[standard]" httpx pydantic

# Live demo UI  → http://localhost:8800
.venv/bin/python -m uvicorn server.app:app --port 8800

# Terminal demo (the flagship case)
.venv/bin/python cli.py --case C01 --rounds 3

# Evaluation over all 12 labelled cases
.venv/bin/python -m evalkit.run_eval --mode detector
.venv/bin/python -m evalkit.run_eval --mode full --rounds 2
```

**No API key required.** It runs on a local Ollama model (`qwen2.5:3b`) by default, so
the live demo cannot fail on a network hiccup or a rate limit. Set `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY` or `GEMINI_API_KEY` and it
switches automatically — `PANEL_PROVIDER` / `PANEL_MODEL` override.

---

## The core idea: a claim ledger, not a chat log

Asking an LLM "did the candidate contradict themselves?" over a long transcript is
unreliable in both directions — it misses cross-round pairs it can't hold in context,
and it invents conflicts that aren't there. So contradiction detection here is **two
stage and mostly deterministic**:

```
answer ──► EXTRACT ──► atomic typed claims ──► SQLite ledger
                                                   │
                        ┌──────────────────────────┘
                        ▼
            STAGE 1  rule engine over structured fields
                     (exhaustive, explainable, zero hallucination)
                        │  candidate pairs only
                        ▼
            STAGE 1.5 deterministic benign-pattern filter
                     (career progression, coexisting scope)
                        │
                        ▼
            STAGE 2  LLM adjudicates ONLY those pairs
                     writes the rationale + challenge question
                        │
                        ▼
                 contradiction + 2 claim ids + verbatim spans
```

Each claim is normalized into `(subject, predicate, object, polarity, value, unit,
ownership)` with a controlled predicate vocabulary. The key piece is the
**ownership ladder**, which turns role language into a comparable integer:

| level | language |
|---|---|
| 4 | led, owned, drove, headed, spearheaded, tech lead |
| 3 | designed, architected, built, implemented |
| 2 | contributed, helped, worked on, collaborated |
| 1 | tested, wrote tests, validated, reviewed |
| 0 | observed, shadowed, sat in on |

A gap of ≥ 2 on the same project is the exact failure the problem statement names:
*"you said you led the project, but earlier you said you only did testing."*

### Why each rule exists

| rule | fires on | example |
|---|---|---|
| R1 ownership | ladder gap ≥ 2, same project | "led the billing migration" vs "mainly wrote tests" |
| R2 polarity | same claim asserted and denied | résumé says 3y Kafka vs "never used Kafka" |
| R3 exclusivity | blanket denial vs stated usage | "never configured Istio" vs "I rolled out Istio" |
| R4 numeric | same quantity, beyond tolerance | team of 4 vs team of 12 |
| R5 timeline | durations beyond tolerance | "two years" vs "a six month project" |

Tolerances are deliberate (`team_size` 34%, `metric` 50%): people round numbers when
they speak, and flagging that would destroy trust in the tool. Durations are
normalized to months so "2 years" and "6 months" are comparable. Numeric predicates
that are scoped by their object (`years_exp`, `metric`) require the object to match —
without that, *"6 years of Python"* and *"3 years of Kafka"* pair up as a conflict,
which is nonsense.

---

## Agentic loop

Each panelist runs its own plan → act → observe cycle, and a new contradiction
**rewrites the next agent's priorities** mid-interview:

```
setup    provision session + résumé DB, seed the ledger from the résumé
         (so a round-1 answer can already contradict the written record)

round N  for each panelist in rotation:
           PLAN      inspect ledger, open contradictions, panel coverage → choose action
           ACT       optionally call a tool, then ask exactly one question
           OBSERVE   candidate answers (screened as untrusted input)
           EXTRACT   answer → atomic claims → ledger
           DETECT    rule engine + adjudication over the WHOLE ledger
           RE-PLAN   a new contradiction primes the next panelist to confront it

score    three independent, evidence-citing scorecards (no peeking)
debate   panelists see each other and may revise, with stated reasons
consensus explicit rule set + Bar-Raiser veto + recorded dissent
approval HUMAN checkpoint — nothing is final without a person
export   PII-redacted signed artifact
```

Agent actions: `ask_new`, `probe_prior`, `challenge_contradiction`,
`verify_with_tool`, `request_code`.

The three agents genuinely differ — separate system prompts, competency maps,
scoring rubrics and weights (HR 0.9, Technical 1.2, Bar-Raiser 1.3). The Bar-Raiser
is the only one that escalates difficulty each round and is instructed to seek
**disconfirming** evidence for the emerging consensus.

### Protocol overrides

A 3B planner is inconsistent about doing the things the panel *must* do, so three
rules are applied deterministically on top of each agent's own plan — and every
override is written to the event trace, visible in the UI:

- **P1** an unaddressed contradiction must be confronted before any new topic opens
- **P2** every panelist must ground at least one question in tool evidence
- **P3** the technical panelist must set a live coding task by round 2

---

## Debate and consensus

The panel does **not** average three scores — that lets a confident HR impression
drown out a specific technical objection. Instead:

1. Three independent scorecards, each citing claim ids.
2. If they disagree, each panelist sees the others and may revise — a change must
   carry a reason, so you can watch opinions actually move.
3. Consensus from an explicit, auditable rule set:

| rule | effect |
|---|---|
| **R1** Bar-Raiser integrity veto | Bar-Raiser votes no-hire + unresolved high-severity contradiction → **no_hire**, dispositive |
| **R2** integrity failure | any prompt-injection attempt → **no_hire** |
| **R3** majority + bar | ≥ 2 hire votes **and** weighted score ≥ 3.2 → **hire** |
| **R4** unanimous | no panelist recommends hire → **no_hire** |
| **R5** split panel | a specific unanswered objection outweighs a positive general impression → **no_hire** (the reversible choice) |

Dissent is **preserved in the record, never papered over**. The moderator explains the
computed decision but cannot change it.

---

## Minimum bar

| requirement | how it's met |
|---|---|
| **Agentic loop** | plan → act → observe → extract → detect → re-plan → score → debate → consensus → approval. **27–60 LLM calls** per interview |
| **Tools** | **5 real tools**, all logged with latency: sandboxed **code execution**, **SQLite** résumé DB, **GitHub REST API**, **Wikipedia/DuckDuckGo** fact-check, macOS **`say`** TTS |
| **Memory** | SQLite claim ledger — claims, turns, tool calls, guardrail events, verdicts, full event trace. Survives across sessions; any run is replayable from the DB |
| **Evaluation** | **12 labelled cases**, 8 planted contradictions, 5 clean controls. Precision / recall / F1 + decision accuracy |
| **Guardrails** | human approval checkpoint **+** 4 demonstrated refusal/failure paths (below) |
| **Data** | synthetic candidate personas + real public APIs (GitHub, Wikipedia) |

### Guardrails in detail

1. **Human approval checkpoint** — no verdict is released until a person clicks
   approve / override / reject. Signed, attributed and timestamped in SQLite.
2. **Prohibited question refusal** — 7 protected-attribute categories (age, family,
   religion, gender, nationality/visa, health, salary history). The panel's *own*
   outbound questions are screened; a blocked question is replaced with a lawful
   probe of the same competency and the refusal is logged as evidence.
3. **Prompt injection defense** — candidate answers are untrusted input. 5 injection
   families detected, neutralized, and treated as an **integrity signal** that forces
   no-hire via consensus rule R2. (Case `C05` demonstrates this.)
4. **Code sandbox refusal** — candidate code is statically screened before execution;
   anything reaching for `os`, `subprocess`, `open`, `eval` is **refused, not run**,
   then executed in a subprocess with a hard timeout and a stripped environment.
   (Case `C07` demonstrates this.)
5. **Tool failure handling** — every tool degrades to a structured failure object
   instead of raising; the interview continues and the handled failure is counted.
6. **PII minimization** — exported artifacts are deep-redacted (email, phone,
   national-ID and card patterns).

---

## Evaluation

12 cases in `evalkit/cases.py`, each carrying an answer key. Metrics are computed by
greedily aligning detections to planted contradictions (`evalkit/run_eval.py`).

**Five cases are deliberately clean**, and they are the hard ones — they measure
false positives, which is what makes a tool like this trustworthy or useless:

- **C08** — "owned the idempotency design" **and** "wrote the integration tests
  myself". A real ownership-ladder gap of 3 on one project that the rules *will*
  flag. Coexisting scope, not an inconsistency.
- **C10** — "started as a contributor … then took over as lead". A ladder gap of 2
  that is an explained career progression.
- **C11** — 5 vs 6 people, 30% vs 35%. Casual rounding, inside tolerance.
- **C12** — "evaluated Istio in a sandbox" vs "never ran a mesh in production".
  Correctly scoped statements that must not be collapsed into a conflict.

### Measured results

**17 cases in two tracks**, scored separately so neither hides behind the other.
`qwen2.5:3b`, detector mode:

| | all 17 | core (12) | startup (5) |
|---|---|---|---|
| contradiction precision | **1.00** | **1.00** | **1.00** |
| contradiction recall | **1.00** (14/14) | **1.00** (8/8) | **1.00** (6/6) |
| F1 | **1.00** | **1.00** | **1.00** |
| clean-case precision | **1.00** (9/9) | **1.00** (7/7) | **1.00** (2/2) |

Full 17-case run: `artifacts/eval_all17.json` (384 s, 246 claims extracted).

All five conflict types detected (ownership, numeric, timeline, polarity,
exclusivity), zero false positives across all 9 clean controls.

**Hire/no-hire decision accuracy: 11/12** on the core track after the consensus
redesign below (measured 8/12 before it).

### What the eval set actually caught

This is the honest part, and it is the reason the eval exists:

1. **An early build over-fired 25 contradictions on a single interview.** Fixed by
   deterministic de-duplication and per-signature suppression.
2. **`years_exp` pairing was nonsense** — "6 years of Python" vs "3 years of Kafka"
   was reported as a conflict. Fixed by object-scoping numeric predicates.
3. **The startup track initially scored P 0.50 / R 0.67** and false-positived on
   *both* clean cases. Five real bugs surfaced, all since fixed:
   - `"wrote the"` sat at ladder level 3, so **"wrote the tests"** scored 3 not 1,
     collapsing the exact gap we exist to detect.
   - `"part of"` matched *"the hardest **part of** the core backend"*, turning a
     difficulty statement into an ownership claim.
   - The unit regex matched the `m` in *"200 **m**erchants"* → 200,000,000 years.
   - `"15,000 downloads"` became a team size of 15 (comma split the number).
   - The coexistence filter read one sentence, so it missed *"I **also** wrote all
     the tests myself"* sitting in the next one — which flagged an honest
     employee-#3 generalist as a liar.
4. **The panel never said "hire".** Decision accuracy was 8/12, and *every* error
   was a strong candidate rejected — the same score a system that blindly answers
   "no_hire" would get. Root cause: panelists scored a candidate **3.5/5** (above
   their own stated bar of 3) and then wrote `strong_no_hire`. The prose label is
   the least reliable field a small model emits. Now the recommendation is
   **derived from the rubric-anchored scores** and the stated label is kept only
   for audit (`llm_stated_recommendation`, `label_overridden`).
5. **Scores alone could not decide either.** A candidate with three proven
   contradictions scored 3.05 while a flawless one scored 3.28. So the consensus
   rules were restructured: **objective evidence gates first** (G1 injection,
   G2 unresolved high-severity contradiction, G3 code refused/failed), and panel
   scores only arbitrate candidates whose record is already clean (S1/S2).

> **Read the 1.00 honestly.** These are **17 cases we wrote ourselves**, and we
> iterated against them. It shows the mechanism works on the failure modes we
> targeted; it is not a claim of general accuracy and the sample is far too small
> for one. What transfers is the list above — every item is a defect the eval
> found and we fixed. The remaining known miss is C11 (expected hire, decided
> no_hire): the model scored it 2.4/2.4/2.0 and that is a model-capability limit,
> not a logic bug.

Results land in `artifacts/eval_core.json`, `artifacts/eval_startup.json` and
`artifacts/eval_full.json`.

---

## Demo script (3 minutes)

1. **Start** `C01 · Rahul Menon` at 3 rounds. The answer-key panel shows what's
   planted — judges can see we're not grading ourselves.
2. **Watch the loop.** Planner lines show each agent *choosing* an action; tool lines
   show real GitHub / SQLite / Wikipedia calls with latencies; ledger lines show
   answers becoming atomic claims.
3. **The catch.** A red card appears quoting both statements verbatim with round
   numbers, the deterministic rule that fired, and the adjudicator's reasoning —
   then a `re-plan` line, then the next panelist opens with **[CHALLENGE]**,
   confronting it live. *That is the money shot.*
4. **Code execution.** The technical panelist sets a live task; the candidate's
   quadratic, index-ignoring solution fails the hidden tests in a sandbox.
5. **Debate.** Three independent scorecards, then panelists arguing, with
   `CHANGED hire → no_hire` badges where someone actually moved.
6. **Consensus.** The exact rules that fired, the Bar-Raiser veto, recorded dissent.
7. **Approval gate.** Click **Approve** — the verdict was pending until a human
   signed it.

Then run **`C05`** (injection neutralized → forced no-hire) and **`C08`** (clean
candidate, zero false positives, hire) to show both guardrails and precision.

---

## Layout

```
panel/
  config.py          settings, provider auto-detection
  llm.py             5 providers, JSON-mode, repair-retry, usage metering
  memory.py          SQLite claim ledger + full audit trail
  claims.py          extraction, ownership ladder, project resolution, unit normalization
  contradictions.py  rule engine + benign filter + LLM adjudication
  tools.py           5 real tools, all logged and failure-tolerant
  guardrails.py      refusals, injection defense, PII redaction, approval gate
  agents.py          HR / Technical / Bar-Raiser + planning + scoring
  verdict.py         debate, consensus rules, moderator narrative
  candidate.py       candidate simulator (deterministic planting + LLM improvisation)
  orchestrator.py    the agentic loop
evalkit/
  cases.py           12 labelled cases with ground truth
  run_eval.py        metrics harness
server/
  app.py             FastAPI + WebSocket
  static/index.html  live demo UI
cli.py               terminal demo
artifacts/           SQLite DB + exported reports + eval results
```

### Tuning

`PANEL_ROUNDS`, `PANEL_DEBATE_ROUNDS`, `PANEL_HIRE_THRESHOLD`,
`PANEL_REQUIRE_APPROVAL`, `PANEL_ENABLE_WEB`, `PANEL_ENABLE_GITHUB`,
`PANEL_ENABLE_SPEECH`, `PANEL_SEED`, `GITHUB_TOKEN` (raises the API rate limit).

---

## Where this is actually used

**The leak it plugs.** Résumé inflation is discovered too late. A candidate says
"I led the billing migration"; nobody probes it in the phone screen because the
screener can't tell what to ask. It surfaces at the onsite — after a recruiter, a
hiring manager and two senior engineers have each spent 45 minutes. At 50 screens
per role that's tens of hours of the most expensive calendar time in the company,
spent on a claim that should have failed on day one. This catches it in round one,
with the two conflicting sentences quoted and attributed to their rounds.

**Who it is for**

- **Startups and early-stage companies.** No recruiter, founders interviewing
  between everything else, no structured process — and one bad hire at ten people
  is 10% of the company. Highest need, least process. This is why the startup
  track exists: `C13` ("CTO" of four people) and `C15` (honest employee #3) are
  *their* candidates, not a large enterprise's.
- **Staffing and recruitment agencies.** They submit candidates to clients; one
  caught inflating damages the relationship that is their whole business.
  Pre-vetting with an evidence trail is directly billable.
- **High-volume funnels.** Run the screens overnight, hand humans a shortlist with
  flagged inconsistencies and cited evidence attached.
- **Litigation-exposed employers.** They must answer "why was this person
  rejected?" months later. The audit trail, cited claim ids and EEOC-style
  question screening are what make an AI interviewer deployable at all.

**For an AI career-intelligence platform specifically**, there are two products
here, not one: an employer-facing screen, and a candidate-facing rehearsal tool
that catches *your own* inconsistencies — and shows you the two sentences that
clashed — before a real interviewer does.

**What makes it deployable rather than a demo** is the compliance layer:
EEOC-style screening of the panel's *own* questions across 7 protected
categories, a named human approval gate on every decision, a full SQLite audit
trail, PII-redacted exports, and verdicts bound to cited claim ids.

**What it is not: a decision-maker.** It does not hire anyone. The human approval
gate is not a rubric checkbox, it is the product's position — the machine finds
the inconsistency and shows the evidence; a person decides what it means.

## Honest limitations

- The default local model is **3B**. It is enough because the reliability-critical
  work (pairing, comparison, tolerances, benign-pattern filtering) is deterministic —
  the model does extraction and narration. A frontier model improves the *prose* and
  the adjudicator's judgement, not the detection mechanics.
- The candidate is simulated. That is a deliberate trade: it is what makes ground
  truth, and therefore real metrics, possible. The UI also supports a human
  candidate (`mode: human`) answering live.
- A ~0.8 s/call local model means a 3-round interview takes roughly 2–4 minutes
  end to end. The UI streams every step so the wait is the demo.
