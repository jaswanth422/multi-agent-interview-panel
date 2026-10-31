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

`python -m evalkit.run_eval --mode detector` on `qwen2.5:3b`, 12 cases, 248 s:

| metric | value |
|---|---|
| contradiction precision | **1.00** (8 TP, 0 FP) |
| contradiction recall | **1.00** (8 of 8 planted) |
| F1 | **1.00** |
| clean-case precision | **1.00** (7 of 7 clean cases kept clean) |
| claims extracted | 160 across 12 cases |

Per-case: all 5 planted contradiction types detected (ownership, numeric, timeline,
polarity, exclusivity), and **zero** false positives on the 7 clean candidates —
including the three hard ones (C08 coexisting scope, C10 narrated progression,
C11 casual rounding).

> **Read this honestly.** F1 = 1.00 is on **12 cases we wrote ourselves**, and we
> iterated against them while building the detector. It demonstrates that the
> mechanism works on the failure modes we targeted — it is *not* a claim of
> general accuracy, and the sample is far too small for one. The numbers that
> transfer are the design choices they forced: deterministic pairing, explicit
> tolerance bands, object-scoped numerics, alias canonicalization and the
> benign-pattern filter. Earlier builds scored P 0.86 / R 0.75 and over-fired
> 25 findings on a single interview; the eval set is what exposed that.

Operational metrics from `--mode full` (decision accuracy, LLM calls, distinct
tools, tool success rate, handled failures, contradictions confronted live) land in
`artifacts/eval_full.json`; detector results in `artifacts/eval_detector.json`.

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
