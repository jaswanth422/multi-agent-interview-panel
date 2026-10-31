# Spoken pitch — ~75 seconds

Say this while the live run streams behind you. Timings assume you clicked
**Start** at 0:00 and the first contradiction lands at **0:19**.

---

## 0:00 — Click Start, then open with the story (25 sec)

> Think about the last phone screen at your company.
>
> A candidate says: *"I led the billing migration."* The screener writes it down
> and moves on. Nobody asks what **led** means — the recruiter can't tell the
> difference, and the engineers at the onsite assume somebody earlier already
> checked.
>
> Three rounds later, four hours of senior engineering time later, it turns out
> he wrote the tests.
>
> The résumé wasn't lying. It was **inflated**. And inflation is invisible to one
> interviewer asking one set of questions.

Then turn:

> An AI interviewer makes that worse, not better. One model is easy to game and
> gives you one-sided scores. So we didn't build an interviewer.
> **We built a panel.**

---

## 0:19 — STOP TALKING

The red contradiction card appears. Let them read it. Then:

> There it is. His résumé says three years of Kafka. He just said he's never
> used it in production. We quote both, with round numbers, and the rule that
> fired.

Three seconds later the **[CHALLENGE]** question appears:

> And that's the loop closing — a new contradiction re-plans the next agent,
> which confronts it inside the interview instead of just logging it.

---

## 0:30 — What kind of agent this is (15 sec)

> It's a multi-agent panel with shared memory, and it's deliberately **hybrid**.
>
> The language model does what it's good at: asking questions, pulling claims out
> of messy speech, writing the reasoning. A **deterministic rule engine** does
> what the model is bad at: comparing facts and deciding. That's why our
> contradiction detection is measurable instead of a vibe.

| Agent | Its job |
|---|---|
| **HR** | ownership and accountability — listens for "we" hiding that you did very little |
| **Technical** | depth, trade-offs, and a live coding task |
| **Bar-Raiser** | escalates every round, hunts **disconfirming** evidence. Paid to disagree. |

---

## 0:46 — The second catch

Two more cards land: team of four versus twelve, and led-the-project versus
only-wrote-tests.

> That second one is the exact example printed on our problem sheet.

---

## 1:00 — The workflow (15 sec — trace it on screen)

> Every turn runs the same loop.
> **Plan** — the agent reads the claim ledger and open contradictions, picks its move.
> **Act** — calls a tool if it needs evidence, then asks one question.
> **Observe** — the answer returns, screened as untrusted input.
> **Extract** — broken into atomic claims, written to SQLite.
> **Detect** — rules compare it against the entire interview *and* the résumé.
> **Re-plan** — a new contradiction rewrites the next agent's priorities.
>
> After the rounds: three independent scorecards, a debate where panelists can
> change position with stated reasons, consensus from explicit rules — and then a
> human has to sign it.

---

## 1:20 — The best line in the demo

Scorecards appear. All three say hire.

> Watch this. All three panelists scored him a **hire** — 3.15, 3.0, 3.5.
> The verdict is going to be **no hire**. Because evidence gates decide before
> scores do: three unreconciled contradictions is an integrity failure, and that
> overrides a favourable impression. That is the entire point of the system.

---

## 2:06 — Verdict lands. Ten seconds on the board (Tab 1)

> Same system, different outcomes. Vikram tried to prompt-inject the panel —
> gate G1. Harsh is an honest employee #3 who owned the backend *and* ran deploys
> *and* answered support tickets — an ownership spread of four, the exact
> signature of inflation, and it's true. So we hire him. The system can say yes.

---

## 2:35 — Reliability

> Seventeen evaluation cases in two tracks. Precision and recall 1.0, nine clean
> controls, zero false positives.

If they press on reliability, give them the real story instead of the number:

> Our first twelve cases scored 1.0. Then we wrote five startup cases and it
> dropped to 0.57 — it found six real bugs, including one where an honest
> generalist got flagged as a liar. We fixed them. We also found the panel never
> said "hire": panelists were scoring 3.5 out of 5 and then writing
> "strong_no_hire", so we now derive the decision from the rubric-anchored
> scores. Eight of twelve to eleven of twelve.

---

## 2:50 — Click ✓ Approve

> No verdict is final until a human signs it. We find the inconsistency and show
> the evidence. A person decides what it means.

## Close

> It doesn't decide who to hire. It makes sure that by the time a human decides,
> nobody's résumé is lying to them.
