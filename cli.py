#!/usr/bin/env python3
"""Headless demo runner.

  python cli.py                      # run the flagship demo case (C01)
  python cli.py --case C08           # run a specific labelled case
  python cli.py --list               # list cases
  python cli.py --rounds 2 --quiet   # shorter run
  python cli.py --approve            # auto-approve at the human checkpoint
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from evalkit.cases import CASES, get_case, stats
from panel.candidate import ScriptedCandidate
from panel.config import SETTINGS
from panel.llm import LLM
from panel.memory import Memory
from panel.orchestrator import InterviewConfig, Orchestrator

C = {"hr": "\033[96m", "tech": "\033[93m", "bar": "\033[95m",
     "cand": "\033[92m", "sys": "\033[90m", "warn": "\033[91m",
     "ok": "\033[92m", "bold": "\033[1m", "off": "\033[0m",
     "mod": "\033[94m"}


def wrap(text: str, width: int = 88, indent: str = "    ") -> str:
    import textwrap
    return "\n".join(textwrap.fill(line, width, initial_indent=indent,
                                   subsequent_indent=indent)
                     for line in (text or "").splitlines() or [""])


def make_printer(quiet: bool):
    def p(phase: str, kind: str, d: dict):
        if kind == "session_created":
            print(f"\n{C['bold']}{'='*92}")
            print(f"  MULTI-AGENT INTERVIEW PANEL")
            print(f"{'='*92}{C['off']}")
            print(f"  session   : {d['session_id']}")
            print(f"  candidate : {d['candidate']}")
            print(f"  role      : {d['role']} ({d['seniority']})")
            print(f"  model     : {d['llm']['provider']} / {d['llm']['model']}")
            print(f"  rounds    : {d['rounds']}  ·  panel of {len(d['panel'])}")
            print(f"  tools     : {', '.join(d['tools_available'])}")

        elif kind == "resume_ingested":
            print(f"{C['sys']}  [memory] seeded {d['claims_seeded']} resume claims "
                  f"into the ledger (projects: {', '.join(d['projects']) or 'none'}){C['off']}")

        elif kind == "round_start":
            print(f"\n{C['bold']}{'─'*92}")
            print(f"  ROUND {d['round']} of {d['of']}")
            print(f"{'─'*92}{C['off']}")

        elif kind == "agent_plan" and not quiet:
            col = C.get(d["agent"], "")
            print(f"{C['sys']}  [plan] {col}{d['display']}{C['sys']} → "
                  f"{C['bold']}{d['action']}{C['off']}{C['sys']}"
                  f"{'  tool=' + d['tool'] if d.get('tool') else ''}"
                  f"  ({d['latency_s']}s)\n"
                  f"         ↳ {d['reasoning']}{C['off']}")

        elif kind == "tool_call":
            status = f"{C['ok']}ok{C['off']}" if d["ok"] else f"{C['warn']}FAILED{C['off']}"
            print(f"{C['sys']}  [tool] {d['tool']}({json.dumps(d['args'])}) → {status}"
                  f" {d['latency_ms']}ms{C['off']}")
            if not quiet:
                print(f"{C['sys']}{wrap(d['result_preview'][:300], indent='         ')}{C['off']}")

        elif kind == "question":
            col = C.get(d["agent"], "")
            tag = f" {C['warn']}[CHALLENGE]{C['off']}" if d["is_challenge"] else ""
            print(f"\n{col}{C['bold']}  {d['emoji']} {d['display']}{C['off']}{tag}"
                  f" {C['sys']}({d['action']} · {d['intent']}){C['off']}")
            print(f"{col}{wrap(d['text'])}{C['off']}")

        elif kind == "answer":
            flag = f" {C['warn']}[INJECTION FLAGGED]{C['off']}" if d["injection_flagged"] else ""
            print(f"{C['cand']}  🗣  Candidate{C['off']}{flag}")
            print(f"{C['cand']}{wrap(d['text'])}{C['off']}")

        elif kind == "claims_extracted" and not quiet:
            print(f"{C['sys']}  [ledger] +{d['count']} claims "
                  f"(total {d['ledger_size']}){C['off']}")
            for c in d["claims"][:6]:
                print(f"{C['sys']}         · {c['text']}{C['off']}")

        elif kind == "code_execution":
            r = d["result"] or {}
            v = r.get("verdict", "?")
            col = C["ok"] if v == "pass" else C["warn"]
            print(f"\n{C['bold']}  ⚙  CODE EXECUTED{C['off']} → {col}{v}{C['off']} "
                  f"({r.get('passed','?')}/{r.get('total','?')} tests, {d['latency_ms']}ms)")
            if r.get("static_check"):
                print(f"{C['warn']}     static check: {r['static_check']}{C['off']}")

        elif kind == "contradiction_detected":
            ev = d["evidence"]
            a, b = ev.get("claim_a", {}), ev.get("claim_b", {})
            print(f"\n{C['warn']}{C['bold']}  ⚠  CONTRADICTION DETECTED"
                  f"  [{d['type']} · {d['severity']} · score {d['score']}]{C['off']}")
            print(f"{C['warn']}     round {b.get('round')}: \"{(b.get('span') or '')[:150]}\"")
            print(f"{C['warn']}     round {a.get('round')}: \"{(a.get('span') or '')[:150]}\"")
            print(f"{C['warn']}     why: {d['rationale']}{C['off']}")
            print(f"{C['sys']}     rule: {ev.get('rule_detail','')}{C['off']}")

        elif kind == "policy_override":
            print(f"{C['sys']}  [policy] {d['display']}: {d['from_action']} → "
                  f"{C['bold']}{d['to_action']}{C['off']}{C['sys']}\n"
                  f"         ↳ {d['reason']}{C['off']}")

        elif kind == "replan":
            print(f"{C['sys']}  [re-plan] {d['trigger']} → {d['effect']}{C['off']}")

        elif kind == "contradiction_addressed":
            print(f"{C['sys']}  [state] contradiction {d['id']} marked addressed "
                  f"by {d['by']}{C['off']}")

        elif kind == "question_refused":
            print(f"\n{C['warn']}{C['bold']}  🛡  GUARDRAIL · QUESTION BLOCKED{C['off']}")
            print(f"{C['warn']}     blocked: \"{d['blocked_question'][:160]}\"")
            print(f"{C['warn']}     reason : {d['reason']}")
            print(f"{C['ok']}     asked instead: \"{d['replacement_question'][:160]}\"{C['off']}")

        elif kind == "injection_blocked":
            print(f"\n{C['warn']}{C['bold']}  🛡  GUARDRAIL · PROMPT INJECTION NEUTRALIZED{C['off']}")
            for f in d["findings"]:
                print(f"{C['warn']}     {f['kind']}: \"{f['matched']}\"{C['off']}")
            print(f"{C['sys']}     {d['detail']}{C['off']}")

        elif kind == "scorecard":
            col = C.get(d["agent"], "")
            print(f"\n{col}{C['bold']}  {d['emoji']} {d['display']} scorecard{C['off']}"
                  f"  → {C['bold']}{d['recommendation'].upper()}{C['off']}"
                  f"  (overall {d['overall']}/5, confidence {d['confidence']})")
            for k, v in d["scores"].items():
                bar = "█" * int(round(v)) + "·" * (5 - int(round(v)))
                print(f"      {bar}  {v:.1f}  {k}")
            for s in d["strengths"][:3]:
                print(f"{C['ok']}      + {s}{C['off']}")
            for s in d["concerns"][:3]:
                print(f"{C['warn']}      − {s}{C['off']}")
            if d.get("integrity_note"):
                print(f"{C['sys']}      integrity: {d['integrity_note']}{C['off']}")

        elif kind == "debate_start":
            print(f"\n{C['bold']}{'─'*92}")
            print(f"  PANEL DEBATE   initial positions: "
                  f"{', '.join(f'{k}={v}' for k, v in d['initial_positions'].items())}")
            print(f"{'─'*92}{C['off']}")

        elif kind == "debate_turn":
            col = C.get(d["agent"], "")
            moved = (f"  {C['warn']}CHANGED {d['from']} → {d['position']}{C['off']}"
                     if d["changed"] else f"  {C['sys']}held {d['position']}{C['off']}")
            print(f"\n{col}  {d['emoji']} {d['display']}{C['off']} (debate r{d['round']}){moved}")
            print(f"{col}{wrap(d['argument'])}{C['off']}")
            if d["changed"] and d["change_reason"]:
                print(f"{C['sys']}      why moved: {d['change_reason']}{C['off']}")
            if d.get("concession"):
                print(f"{C['sys']}      concedes: {d['concession']}{C['off']}")

        elif kind == "debate_skipped":
            print(f"{C['sys']}  [debate] closed early: {d['reason']}{C['off']}")

        elif kind == "consensus_computed":
            print(f"\n{C['bold']}{'─'*92}")
            print(f"  CONSENSUS")
            print(f"{'─'*92}{C['off']}")
            print(f"  votes          : {json.dumps(d['votes'])}")
            print(f"  hire votes     : {d['hire_votes']}/3")
            print(f"  weighted score : {d['weighted_score']} (bar {d['threshold']})")
            print(f"  panel agreement: {d['panel_agreement']}  spread {d['score_spread']}")
            print(f"  {C['bold']}rules fired:{C['off']}")
            for r in d["rules_fired"]:
                print(f"{C['warn']}{wrap(r, indent='      → ')}{C['off']}")

        elif kind == "verdict":
            dec = d["decision"]
            col = C["ok"] if dec == "hire" else C["warn"]
            print(f"\n{col}{C['bold']}{'='*92}")
            print(f"  DECISION: {dec.upper().replace('_',' ')}"
                  f"   (confidence {d['confidence']})")
            print(f"{'='*92}{C['off']}")
            print(f"{C['mod']}{wrap(d['rationale'])}{C['off']}")
            if d.get("decision_drivers"):
                print(f"\n  {C['bold']}decision drivers:{C['off']}")
                for x in d["decision_drivers"]:
                    print(f"      · {x}")
            if d.get("evidence"):
                print(f"\n  {C['bold']}key evidence:{C['off']}")
                for x in d["evidence"]:
                    print(f"      · {x}")
            if d.get("dissent"):
                print(f"\n  {C['warn']}{C['bold']}recorded dissent:{C['off']}")
                for x in d["dissent"]:
                    print(f"{C['warn']}      · {x['display']} voted {x['position']} "
                          f"(overall {x['overall']}){C['off']}")
                    for cc in x.get("their_concerns", [])[:2]:
                        print(f"{C['warn']}          − {cc}{C['off']}")
            else:
                print(f"\n  {C['ok']}panel was unanimous{C['off']}")
            if d.get("risk_if_hired"):
                print(f"\n  risk if hired : {d['risk_if_hired']}")
            if d.get("recommended_next_step"):
                print(f"  next step     : {d['recommended_next_step']}")

        elif kind == "approval_required":
            print(f"\n{C['warn']}{C['bold']}{'='*92}")
            print(f"  🛡  HUMAN APPROVAL CHECKPOINT")
            print(f"{'='*92}{C['off']}")
            print(wrap(d["message"], indent="  "))
            print(f"  options: {', '.join(d['options'])}")

        elif kind == "approval_recorded":
            print(f"\n{C['ok']}  ✓ human {d['decision']} recorded "
                  f"by {d['by']}{C['off']}"
                  + (f" — \"{d['note']}\"" if d.get("note") else ""))

        elif kind == "report_written":
            print(f"{C['sys']}  [export] {d['path']} ({d['bytes']} bytes){C['off']}")

    return p


def main():
    ap = argparse.ArgumentParser(description="Multi-agent interview panel")
    ap.add_argument("--case", default="C01", help="evaluation case id (default C01)")
    ap.add_argument("--rounds", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="hide planner/ledger chatter")
    ap.add_argument("--approve", action="store_true", help="auto-approve the verdict")
    ap.add_argument("--speech", action="store_true", help="speak questions aloud")
    ap.add_argument("--json", default="", help="also write the report to this path")
    args = ap.parse_args()

    if args.list:
        print(f"\n{json.dumps(stats(), indent=2)}\n")
        for c in CASES:
            print(f"  {c['id']:28} {c['name']:18} expect={c['expected_decision']:8} "
                  f"planted={len(c['ground_truth'])}")
            if c.get("notes"):
                print(f"{C['sys']}{wrap(c['notes'], 86, '      ')}{C['off']}")
        return 0

    case = get_case(args.case)
    if args.speech:
        SETTINGS.enable_speech = True

    llm = LLM()
    mem = Memory()
    orch = Orchestrator(llm=llm, memory=mem, settings=SETTINGS,
                        emit_fn=make_printer(args.quiet))

    cfg = InterviewConfig(
        candidate_name=case["name"], role=case["role"],
        seniority=case["seniority"], rounds=args.rounds or SETTINGS.rounds,
        enable_speech=args.speech)

    orch.setup(cfg, case["resume"])
    cand = ScriptedCandidate(
        name=case["name"], persona=case["persona"], resume=case["resume"],
        bank=case["bank"], llm=llm, code_answer=case.get("code_answer", ""))

    t0 = time.time()
    report = orch.run_interview(cand)

    if SETTINGS.require_human_approval:
        if args.approve:
            decision, by, note = "approve", "demo-operator", "auto-approved via --approve"
        else:
            try:
                raw = input(f"\n  {C['bold']}approve / override / reject "
                            f"(or enter to skip): {C['off']}").strip().lower()
            except (EOFError, KeyboardInterrupt):
                raw = ""
            if raw in ("approve", "override", "reject"):
                decision = raw
                by = input("  your name: ").strip() or "human-reviewer"
                note = input("  note (optional): ").strip()
            else:
                decision = ""
                print(f"{C['sys']}  verdict left PENDING — it is not released "
                      f"to the hiring team.{C['off']}")
        if decision:
            report = orch.record_approval(decision, by, note)

    # ---- run summary -------------------------------------------------
    mb = report["minimum_bar"]
    cs = report["contradiction_summary"]
    print(f"\n{C['bold']}{'='*92}")
    print(f"  RUN SUMMARY  ·  {round(time.time()-t0,1)}s wall clock")
    print(f"{'='*92}{C['off']}")
    print(f"  LLM calls            : {report['llm_usage']['calls']} "
          f"(mean {report['llm_usage']['mean_latency_s']}s, "
          f"{report['llm_usage']['json_repairs']} json repairs, "
          f"{report['llm_usage']['failures']} hard failures)")
    print(f"  tool calls           : {report['tool_usage']['total_calls']} across "
          f"{report['tool_usage']['distinct_tools']} distinct tools "
          f"(success {report['tool_usage']['success_rate']})")
    print(f"  claims in ledger     : {mb['memory']['claims_persisted']}")
    print(f"  contradictions       : {cs['total']} "
          f"({cs['cross_round']} cross-round, {cs['addressed_in_interview']} confronted live)")
    print(f"                         by severity {json.dumps(cs['by_severity'])}")
    print(f"                         by type {json.dumps(cs['by_type'])}")
    print(f"  detector             : {report['detector_stats']}")
    print(f"  guardrails           : {mb['guardrails']['refusals_demonstrated']} refusals, "
          f"{mb['guardrails']['injections_neutralized']} injections neutralized, "
          f"{mb['guardrails']['tool_failures_handled']} tool failures handled")
    print(f"  re-plans triggered   : {mb['agentic_loop']['replans_triggered']}")
    print(f"\n  {C['bold']}MINIMUM BAR{C['off']}")
    for k in ("agentic_loop", "tools", "memory", "guardrails"):
        ok = mb[k]["satisfied"]
        print(f"    {C['ok'] if ok else C['warn']}{'✓' if ok else '✗'}{C['off']} {k}")

    print(f"\n  expected (case label): {case['expected_decision']}")
    print(f"  panel decided        : {report['verdict']['decision']}")
    match = case["expected_decision"] == report["verdict"]["decision"]
    print(f"  {'✓ matches label' if match else '✗ differs from label'}")
    print(f"\n  artifact: artifacts/interview_{report['session_id']}.json")
    print(f"  database: {mem.path}\n")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  report also written to {args.json}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
