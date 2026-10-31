#!/usr/bin/env python3
"""Evaluation harness over the 12 labelled cases.

Two modes:

  detector (default)  Feed each case's scripted answers through the real
                      extraction + detection pipeline and score the findings
                      against the planted answer key.
                      Reports precision / recall / F1 and the false-positive
                      rate on clean candidates. ~2 min.

  full                Run the complete multi-agent interview for each case and
                      additionally score hire/no-hire decision accuracy.
                      Slower (several minutes per case).

  python -m evalkit.run_eval
  python -m evalkit.run_eval --mode full --rounds 2
  python -m evalkit.run_eval --cases C01,C08,C10
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any

from evalkit.cases import CASES, stats
from panel.candidate import ScriptedCandidate
from panel.claims import ClaimExtractor, normalize_project
from panel.config import ARTIFACTS, SETTINGS
from panel.contradictions import ContradictionDetector
from panel.llm import LLM
from panel.memory import Claim, Memory
from panel.orchestrator import InterviewConfig, Orchestrator

# A detected type counts as matching a planted type if they're in the same
# family. "never used Kafka" vs "used Kafka" can legitimately surface as either
# a polarity or an exclusivity conflict depending on which claim pairs first.
FAMILY = {
    "ownership_conflict": {"ownership_conflict"},
    "numeric_conflict": {"numeric_conflict", "timeline_conflict"},
    "timeline_conflict": {"timeline_conflict", "numeric_conflict"},
    "polarity_conflict": {"polarity_conflict", "exclusivity_conflict"},
    "exclusivity_conflict": {"exclusivity_conflict", "polarity_conflict"},
}


def _text_of(x: dict[str, Any]) -> str:
    ev = x.get("evidence", {}) or {}
    a, b = ev.get("claim_a", {}), ev.get("claim_b", {})
    return " ".join([
        str(x.get("ctype", "")), str(x.get("rationale", "")),
        str(ev.get("rule_detail", "")),
        str(a.get("text", "")), str(a.get("span", "")),
        str(b.get("text", "")), str(b.get("span", "")),
    ]).lower()


def match_findings(detected: list[dict[str, Any]],
                   truth: list[dict[str, Any]]) -> dict[str, Any]:
    """Greedily align detections to the answer key."""
    unmatched_truth = list(range(len(truth)))
    tp_pairs, fp = [], []

    for d in detected:
        dtype = d.get("ctype", "")
        dtext = _text_of(d)
        hit = None
        for ti in unmatched_truth:
            g = truth[ti]
            if dtype not in FAMILY.get(g["type"], {g["type"]}):
                continue
            about = str(g.get("about", "")).lower().strip()
            if about and about not in dtext:
                # try the individual words of the project name
                words = [w for w in about.split() if len(w) > 3]
                if not words or not any(w in dtext for w in words):
                    continue
            hit = ti
            break
        if hit is None:
            fp.append({"ctype": dtype, "rationale": d.get("rationale", "")[:160],
                       "severity": d.get("severity")})
        else:
            unmatched_truth.remove(hit)
            tp_pairs.append({"ctype": dtype, "matched_truth": truth[hit]["desc"],
                             "severity": d.get("severity")})

    return {"tp": len(tp_pairs), "fp": len(fp), "fn": len(unmatched_truth),
            "tp_detail": tp_pairs, "fp_detail": fp,
            "fn_detail": [truth[i]["desc"] for i in unmatched_truth]}


# ------------------------------------------------------------------ detector
def eval_detector_case(case: dict[str, Any], llm: LLM) -> dict[str, Any]:
    """Run extraction + detection over the case's scripted answers only."""
    t0 = time.time()
    ex = ClaimExtractor(llm)
    det = ContradictionDetector(llm)

    lex: list[str] = []
    claims: list[Claim] = []

    # Seed resume claims exactly as the orchestrator does, so resume-vs-interview
    # conflicts (the Kafka case) are reachable.
    from panel.claims import ownership_of
    for p in case["resume"].get("projects", []):
        proj = normalize_project(p.get("name", ""))
        if proj:
            lex.append(proj)
        if p.get("role"):
            claims.append(Claim(subject=proj, predicate="role_on", object=p["role"],
                                ownership=ownership_of(p["role"]), source="resume",
                                raw_span=f"Resume: {p.get('name')} - {p['role']}",
                                confidence=0.95, round=0))
        if p.get("team_size"):
            claims.append(Claim(subject=proj, predicate="team_size",
                                value_num=float(p["team_size"]), unit="people",
                                source="resume", round=0,
                                raw_span=f"Resume: team of {p['team_size']}"))
        if p.get("months"):
            claims.append(Claim(subject=proj, predicate="timeline",
                                value_num=float(p["months"]), unit="months",
                                source="resume", round=0,
                                raw_span=f"Resume: {p['months']} months"))
    for s in case["resume"].get("skills", []):
        claims.append(Claim(subject="self", predicate="used_tech",
                            object=s.get("name", ""), source="resume", round=0,
                            raw_span=f"Resume skill: {s.get('name')}"))

    # Feed the scripted answers in plant order (that is the interview order).
    ordered = sorted(case["bank"], key=lambda a: (a.plant_round or 99))
    detected: list[dict[str, Any]] = []
    seen_pairs: list[tuple[str, str]] = []
    for i, ans in enumerate(ordered, start=1):
        rnd = ans.plant_round or i
        new = ex.extract(ans.text, round_=rnd, turn_id=f"t{i}",
                         context_projects=lex)
        for c in new:
            if c.subject not in ("self", "unspecified") and c.subject not in lex:
                lex.append(c.subject)
        claims.extend(new)
        found = det.detect(claims, round_=rnd, skip_pairs=seen_pairs, max_new=4)
        for x in found:
            seen_pairs.append((x.claim_a, x.claim_b))
            d = asdict(x)
            detected.append(d)

    m = match_findings(detected, case["ground_truth"])
    return {"case": case["id"], "name": case["name"],
            "clean": not case["ground_truth"],
            "planted": len(case["ground_truth"]),
            "detected": len(detected),
            "claims_extracted": len(claims),
            "detector_stats": dict(det.stats),
            "elapsed_s": round(time.time() - t0, 1),
            **m}


# ----------------------------------------------------------------- full run
def eval_full_case(case: dict[str, Any], rounds: int) -> dict[str, Any]:
    t0 = time.time()
    llm = LLM()
    mem = Memory()
    orch = Orchestrator(llm=llm, memory=mem, settings=SETTINGS)
    cfg = InterviewConfig(candidate_name=case["name"], role=case["role"],
                          seniority=case["seniority"], rounds=rounds)
    orch.setup(cfg, case["resume"])
    cand = ScriptedCandidate(name=case["name"], persona=case["persona"],
                             resume=case["resume"], bank=case["bank"], llm=llm,
                             code_answer=case.get("code_answer", ""))
    report = orch.run_interview(cand)
    orch.record_approval("approve", "eval-harness", "automated evaluation run")

    detected = report["contradictions"]
    m = match_findings(detected, case["ground_truth"])
    decided = report["verdict"]["decision"]
    guards = report["guardrails"]
    tool = report["tool_usage"]
    return {"case": case["id"], "name": case["name"],
            "clean": not case["ground_truth"],
            "planted": len(case["ground_truth"]), "detected": len(detected),
            "expected_decision": case["expected_decision"],
            "decided": decided,
            "decision_correct": decided == case["expected_decision"],
            "confronted_live": report["contradiction_summary"]["addressed_in_interview"],
            "llm_calls": report["llm_usage"]["calls"],
            "distinct_tools": tool.get("distinct_tools", 0),
            "tool_success_rate": tool.get("success_rate", 0),
            "handled_tool_failures": tool.get("handled_failures", 0),
            "refusals": guards.get("refusal_count", 0),
            "injections": guards.get("injection_count", 0),
            "claims_extracted": len(report["claims"]),
            "session_id": report["session_id"],
            "elapsed_s": round(time.time() - t0, 1),
            **m}


# ------------------------------------------------------------------- metrics
def aggregate(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    clean = [r for r in rows if r["clean"]]
    dirty = [r for r in rows if not r["clean"]]
    clean_fp = sum(r["fp"] for r in clean)
    clean_with_fp = len([r for r in clean if r["fp"] > 0])

    out = {
        "mode": mode,
        "cases_run": len(rows),
        "contradiction_detection": {
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
            "planted_total": sum(r["planted"] for r in rows),
            "detected_total": sum(r["detected"] for r in rows),
        },
        "false_positive_control": {
            "clean_cases": len(clean),
            "clean_cases_with_a_false_positive": clean_with_fp,
            "false_positives_on_clean_cases": clean_fp,
            "clean_case_precision": round(1 - (clean_with_fp / len(clean)), 3) if clean else None,
        },
        "recall_on_cases_with_contradictions": {
            "cases": len(dirty),
            "planted": sum(r["planted"] for r in dirty),
            "found": sum(r["tp"] for r in dirty),
            "recall": round(sum(r["tp"] for r in dirty)
                            / max(sum(r["planted"] for r in dirty), 1), 3),
        },
        "latency": {
            "total_s": round(sum(r["elapsed_s"] for r in rows), 1),
            "mean_per_case_s": round(sum(r["elapsed_s"] for r in rows) / max(len(rows), 1), 1),
        },
    }
    if mode == "full":
        correct = len([r for r in rows if r.get("decision_correct")])
        out["decision_accuracy"] = {
            "correct": correct, "total": len(rows),
            "accuracy": round(correct / max(len(rows), 1), 3),
            "errors": [{"case": r["case"], "expected": r["expected_decision"],
                        "got": r["decided"]} for r in rows
                       if not r.get("decision_correct")],
        }
        out["operational"] = {
            "mean_llm_calls_per_interview": round(
                sum(r["llm_calls"] for r in rows) / max(len(rows), 1), 1),
            "min_distinct_tools_used": min((r["distinct_tools"] for r in rows), default=0),
            "mean_tool_success_rate": round(
                sum(r["tool_success_rate"] for r in rows) / max(len(rows), 1), 3),
            "total_handled_tool_failures": sum(r["handled_tool_failures"] for r in rows),
            "total_injections_neutralized": sum(r["injections"] for r in rows),
            "total_prohibited_questions_blocked": sum(r["refusals"] for r in rows),
            "contradictions_confronted_live": sum(r["confronted_live"] for r in rows),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["detector", "full"], default="detector")
    ap.add_argument("--cases", default="", help="comma separated case ids")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    selected = CASES
    if args.cases:
        want = {c.strip().upper() for c in args.cases.split(",")}
        selected = [c for c in CASES
                    if c["id"].upper() in want or c["id"].split("-")[0].upper() in want]

    print(f"\n{'='*92}")
    print(f"  EVALUATION · mode={args.mode} · {len(selected)} cases")
    print(f"{'='*92}")
    print(json.dumps(stats(), indent=2))
    print()

    t0 = time.time()
    rows: list[dict[str, Any]] = []

    if args.mode == "detector":
        llm = LLM()
        print(f"  model: {llm.describe()}\n")
        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futs = {pool.submit(eval_detector_case, c, llm): c for c in selected}
                for f in futs:
                    pass
                for f, c in futs.items():
                    try:
                        r = f.result()
                    except Exception as e:  # noqa: BLE001
                        r = {"case": c["id"], "name": c["name"],
                             "clean": not c["ground_truth"],
                             "planted": len(c["ground_truth"]), "detected": 0,
                             "claims_extracted": 0, "tp": 0, "fp": 0,
                             "fn": len(c["ground_truth"]), "tp_detail": [],
                             "fp_detail": [], "fn_detail": ["harness error"],
                             "detector_stats": {}, "elapsed_s": 0.0,
                             "error": str(e)[:200]}
                    rows.append(r)
                    _print_row(r)
        else:
            for c in selected:
                r = eval_detector_case(c, llm)
                rows.append(r)
                _print_row(r)
    else:
        for c in selected:
            try:
                r = eval_full_case(c, args.rounds)
            except Exception as e:  # noqa: BLE001
                r = {"case": c["id"], "name": c["name"],
                     "clean": not c["ground_truth"],
                     "planted": len(c["ground_truth"]), "detected": 0,
                     "expected_decision": c["expected_decision"], "decided": "ERROR",
                     "decision_correct": False, "confronted_live": 0, "llm_calls": 0,
                     "distinct_tools": 0, "tool_success_rate": 0,
                     "handled_tool_failures": 0, "refusals": 0, "injections": 0,
                     "claims_extracted": 0, "session_id": "", "tp": 0, "fp": 0,
                     "fn": len(c["ground_truth"]), "tp_detail": [], "fp_detail": [],
                     "fn_detail": ["harness error"], "elapsed_s": 0.0,
                     "error": str(e)[:300]}
            rows.append(r)
            _print_row(r)

    rows.sort(key=lambda r: r["case"])
    summary = aggregate(rows, args.mode)
    summary["wall_clock_s"] = round(time.time() - t0, 1)
    summary["case_results"] = rows
    summary["case_mix"] = stats()

    print(f"\n{'='*92}")
    print(f"  RESULTS")
    print(f"{'='*92}")
    cd = summary["contradiction_detection"]
    print(f"  contradiction detection")
    print(f"    precision {cd['precision']}   recall {cd['recall']}   F1 {cd['f1']}")
    print(f"    tp={cd['true_positives']}  fp={cd['false_positives']}  fn={cd['false_negatives']}"
          f"   (planted {cd['planted_total']}, detected {cd['detected_total']})")
    fpc = summary["false_positive_control"]
    print(f"  false-positive control on clean candidates")
    print(f"    {fpc['clean_cases'] - fpc['clean_cases_with_a_false_positive']}"
          f"/{fpc['clean_cases']} clean cases kept clean "
          f"(clean-case precision {fpc['clean_case_precision']})")
    if "decision_accuracy" in summary:
        da = summary["decision_accuracy"]
        print(f"  hire/no-hire decision accuracy: {da['correct']}/{da['total']} = {da['accuracy']}")
        for e in da["errors"]:
            print(f"    ✗ {e['case']}: expected {e['expected']}, got {e['got']}")
        print(f"  operational: {json.dumps(summary['operational'], indent=4)}")
    print(f"  wall clock: {summary['wall_clock_s']}s")

    out = args.out or str(ARTIFACTS / f"eval_{args.mode}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  full report: {out}\n")
    return 0


def _print_row(r: dict[str, Any]):
    flag = "CLEAN" if r["clean"] else "     "
    status = "ok "
    if r["fp"] or r["fn"]:
        status = "!! "
    extra = ""
    if "decided" in r:
        extra = f"  decision={r['decided']}({'✓' if r['decision_correct'] else '✗'})"
    print(f"  {status}{r['case']:28} {flag}  planted={r['planted']} detected={r['detected']} "
          f"tp={r['tp']} fp={r['fp']} fn={r['fn']}  claims={r['claims_extracted']}"
          f"  {r['elapsed_s']}s{extra}")
    for d in r.get("fp_detail", [])[:3]:
        print(f"      FP: {d['ctype']} — {d['rationale'][:110]}")
    for d in r.get("fn_detail", [])[:3]:
        print(f"      FN: {d[:110]}")
    if r.get("error"):
        print(f"      ERROR: {r['error']}")


if __name__ == "__main__":
    raise SystemExit(main())
