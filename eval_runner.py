"""
src/eval_runner.py
Runs the full evaluation pipeline against the golden dataset.

Exit codes:
  0 – all checks passed
  1 – one or more quality gates failed (blocks the merge)

Usage:
  python src/eval_runner.py
  python src/eval_runner.py --dataset evals/golden_dataset.json --output evals/results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from pipeline import query_llm
from metrics import answer_relevancy, is_hallucination, keyword_hit_rate, aggregate

# ── Quality gates (edit these thresholds) ────────────────────────────────────
GATE_HALLUCINATION_RATE_MAX_PCT = float(os.getenv("GATE_HALLUCINATION_MAX", "20"))  # %
GATE_P95_LATENCY_MAX_MS         = float(os.getenv("GATE_P95_LATENCY_MAX", "5000"))  # ms
GATE_AVG_RELEVANCY_MIN          = float(os.getenv("GATE_RELEVANCY_MIN", "0.4"))


def run_eval(dataset_path: str, output_path: str | None = None) -> tuple[dict, bool]:
    dataset = json.loads(Path(dataset_path).read_text())
    results = []

    print(f"\n{'─'*60}")
    print(f"  LLM Eval Pipeline  |  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'─'*60}")
    print(f"  Dataset : {dataset_path}  ({len(dataset)} questions)")
    print(f"  Model   : {os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')}")
    print(f"{'─'*60}\n")

    for item in dataset:
        print(f"  [{item['id']}] {item['question'][:65]}…")
        result = query_llm(item["question"])

        khr    = keyword_hit_rate(result["answer"], item.get("expected_keywords", []))
        hall   = is_hallucination(result["answer"], item.get("expected_keywords", []))
        relev  = answer_relevancy(result["answer"], item["expected_answer"])

        row = {
            "id":                item["id"],
            "question":          item["question"],
            "expected_answer":   item["expected_answer"],
            "actual_answer":     result["answer"],
            "tags":              item.get("tags", []),
            "keyword_hit_rate":  round(khr, 4),
            "hallucination_flag": hall,
            "answer_relevancy":  relev,
            "latency_ms":        result["latency_ms"],
            "cost_usd":          result["cost_usd"],
            "model":             result["model"],
        }
        results.append(row)

        status = "❌ HALL" if hall else "✅ OK  "
        print(f"         {status}  relevancy={relev:.2f}  latency={result['latency_ms']:.0f}ms  cost=${result['cost_usd']:.5f}")

    agg = aggregate(results)

    print(f"\n{'─'*60}")
    print("  AGGREGATE METRICS")
    print(f"{'─'*60}")
    for k, v in agg.items():
        print(f"  {k:<35} {v}")

    # ── Quality gates ─────────────────────────────────────────────────────────
    gates_passed = True
    failures: list[str] = []

    if agg["hallucination_rate_pct"] > GATE_HALLUCINATION_RATE_MAX_PCT:
        failures.append(
            f"Hallucination rate {agg['hallucination_rate_pct']}% "
            f"> threshold {GATE_HALLUCINATION_RATE_MAX_PCT}%"
        )
        gates_passed = False

    if agg["p95_latency_ms"] > GATE_P95_LATENCY_MAX_MS:
        failures.append(
            f"p95 latency {agg['p95_latency_ms']}ms "
            f"> threshold {GATE_P95_LATENCY_MAX_MS}ms"
        )
        gates_passed = False

    if agg["avg_relevancy"] < GATE_AVG_RELEVANCY_MIN:
        failures.append(
            f"Avg relevancy {agg['avg_relevancy']} "
            f"< threshold {GATE_AVG_RELEVANCY_MIN}"
        )
        gates_passed = False

    print(f"\n{'─'*60}")
    if gates_passed:
        print("  ✅  ALL QUALITY GATES PASSED — safe to merge")
    else:
        print("  ❌  QUALITY GATES FAILED — merge blocked")
        for f in failures:
            print(f"      • {f}")
    print(f"{'─'*60}\n")

    # ── Persist results ───────────────────────────────────────────────────────
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "aggregate": agg,
        "gates_passed": gates_passed,
        "failures": failures,
        "results": results,
    }

    out = output_path or "evals/results.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2))
    print(f"  Results saved → {out}\n")

    return report, gates_passed


def main():
    parser = argparse.ArgumentParser(description="LLM Eval CI/CD Runner")
    parser.add_argument("--dataset", default="evals/golden_dataset.json")
    parser.add_argument("--output",  default="evals/results.json")
    args = parser.parse_args()

    _, passed = run_eval(args.dataset, args.output)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
