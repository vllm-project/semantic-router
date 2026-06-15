#!/usr/bin/env python3
"""Offline confidence threshold verification using OfflineAnalyzer.

Loads MMLU-Pro 7B/72B results, computes per-category optimal thresholds
using the severity-weighted framework, and classifies into ESCALATE/SELECTIVE/AVOID.
"""

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tuning import OfflineAnalyzer
from tuning.scenarios.confidence import (
    normalize_logprob,
    question_severity,
)

DATA_DIR = Path("/data/vsr-loop-paper/causal-dsl-tuning/eval/data/calibration_25")


@dataclass
class Question:
    qid: str
    category: str
    confidence: float
    correct_small: bool
    correct_large: bool
    quadrant: str


def load_questions() -> list[Question]:
    small = json.loads((DATA_DIR / "small_results.json").read_text())
    large = json.loads((DATA_DIR / "large_results.json").read_text())
    large_map = {r["question_id"]: r for r in large}

    questions = []
    for s in small:
        qid = s["question_id"]
        lg = large_map.get(qid)
        if lg is None:
            continue
        conf = normalize_logprob(s["avg_logprob"])
        c_small = s["predicted"] == s["correct_answer"]
        c_large = lg["predicted"] == lg["correct_answer"]

        if not c_small and c_large:
            quadrant = "uplift"
        elif c_small and not c_large:
            quadrant = "regression"
        elif c_small and c_large:
            quadrant = "both_correct"
        else:
            quadrant = "both_wrong"

        questions.append(
            Question(
                qid=qid,
                category=s["category"],
                confidence=conf,
                correct_small=c_small,
                correct_large=c_large,
                quadrant=quadrant,
            )
        )
    return questions


def main():
    questions = load_questions()
    print(
        f"Loaded {len(questions)} questions across {len({q.category for q in questions})} categories"
    )

    small_acc = sum(1 for q in questions if q.correct_small) / len(questions)
    large_acc = sum(1 for q in questions if q.correct_large) / len(questions)
    print(f"7B baseline: {100*small_acc:.1f}%  |  72B baseline: {100*large_acc:.1f}%")

    by_cat = defaultdict(list)
    for q in questions:
        by_cat[q.category].append(q)

    analyzer = OfflineAnalyzer(severity_fn=question_severity)

    results = {}
    strategy_counts = defaultdict(int)
    total_correct = 0
    total_escalated = 0
    total_questions = 0

    print(
        f"\n{'Category':20s} {'N':>4s} {'Strategy':>10s} {'Thr':>7s} {'Acc%':>6s} {'Esc%':>6s} {'Net':>5s}"
    )
    print("-" * 70)

    for cat in sorted(by_cat.keys()):
        items = by_cat[cat]
        result = analyzer.find_optimal_threshold(
            items=items,
            confidence_fn=lambda q: q.confidence,
            quadrant_fn=lambda q: q.quadrant,
            correct_small_fn=lambda q: q.correct_small,
            correct_large_fn=lambda q: q.correct_large,
            id_fn=lambda q: q.qid,
        )
        results[cat] = result
        strategy_counts[result["strategy"]] += 1
        total_correct += result["best_accuracy"]
        total_escalated += round(result["escalation_rate"] * len(items) / 100)
        total_questions += len(items)

        print(
            f"{cat:20s} {len(items):4d} {result['strategy']:>10s} "
            f"{result['optimal_threshold']:7.4f} "
            f"{result['best_accuracy_pct']:5.1f}% "
            f"{result['escalation_rate']:5.1f}% "
            f"{result['net_uplift']:+4d}"
        )

    overall_acc = 100 * total_correct / total_questions
    overall_esc = 100 * total_escalated / total_questions

    print("-" * 70)
    print(
        f"{'OVERALL':20s} {total_questions:4d} {'':10s} {'':>7s} "
        f"{overall_acc:5.1f}% {overall_esc:5.1f}%"
    )
    print(
        f"\nStrategy distribution: "
        f"ESCALATE={strategy_counts['ESCALATE']}, "
        f"SELECTIVE={strategy_counts['SELECTIVE']}, "
        f"AVOID={strategy_counts['AVOID']}"
    )

    output = {
        "scenario": "confidence_threshold_tuning",
        "method": "analytical_trace_diagnosis",
        "pipeline": "offline_per_category_optimization",
        "num_questions": total_questions,
        "num_categories": len(by_cat),
        "overall_accuracy_pct": round(overall_acc, 2),
        "overall_escalation_pct": round(overall_esc, 2),
        "strategy_counts": dict(strategy_counts),
        "baselines": {
            "always_7b": round(100 * small_acc, 2),
            "always_72b": round(100 * large_acc, 2),
        },
        "per_category": {
            cat: {
                "n": r["n_items"],
                "strategy": r["strategy"],
                "threshold": r["optimal_threshold"],
                "accuracy_pct": r["best_accuracy_pct"],
                "escalation_pct": r["escalation_rate"],
                "net_uplift": r["net_uplift"],
                "baseline_pct": r["baseline_accuracy_pct"],
            }
            for cat, r in results.items()
        },
        "router_queries": 0,
    }

    out_path = Path(__file__).parent / "confidence_verification.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
