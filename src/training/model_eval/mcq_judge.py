#!/usr/bin/env python3
"""Generic MCQ judge: extract answer letter, exact match, produce analysis.

Reads records.jsonl (model responses) and tasks.jsonl (ground truth),
extracts answer letters using standard MCQ patterns, compares to expected
answers, and outputs:
  - judged.jsonl: per-question result (task_id, is_correct, tokens, ...)
  - summary.json: accuracy by dataset/language/category + token/latency stats
  - analysis.json: per-category accuracy in SR's result_to_config.py format

Pure stdlib — no external dependencies required.

Answer extraction protocol:
  1. Last "Answer: X" / "answer: X" / "答案：X" occurrence in the text;
  2. Fallback: last standalone "(X)" or "X" line;
  3. Failure -> no_answer (counted as wrong, tracked separately).

Usage:
    python mcq_judge.py \\
        --records results/dsv4-flash/accept_office/records.jsonl \\
        --tasks tasks/accept_office.jsonl \\
        --out results/dsv4-flash/accept_office/judged.jsonl \\
        --summary results/dsv4-flash/accept_office/summary.json \\
        [--analysis results/dsv4-flash/accept_office/analysis.json]
"""
import argparse
import json
import re
import statistics
from collections import defaultdict

ANSWER_RE = re.compile(r"(?:[Aa]nswer|ANSWER|答案)\s*[:：]\s*\(?([A-J])\)?")
STANDALONE_RE = re.compile(r"^\(?([A-J])\)?$")


def extract_letter(text):
    """Extract the answer letter from model response text."""
    if not text:
        return None
    matches = ANSWER_RE.findall(text)
    if matches:
        return matches[-1].upper()
    for line in reversed(text.strip().splitlines()):
        s = line.strip()
        if STANDALONE_RE.match(s):
            return s.strip("()").upper()
    return None


def pct(a, b):
    return round(100.0 * a / b, 2) if b else None


def stats_block(vals):
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": round(statistics.fmean(vals), 2),
        "p50": round(statistics.median(vals), 2),
        "max": round(max(vals), 2),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", required=True, help="Model responses JSONL")
    ap.add_argument("--tasks", required=True, help="Ground truth tasks JSONL")
    ap.add_argument("--out", required=True, help="Output judged JSONL")
    ap.add_argument("--summary", default=None, help="Output summary JSON")
    ap.add_argument(
        "--analysis",
        default=None,
        help="Output analysis.json (SR result_to_config.py format)",
    )
    args = ap.parse_args()

    with open(args.tasks, encoding="utf-8") as f:
        tasks = {json.loads(l)["task_id"]: json.loads(l) for l in f}
    with open(args.records, encoding="utf-8") as f:
        records = [json.loads(l) for l in f]

    judged = []
    for rec in records:
        task = tasks.get(rec["task_id"])
        if task is None:
            print(f"[judge] WARN unknown task_id {rec['task_id']}, skipped")
            continue
        letter = extract_letter(rec.get("completion") or "")
        expected = task["answer"]
        judged.append(
            {
                "task_id": rec["task_id"],
                "dataset": rec.get("dataset"),
                "category": rec.get("category"),
                "language": rec.get("language"),
                "answer_expected": expected,
                "answer_extracted": letter,
                "is_correct": letter == expected,
                "no_answer": letter is None,
                "gen_error": rec.get("error"),
                "finish_reason": rec.get("finish_reason"),
                "completion_tokens": rec.get("completion_tokens"),
                "latency_s": rec.get("latency_s"),
                "completion": rec.get("completion"),
            }
        )

    with open(args.out, "w", encoding="utf-8") as f:
        for j in judged:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")

    # --- summary.json (our format) ---
    def acc_of(rows):
        return pct(sum(1 for r in rows if r["is_correct"]), len(rows))

    summary = {
        "n_total": len(judged),
        "accuracy": acc_of(judged),
        "no_answer_count": sum(1 for r in judged if r["no_answer"]),
        "gen_error_count": sum(1 for r in judged if r["gen_error"]),
        "by_dataset": {},
        "by_language": {},
        "by_category": {},
        "tokens_completion": stats_block(
            [
                r["completion_tokens"]
                for r in judged
                if r["completion_tokens"] is not None
            ]
        ),
        "latency_s": stats_block(
            [r["latency_s"] for r in judged if r["latency_s"] is not None]
        ),
    }
    for key in ("dataset", "language", "category"):
        groups = {}
        for r in judged:
            groups.setdefault(r.get(key), []).append(r)
        for gname, rows in sorted(groups.items(), key=lambda x: str(x[0])):
            block = {
                "n": len(rows),
                "accuracy": acc_of(rows),
                "no_answer": sum(1 for r in rows if r["no_answer"]),
            }
            if key == "dataset":
                block["truncated"] = sum(
                    1 for r in rows if r["finish_reason"] == "length"
                )
            summary[f"by_{key}"][gname] = block

    out_summary = args.summary or args.out.replace(".jsonl", "_summary.json")
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # --- analysis.json (SR result_to_config.py format) ---
    if args.analysis:
        cat_acc = {}
        cat_total = defaultdict(int)
        cat_correct = defaultdict(int)
        for r in judged:
            cat = r.get("category", "unknown")
            cat_total[cat] += 1
            if r["is_correct"]:
                cat_correct[cat] += 1
        for cat in sorted(cat_total.keys()):
            acc = cat_correct[cat] / cat_total[cat] if cat_total[cat] > 0 else 0.0
            cat_acc[cat] = round(acc, 6)
        total = len(judged)
        correct = sum(cat_correct.values())
        analysis = {
            "category_accuracy": cat_acc,
            "overall_accuracy": round(correct / total, 6) if total > 0 else 0.0,
            "total_questions": total,
            "total_correct": correct,
        }
        with open(args.analysis, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)
        print(f"[judge] analysis.json written to {args.analysis}")

    print(
        f"[judge] DONE {len(judged)} judged, "
        f"acc={summary['accuracy']}%, "
        f"no_answer={summary['no_answer_count']}"
    )


if __name__ == "__main__":
    main()
