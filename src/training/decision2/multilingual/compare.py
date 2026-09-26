"""Compare two native models on matched multilingual validation source IDs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from multilingual.audit import sha256
from multilingual.parallel_score import _exact_mcnemar_p, score
from multilingual.score import _interpret, _load_jsonl


def compare(panel: Path, baseline: Path, candidate: Path) -> dict:
    baseline_report = score(panel, baseline)
    candidate_report = score(panel, candidate)
    targets = {row["id"]: row for row in _load_jsonl(panel / "targets.jsonl")}
    predicted = [
        {row["id"]: row for row in _load_jsonl(path)} for path in (baseline, candidate)
    ]
    by_group = defaultdict(list)
    for row_id, target in targets.items():
        correct = []
        for model in predicted:
            semantic, valid = _interpret(model[row_id]["answers"]["decision"], target)
            correct.append(int(valid and semantic == target["semantic_gold"]))
        by_group[(target["corpus"], target["language"])].append(tuple(correct))
    summary = {}
    for (corpus, language), pairs in sorted(by_group.items()):
        baseline_only = sum(old and not new for old, new in pairs)
        candidate_only = sum(new and not old for old, new in pairs)
        summary.setdefault(corpus, {})[language] = {
            "independent_base_cases": len(pairs),
            "baseline_correct": sum(old for old, _ in pairs),
            "candidate_correct": sum(new for _, new in pairs),
            "candidate_minus_baseline": candidate_only - baseline_only,
            "baseline_only_correct": baseline_only,
            "candidate_only_correct": candidate_only,
            "paired_exact_mcnemar_p": _exact_mcnemar_p(baseline_only, candidate_only),
        }
    return {
        "schema_version": "decision2-multilingual-model-paired-compare/1",
        "scope": "public parallel validation development only; no sealed final",
        "panel_manifest_sha256": baseline_report["manifest_sha256"],
        "baseline_model": baseline_report["model"],
        "candidate_model": candidate_report["model"],
        "baseline_predictions_sha256": sha256(baseline),
        "candidate_predictions_sha256": sha256(candidate),
        "by_corpus_language": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = compare(args.panel, args.baseline, args.candidate)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report_sha256": sha256(args.output),
                "by_corpus_language": report["by_corpus_language"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
