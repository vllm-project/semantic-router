"""Aggregate paired Kai development errors without exporting private examples.

The report contains only counts, label-level metrics, and input hashes. Input
or prediction text, IDs, and individual answers never leave the analysis host.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from benchmark.score import evaluate_answer
from transfer.score import evaluate as evaluate_css, read_jsonl, macro_f1
from transfer.build import sha_file


def _paths(values: list[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for item in values:
        name, sep, filename = item.partition("=")
        if not sep or not name or not filename or name in paths:
            raise ValueError("Use distinct NAME=PATH prediction arguments")
        paths[name] = Path(filename)
    return paths


def _label_metrics(
    golds: list[str], guesses: list[str | None], labels: list[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in labels:
        support = sum(g == label for g in golds)
        predicted = sum(p == label for p in guesses)
        tp = sum(g == p == label for g, p in zip(golds, guesses))
        denominator = support + predicted
        result[label] = {
            "support": support,
            "predicted": predicted,
            "true_positive": tp,
            "precision": tp / predicted if predicted else 0.0,
            "recall": tp / support if support else 0.0,
            "f1": 2 * tp / denominator if denominator else 0.0,
            "confusion": dict(
                sorted(
                    Counter(
                        p if p is not None else "<invalid>"
                        for g, p in zip(golds, guesses)
                        if g == label
                    ).items()
                )
            ),
        }
    return result


def _paired(
    gold: dict[str, dict[str, Any]], results: dict[str, dict[str, bool]], source: str
) -> dict[str, Any]:
    pairs = {}
    for name in results:
        if name == source:
            continue
        by_group: dict[str, Counter[str]] = defaultdict(Counter)
        for item_id, row in gold.items():
            group = row.get("task", row.get("family", "all"))
            a, b = results[source][item_id], results[name][item_id]
            by_group[group][
                (
                    "both_right"
                    if a and b
                    else "source_only" if a else "candidate_only" if b else "both_wrong"
                )
            ] += 1
        pairs[name] = {
            group: dict(sorted(counts.items()))
            for group, counts in sorted(by_group.items())
        }
    return pairs


def analyze_css(gold_path: Path, files: dict[str, Path], source: str) -> dict[str, Any]:
    gold = read_jsonl(gold_path)
    predictions = {name: read_jsonl(path) for name, path in files.items()}
    if any(set(rows) != set(gold) for rows in predictions.values()):
        raise ValueError("CSS predictions must cover the same frozen gold IDs")
    for name, rows in predictions.items():
        for item_id, prediction in rows.items():
            if prediction.get("source_input_sha256") != gold[item_id]["input_sha256"]:
                raise ValueError(f"{name} CSS input digest differs")
    output: dict[str, Any] = {
        "gold_sha256": sha_file(gold_path),
        "predictions_sha256": {name: sha_file(path) for name, path in files.items()},
        "models": {},
    }
    correctness: dict[str, dict[str, bool]] = {}
    for name, rows in predictions.items():
        correctness[name] = {}
        by_task: dict[
            str, list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]
        ] = defaultdict(list)
        for item_id, truth in gold.items():
            prediction = rows[item_id]
            result = evaluate_css(truth, prediction)
            correctness[name][item_id] = bool(result.get("correct", False))
            by_task[truth["task"]].append((truth, prediction, result))
        tasks: dict[str, Any] = {}
        for task, triples in sorted(by_task.items()):
            labels = triples[0][0]["labels"]
            golds = [truth["gold"] for truth, _, _ in triples]
            guesses = [result["choice"] for _, _, result in triples]
            tasks[task] = {
                "n": len(triples),
                "correct": sum(g == p for g, p in zip(golds, guesses)),
                "macro_f1": macro_f1(golds, guesses, labels),
                "invalid": sum(not result["valid"] for _, _, result in triples),
                "overflow_by_gold_label": dict(
                    sorted(
                        Counter(
                            truth["gold"]
                            for truth, prediction, _ in triples
                            if prediction.get("invalid_reason") == "context_overflow"
                        ).items()
                    )
                ),
                "labels": _label_metrics(golds, guesses, labels),
            }
        output["models"][name] = tasks
    output["paired_vs_source"] = _paired(gold, correctness, source)
    return output


def analyze_dev(gold_path: Path, files: dict[str, Path], source: str) -> dict[str, Any]:
    gold = read_jsonl(gold_path)
    predictions = {name: read_jsonl(path) for name, path in files.items()}
    if any(set(rows) != set(gold) for rows in predictions.values()):
        raise ValueError("DEV predictions must cover the same frozen gold IDs")
    output: dict[str, Any] = {
        "gold_sha256": sha_file(gold_path),
        "predictions_sha256": {name: sha_file(path) for name, path in files.items()},
        "models": {},
    }
    correctness: dict[str, dict[str, bool]] = {}
    for name, rows in predictions.items():
        correctness[name] = {}
        by_type: dict[str, Counter[str]] = defaultdict(Counter)
        by_family: dict[str, Counter[str]] = defaultdict(Counter)
        point_confusions: dict[str, Counter[str]] = defaultdict(Counter)
        for item_id, truth in gold.items():
            prediction = rows[item_id]
            if (
                prediction.get("source_input_sha256")
                != truth["provenance"]["payload_sha256"]
            ):
                raise ValueError(f"{name} DEV input digest differs")
            key = next(iter(truth["questions"]))
            qtype = truth["questions"][key]["type"]
            result = evaluate_answer(
                truth["questions"][key],
                truth["gold"][key],
                prediction.get("answers", {}).get(key),
            )
            correct = (
                bool(result.get("correct", False))
                if result["status"] == "ok"
                else False
            )
            correctness[name][item_id] = correct
            for dest in (by_type[qtype], by_family[truth["family"]]):
                dest["n"] += 1
                dest["correct"] += correct
                dest["invalid"] += result["status"] != "ok"
            if qtype in {"noul", "score"}:
                target = str(truth["gold"][key]["value"])
                point = (
                    str(result.get("point"))
                    if result["status"] == "ok"
                    else "<invalid>"
                )
                point_confusions[qtype][target + "->" + point] += 1
        output["models"][name] = {
            "by_type": {k: dict(v) for k, v in sorted(by_type.items())},
            "by_family": {k: dict(v) for k, v in sorted(by_family.items())},
            "noul_score_confusions": {
                k: dict(sorted(v.items())) for k, v in sorted(point_confusions.items())
            },
        }
    output["paired_vs_source"] = _paired(gold, correctness, source)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--css-gold", type=Path, required=True)
    parser.add_argument("--dev-gold", type=Path, required=True)
    parser.add_argument("--css-pred", action="append", default=[], required=True)
    parser.add_argument("--dev-pred", action="append", default=[], required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    css, dev = _paths(args.css_pred), _paths(args.dev_pred)
    if set(css) != set(dev) or args.source not in css or args.output.exists():
        raise ValueError(
            "Both panels need identical model names, a source, and new output"
        )
    report = {
        "schema_version": "kai06b-development-aggregate-diagnostics/1",
        "css": analyze_css(args.css_gold, css, args.source),
        "dev": analyze_dev(args.dev_gold, dev, args.source),
        "privacy": "aggregate counts only; no prompt text, IDs, or row predictions",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "css_rows": sum(
                    row["n"] for row in report["css"]["models"][args.source].values()
                ),
                "dev_rows": sum(
                    row["n"]
                    for row in report["dev"]["models"][args.source]["by_type"].values()
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
