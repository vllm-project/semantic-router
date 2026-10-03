from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from router_calibration_http import ensure_success, http_json, normalize_router_url

PREVIEW_PATH = "/api/v1/routing/preview"
NO_MATCH = "(none)"


def load_labeled_set(path: Path) -> list[dict[str, str]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} must be a non-empty JSON array")
    for index, record in enumerate(records):
        if not (
            isinstance(record, dict)
            and isinstance(record.get("text"), str)
            and isinstance(record.get("true_label"), str)
        ):
            raise ValueError(f"{path}[{index}] needs string text and true_label")
    return records


def predicted_label(response: dict[str, Any], observation: str) -> str:
    """Return the one label the preview matched, or NO_MATCH when none did."""
    matched_signals = (response.get("decision_result") or {}).get(
        "matched_signals"
    ) or {}
    labels = matched_signals.get(observation) or []
    if len(labels) > 1:
        raise ValueError(f"preview matched several {observation} labels: {labels}")
    return labels[0] if labels else NO_MATCH


def score(pairs: list[tuple[str, str]], labels: list[str]) -> dict[str, Any]:
    """Per-label precision and recall, and a confusion matrix keyed true -> predicted."""
    columns = labels + sorted({predicted for _, predicted in pairs} - set(labels))
    confusion = {label: dict.fromkeys(columns, 0) for label in labels}
    for true_label, predicted in pairs:
        confusion[true_label][predicted] += 1

    per_label = {}
    for label in labels:
        hits = confusion[label][label]
        predicted_total = sum(row[label] for row in confusion.values())
        support = sum(confusion[label].values())
        precision = hits / predicted_total if predicted_total else 0.0
        recall = hits / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    correct = sum(1 for true_label, predicted in pairs if true_label == predicted)
    return {
        "accuracy": correct / len(pairs),
        "macro_f1": sum(item["f1"] for item in per_label.values()) / len(labels),
        "per_label": per_label,
        "columns": columns,
        "confusion": confusion,
    }


def evaluate(
    router_url: str, records: list[dict[str, str]], observation: str, timeout: float
) -> list[dict[str, str]]:
    url = normalize_router_url(router_url) + PREVIEW_PATH
    results = []
    for record in records:
        payload = {"messages": [{"role": "user", "content": record["text"]}]}
        status, response = http_json(
            "POST", url, payload=payload, timeout_seconds=timeout
        )
        response = ensure_success(status, response, "routing preview")
        results.append(
            {
                "text": record["text"],
                "true_label": record["true_label"],
                "predicted": predicted_label(response, observation),
                "decision": (response.get("decision_result") or {}).get(
                    "decision_name", ""
                ),
            }
        )
    return results


def render_report(report: dict[str, Any], misses: list[dict[str, str]]) -> str:
    lines = [
        f"accuracy {report['accuracy']:.3f}, macro F1 {report['macro_f1']:.3f}",
        "",
        "| label | precision | recall | F1 | support |",
        "|---|---|---|---|---|",
    ]
    for label, item in report["per_label"].items():
        lines.append(
            f"| {label} | {item['precision']:.2f} | {item['recall']:.2f} "
            f"| {item['f1']:.2f} | {item['support']} |"
        )
    columns = report["columns"]
    lines += [
        "",
        "Confusion matrix (rows are true labels, columns are predictions):",
        "",
        "| true \\ predicted | " + " | ".join(columns) + " |",
        "|---" * (len(columns) + 1) + "|",
    ]
    for label, row in report["confusion"].items():
        lines.append(
            f"| {label} | " + " | ".join(str(row[column]) for column in columns) + " |"
        )
    lines += ["", f"Misses ({len(misses)}):"]
    lines += [
        f"- want {miss['true_label']}, got {miss['predicted']}: "
        f"{json.dumps(miss['text'], ensure_ascii=False)}"
        for miss in misses
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Score a labeled prompt set through the routing preview API and report "
            "per-label precision, recall and a confusion matrix."
        )
    )
    parser.add_argument("--router-url", default="http://localhost:8080")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--observation",
        default="action",
        help="decision_result.matched_signals field that holds the predicted label",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)

    records = load_labeled_set(args.dataset)
    results = evaluate(args.router_url, records, args.observation, args.timeout)
    labels = list(dict.fromkeys(record["true_label"] for record in records))
    report = score(
        [(result["true_label"], result["predicted"]) for result in results], labels
    )
    misses = [
        result for result in results if result["predicted"] != result["true_label"]
    ]
    print(render_report(report, misses))
    if args.json_output:
        args.json_output.write_text(
            json.dumps({**report, "results": results}, indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
