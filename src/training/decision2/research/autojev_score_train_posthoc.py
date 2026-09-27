"""CPU-only descriptive slices of the frozen private Score teacher artifact.

This never calls a model. Slices are explicitly post hoc and cannot modify the
pre-registered 516-row result or select a student checkpoint.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from research.autojev_score_train_audit import (
    TRAIN_SHA256,
    aggregate,
    roster_sha256,
    score_rows,
)
from research.eikos_teacher_train_pilot import write_once
from training.model.data import file_sha256, load_partition

AGGREGATE_SHA256 = "ff703bd206ae9958bd4afc6afbfc90c583cb85ef4a8263ddc6ba7ea7156f59ec"
ARTIFACT_SHA256 = "072cd519657caaa883eea1f5077789e5bacbf85f8ee20ab44cd562acc317701b"


def _aligned(train: list[dict[str, Any]], distributions: list[dict[str, Any]]) -> None:
    if len(train) != len(distributions):
        raise ValueError("Teacher artifact lost TRAIN Score rows")
    for row, predicted in zip(train, distributions):
        expected = {
            "id": row["id"],
            "input_sha256": row["input_sha256"],
            "group_id": row["group_id"],
            "source": row["source"],
            "family": row["family"],
            "level_count": len(row["options"]),
        }
        if any(predicted.get(key) != value for key, value in expected.items()):
            raise ValueError("Teacher artifact identity or order differs")


def slice_metrics(
    train: list[dict[str, Any]], distributions: list[dict[str, Any]]
) -> dict[str, Any]:
    _aligned(train, distributions)
    result = {}
    for language in ("en", "zh", "all"):
        for levels in ("3", "4-8", "all"):
            selected = [
                (row, predicted)
                for row, predicted in zip(train, distributions)
                if (language == "all" or row["language"] == language)
                and (
                    levels == "all"
                    or (
                        len(row["options"]) == 3
                        if levels == "3"
                        else len(row["options"]) >= 4
                    )
                )
            ]
            if not selected:
                continue
            predictions = iter(item[1] for item in selected)

            def decide(
                state: Any,
                question: dict[str, Any],
                teacher_rows: Iterator[dict[str, Any]] = predictions,
            ) -> tuple[dict[str, Any], int]:
                return {
                    "type": "score",
                    "probabilities": next(teacher_rows)["probabilities"],
                }, 0

            stats, _ = aggregate([item[0] for item in selected], decide)
            result[f"{language}|{levels}"] = stats["overall"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--aggregate", required=True, type=Path)
    parser.add_argument("--distributions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if (
        file_sha256(args.train) != TRAIN_SHA256
        or file_sha256(args.aggregate) != AGGREGATE_SHA256
        or file_sha256(args.distributions) != ARTIFACT_SHA256
    ):
        raise ValueError("Frozen TRAIN or teacher result bytes differ")
    rows = score_rows(load_partition(args.train, "train"))
    receipt = json.loads(args.aggregate.read_text())
    artifact = json.loads(args.distributions.read_text())
    identity = roster_sha256(rows)
    if (
        receipt["roster_sha256"] != identity
        or artifact["roster_sha256"] != identity
        or receipt["private_distribution_sha256"] != ARTIFACT_SHA256
        or receipt["metrics"]["overall"]["valid"] != len(rows)
    ):
        raise ValueError("Teacher result is not the complete frozen Score roster")
    payload = {
        "schema": "decision2-autojev-score-train-posthoc/1",
        "provenance": "descriptive-posthoc-not-pre-registered",
        "train_sha256": TRAIN_SHA256,
        "aggregate_sha256": AGGREGATE_SHA256,
        "distribution_sha256": ARTIFACT_SHA256,
        "roster_sha256": identity,
        "slices": slice_metrics(rows, artifact["rows"]),
    }
    write_once(args.output, payload)
    print(
        json.dumps({k: v for k, v in payload.items() if k != "slices"}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
