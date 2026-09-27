"""Aggregate-only quarantine accounting for a frozen ANLI TRAIN audit."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from training.data.audit_anli_score_diagnostic import (
    ROUND_FILES as DEV_FILES,
)
from training.data.audit_anli_score_diagnostic import (
    TOKENIZER_FILES,
    normalize,
    write_once,
)
from training.data.audit_anli_score_train_source import (
    PROTECTED_MANIFEST_SHA256,
    TrainRow,
    read_dev_inputs,
    read_train,
    sample_whole_groups,
)
from training.model.data import file_sha256


def dev_input_spans(directory: Path) -> set[str]:
    spans = set()
    for filename, expected_rows, expected_sha in DEV_FILES.values():
        path = directory / filename
        if not path.is_file() or file_sha256(path) != expected_sha:
            raise ValueError("Pinned ANLI dev bytes changed")
        table = pq.read_table(path, columns=["premise", "hypothesis"])
        if table.num_rows != expected_rows:
            raise ValueError("ANLI dev count changed")
        for name in ("premise", "hypothesis"):
            spans.update(
                normalized
                for value in table[name].to_pylist()
                if isinstance(value, str)
                for normalized in [normalize(value)]
                if len(normalized) >= 20
            )
    return spans


def quarantine_profile(
    selected: list[TrainRow], source: list[TrainRow], dev_spans: set[str]
) -> dict[str, Any]:
    exact_dev_groups = {
        row.group
        for row in selected
        if normalize(row.premise) in dev_spans or normalize(row.hypothesis) in dev_spans
    }
    labels_by_pair: dict[tuple[str, str], set[int]] = collections.defaultdict(set)
    rows_by_pair = collections.Counter()
    for row in source:
        key = (row.group, normalize(row.hypothesis))
        labels_by_pair[key].add(row.label)
        rows_by_pair[key] += 1
    conflict_groups = {
        group for (group, _), labels in labels_by_pair.items() if len(labels) > 1
    }
    duplicate_groups = {
        group for (group, _), count in rows_by_pair.items() if count > 1
    }
    selected_groups = {row.group for row in selected}
    selected_conflicts = selected_groups & conflict_groups
    selected_duplicates = selected_groups & duplicate_groups
    affected = exact_dev_groups | selected_conflicts
    return {
        "selected_rows": len(selected),
        "selected_groups": len(selected_groups),
        "selected_exact_dev_affected_groups": len(exact_dev_groups),
        "selected_exact_dev_affected_rows": sum(
            row.group in exact_dev_groups for row in selected
        ),
        "selected_source_conflict_groups": len(selected_conflicts),
        "selected_source_conflict_rows": sum(
            row.group in selected_conflicts for row in selected
        ),
        "selected_duplicate_pair_groups": len(selected_duplicates),
        "selected_duplicate_pair_rows": sum(
            row.group in selected_duplicates for row in selected
        ),
        "fixed_sample_after_exact_dev_and_conflict_quarantine_upper_bound": {
            "rows": sum(row.group not in affected for row in selected),
            "groups": len(selected_groups - affected),
        },
        "actual_admitted_rows": 0,
        "reason": "Original-corpus rights and neutral-rubric review remain pending; no resampling or model training.",
    }


def audit(
    train_directory: Path,
    dev_directory: Path,
    tokenizer_directory: Path,
    prior_receipt: Path,
    prior_sha256: str,
) -> dict[str, Any]:
    if not prior_receipt.is_file() or file_sha256(prior_receipt) != prior_sha256:
        raise ValueError("Frozen parent audit receipt changed")
    parent = json.loads(prior_receipt.read_text(encoding="utf-8"))
    if (
        parent.get("schema") != "decision2-anli-score-train-source-screen/1"
        or parent.get("protected_manifest_sha256") != PROTECTED_MANIFEST_SHA256
        or not parent["gates"]["whole_source_protected_exact_zero"]
        or any(
            parent["selected_bounded_overlap_by_role"][role]["counts"][name]
            for role in parent["selected_bounded_overlap_by_role"]
            for name in ("exact_raw", "exact_normalized", "near", "same_row_ids")
        )
        or parent["selected_open_dev_bounded_overlap"]["counts"]["near"]
    ):
        raise ValueError("Parent audit is not eligible for bounded supplement")
    source, hashes = read_train(train_directory)
    if hashes != parent["source_sha256"]:
        raise ValueError("Source identity differs from parent audit")
    for name, expected in TOKENIZER_FILES.items():
        path = tokenizer_directory / name
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError("Pinned tokenizer changed")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_directory, local_files_only=True, trust_remote_code=False
    )
    selected, selection, _ = sample_whole_groups(
        source, read_dev_inputs(dev_directory), set(), tokenizer
    )
    for field in ("selected_rows", "selected_groups", "selected_native_tokens"):
        if selection[field] != parent["selection"][field]:
            raise ValueError("Frozen selection identity changed")
    result = quarantine_profile(selected, source, dev_input_spans(dev_directory))
    if (
        not result["selected_exact_dev_affected_groups"]
        or not parent["selected_open_dev_bounded_overlap"]["counts"]["exact_normalized"]
    ):
        raise ValueError("Expected aggregate dev collision is absent")
    return {
        "schema": "decision2-anli-score-train-quarantine-supplement/1",
        "parent_receipt_sha256": prior_sha256,
        "frozen_selection": result,
        "admission": "HOLD_NO_TRAINING_OR_REDISTRIBUTION",
        "gpu_hours": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-directory", type=Path, required=True)
    parser.add_argument("--dev-directory", type=Path, required=True)
    parser.add_argument("--tokenizer-directory", type=Path, required=True)
    parser.add_argument("--prior-receipt", type=Path, required=True)
    parser.add_argument("--prior-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.train_directory,
        args.dev_directory,
        args.tokenizer_directory,
        args.prior_receipt,
        args.prior_sha256,
    )
    write_once(args.output, result)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "admission": result["admission"],
                "technical_upper_bound": result["frozen_selection"][
                    "fixed_sample_after_exact_dev_and_conflict_quarantine_upper_bound"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
