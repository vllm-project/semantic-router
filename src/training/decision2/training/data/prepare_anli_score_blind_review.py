"""Prepare a small, source-key-blind native Score rubric review packet.

The packet and its key are private outputs. No protected evaluation gold is read.
This tool does not admit ANLI for training, evaluation, or redistribution.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

from training.data.audit_anli_score_diagnostic import (
    SCORE_CRITERIA,
    SCORE_INSTRUCTIONS,
    Pair,
    normalize,
    score_row,
    write_once,
)
from training.data.audit_anli_score_train_quarantine import dev_input_spans
from training.data.audit_anli_score_train_source import (
    TrainRow,
    read_dev_inputs,
    read_train,
    sample_whole_groups,
)
from training.model.data import file_sha256

REVIEW_SALT = "decision2-anli-score-rubric-review-v1"
REVIEW_GROUPS_PER_ROUND = 3
MIN_GROUP_ROWS = 3
MAX_GROUP_ROWS = 12
SOURCE_TO_SCORE = {0: 2, 1: 1, 2: 0}
NATIVE_INPUT_KEYS = {"id", "state", "instructions", "options", "task_type"}


def _rank(group: str) -> tuple[str, str]:
    return hashlib.sha256((REVIEW_SALT + "\0" + group).encode()).hexdigest(), group


def _review_id(round_id: int, group: str, position: int) -> str:
    digest = hashlib.sha256(
        (REVIEW_SALT + "\0" + group + "\0" + str(position)).encode()
    ).hexdigest()
    return f"r{round_id}-{digest[:20]}"


def _forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        if any(
            key.casefold() in {"answer", "answers", "label", "target", "gold", "reason"}
            for key in value
        ):
            return True
        return any(_forbidden_key(item) for item in value.values())
    if isinstance(value, list):
        return any(_forbidden_key(item) for item in value)
    return False


def build_packet(
    selected: list[TrainRow], source: list[TrainRow], open_dev_spans: set[str]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    """Deterministic whole-group review sample, with source labels only in key."""
    pair_labels: dict[tuple[str, str], set[int]] = collections.defaultdict(set)
    pair_rows: collections.Counter[tuple[str, str]] = collections.Counter()
    for row in source:
        pair = (row.group, normalize(row.hypothesis))
        pair_labels[pair].add(row.label)
        pair_rows[pair] += 1
    excluded = {
        group for (group, _), labels in pair_labels.items() if len(labels) != 1
    } | {group for (group, _), count in pair_rows.items() if count != 1}
    excluded.update(
        row.group
        for row in selected
        if normalize(row.premise) in open_dev_spans
        or normalize(row.hypothesis) in open_dev_spans
    )
    groups: dict[str, list[TrainRow]] = collections.defaultdict(list)
    for row in selected:
        groups[row.group].append(row)
    packet_groups = []
    answers = []
    counts = {}
    seen_ids = set()
    for round_id in (1, 2, 3):
        eligible = sorted(
            (
                group
                for group, items in groups.items()
                if group not in excluded
                and items[0].round == round_id
                and MIN_GROUP_ROWS <= len(items) <= MAX_GROUP_ROWS
                and {row.label for row in items} == set(SOURCE_TO_SCORE)
            ),
            key=_rank,
        )
        if len(eligible) < REVIEW_GROUPS_PER_ROUND:
            raise ValueError("Insufficient complete, three-relation review groups")
        chosen = eligible[:REVIEW_GROUPS_PER_ROUND]
        counts[f"r{round_id}_groups"] = len(chosen)
        counts[f"r{round_id}_rows"] = sum(len(groups[group]) for group in chosen)
        for group in chosen:
            group_id = f"r{round_id}-{_rank(group)[0][:20]}"
            items = []
            for row in sorted(groups[group], key=lambda item: item.position):
                review_id = _review_id(round_id, group, row.position)
                if review_id in seen_ids:
                    raise ValueError("Review ID collision")
                seen_ids.add(review_id)
                native = score_row(
                    Pair(row.round, row.premise, row.hypothesis, row.label, False),
                    row.position,
                )
                request = {key: native[key] for key in NATIVE_INPUT_KEYS}
                request["id"] = review_id
                if _forbidden_key(request):
                    raise ValueError("Source answer-like field entered review request")
                items.append({"review_id": review_id, "request": request})
                answers.append(
                    {
                        "review_id": review_id,
                        "source_label": row.label,
                        "mapped_native_score": SOURCE_TO_SCORE[row.label],
                    }
                )
            packet_groups.append({"review_group_id": group_id, "items": items})
    return (
        {
            "schema": "decision2-anli-score-rubric-blind-packet/1",
            "review_instructions": (
                "For every claim, use only the supplied evidence and native 0/1/2 "
                "criteria. Give one score and a brief evidence-based reason. "
                "Flag any ambiguous, unsupported, misleading, or shortcut-prone "
                "item and assess each complete premise group together. Do not "
                "consult the separate source-label key."
            ),
            "native_score_instructions": SCORE_INSTRUCTIONS,
            "native_score_criteria": SCORE_CRITERIA,
            "groups": packet_groups,
        },
        {"schema": "decision2-anli-score-rubric-private-key/1", "answers": answers},
        counts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-directory", type=Path, required=True)
    parser.add_argument("--dev-directory", type=Path, required=True)
    parser.add_argument("--tokenizer-directory", type=Path, required=True)
    parser.add_argument("--parent-receipt", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    paths = (args.packet, args.private_key, args.receipt)
    if len(set(paths)) != 3 or any(
        path.exists() or path.is_symlink() for path in paths
    ):
        raise ValueError("Review output paths must be distinct and new")
    if file_sha256(args.parent_receipt) != args.parent_sha256:
        raise ValueError("Frozen parent source audit receipt changed")
    parent = json.loads(args.parent_receipt.read_text(encoding="utf-8"))
    if (
        parent.get("schema") != "decision2-anli-score-train-source-screen/1"
        or parent["selection"]["selected_rows"] != 12000
        or parent["selection"]["selected_groups"] != 930
        or parent["selection"]["selected_native_tokens"] != 2599097
    ):
        raise ValueError("Frozen parent source audit is ineligible")
    source, hashes = read_train(args.train_directory)
    if hashes != parent["source_sha256"]:
        raise ValueError("ANLI TRAIN source changed")
    from training.data.audit_anli_score_diagnostic import TOKENIZER_FILES

    for name, expected in TOKENIZER_FILES.items():
        if file_sha256(args.tokenizer_directory / name) != expected:
            raise ValueError("Native tokenizer changed")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_directory, local_files_only=True, trust_remote_code=False
    )
    selected, selection, _ = sample_whole_groups(
        source, read_dev_inputs(args.dev_directory), set(), tokenizer
    )
    if any(
        selection[key] != parent["selection"][key]
        for key in ("selected_rows", "selected_groups", "selected_native_tokens")
    ):
        raise ValueError("Frozen source selection changed")
    packet, key, counts = build_packet(
        selected, source, dev_input_spans(args.dev_directory)
    )
    write_once(args.packet, packet)
    write_once(args.private_key, key)
    write_once(
        args.receipt,
        {
            "schema": "decision2-anli-score-rubric-packet-receipt/1",
            "parent_receipt_sha256": args.parent_sha256,
            "packet_sha256": file_sha256(args.packet),
            "private_key_sha256": file_sha256(args.private_key),
            "counts": counts,
            "admission": "HOLD_NO_TRAINING_OR_REDISTRIBUTION",
            "gpu_hours": 0,
        },
    )
    print(
        json.dumps(
            {
                "schema": "decision2-anli-score-rubric-packet-receipt/1",
                "counts": counts,
                "admission": "HOLD",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
