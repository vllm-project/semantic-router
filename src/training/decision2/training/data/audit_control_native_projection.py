"""CPU-only native Choice/Noul projection screen for publisher ConTRoL TRAIN.

This emits aggregate counts, never student TRAIN rows or source text. A clean
screen is necessary but not sufficient for admission: source semantics, near
neighbors and the matched training-arm budget still need independent review.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

from training.data.audit_control_nli_source import read_train
from training.data.audit_nli_evidence_score import (
    Pair,
    normalize,
    overlap_screen,
    read_protected,
    sha_file,
)
from training.model.data import INPUT_FIELDS, digest, validate_row
from training.model.decision_model import encode

CHOICE = (
    ("supports", "The context establishes the claim."),
    ("contradicts", "The context establishes that the claim is false."),
    ("undetermined", "The context neither establishes nor refutes the claim."),
)
CHOICE_GOLD = {
    "entailment": "supports",
    "contradiction": "contradicts",
    "neutral": "undetermined",
}
NOUL_QUESTIONS = {
    "supports": "Does the context establish the claim?",
    "contradicts": "Does the context establish that the claim is false?",
}


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def quarantined_groups(
    pairs: list[Pair], protected: list[tuple[str, str]]
) -> tuple[set[str], dict[str, Any]]:
    overlap = overlap_screen(pairs, protected)
    text_groups: dict[str, set[str]] = collections.defaultdict(set)
    for pair in pairs:
        for field in (pair.premise, pair.hypothesis):
            text_groups[_digest(normalize(field))].add(pair.group)
    blocked = {
        group
        for _, _, text_hash, _ in overlap["hashed_matches"]
        for group in text_groups[text_hash]
    }
    if len(blocked) != overlap["matched_source_group_count"]:
        raise ValueError("Overlap hashes do not recover exactly the source groups")
    return blocked, overlap


def _row(pair: Pair, *, task: str) -> dict[str, Any]:
    if task not in {"choice", "supports", "contradicts"}:
        raise ValueError("Unknown native projection")
    identity = _digest(f"{pair.position}\0{pair.group}\0{task}")
    if task == "choice":
        options = [
            {"key": key, "description": description} for key, description in CHOICE
        ]
        shift = int(identity[:8], 16) % len(options)
        options = options[shift:] + options[:shift]
        gold = CHOICE_GOLD[pair.label]
        kind = "choice"
        question = "Which relation between the context and claim is established?"
    else:
        options = [
            {"key": "false", "description": "No; not established by the context."},
            {"key": "true", "description": "Yes; established by the context."},
        ]
        if int(identity[:8], 16) % 2:
            options.reverse()
        gold = "true" if CHOICE_GOLD[pair.label] == task else "false"
        kind = "noul"
        question = NOUL_QUESTIONS[task]
    row = {
        "id": f"control-train-{identity[:20]}",
        "state": pair.premise,
        "instructions": {"claim": pair.hypothesis, "question": question},
        "options": options,
        "label": next(i for i, option in enumerate(options) if option["key"] == gold),
        "task_type": kind,
        "family": "contextual_nli",
        "group_id": f"control:{_digest(pair.group)[:20]}",
        "language": "en",
        "split": "train",
        "source": "ConTRoL publisher TRAIN",
        "evaluation_role": "train",
        "render_template": "control-native-choice-noul-screen-v1",
        "audit_metadata": {
            "source_position": pair.position,
            "source_label": pair.label,
        },
    }
    row["input_sha256"] = digest({field: row[field] for field in INPUT_FIELDS})
    validate_row(row, "train")
    return row


def _summary(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)
    return {
        "median": ordered[int(0.5 * (len(ordered) - 1))],
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
        "p99": ordered[int(0.99 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def audit(
    repo: Path,
    *,
    protected_inventory: Path,
    tokenizer_path: Path,
    max_length: int = 8192,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    pairs, provenance = read_train(repo)
    protected, inventory_receipt = read_protected(protected_inventory, [])
    blocked, overlap = quarantined_groups(pairs, protected)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    token_files = {
        name: sha_file(tokenizer_path / name)
        for name in ("tokenizer.json", "tokenizer_config.json")
    }

    seen_pairs: dict[tuple[str, str], str] = {}
    duplicate_rows = 0
    conflicting_groups: set[str] = set()
    candidate_pairs = []
    for pair in pairs:
        if pair.group in blocked:
            continue
        key = normalize(pair.premise), normalize(pair.hypothesis)
        prior = seen_pairs.get(key)
        if prior is not None:
            duplicate_rows += 1
            if prior != pair.label:
                conflicting_groups.add(pair.group)
            continue
        seen_pairs[key] = pair.label
        candidate_pairs.append(pair)
    if conflicting_groups:
        candidate_pairs = [
            pair for pair in candidate_pairs if pair.group not in conflicting_groups
        ]

    length_by_task: dict[str, list[int]] = collections.defaultdict(list)
    class_counts: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    position_counts: dict[str, collections.Counter[int]] = collections.defaultdict(
        collections.Counter
    )
    hash_inputs = hashlib.sha256()
    for pair in candidate_pairs:
        for task in ("choice", "supports", "contradicts"):
            row = _row(pair, task=task)
            tokenized = encode(row, tokenizer, max_length=2**31 - 1)
            length_by_task[task].append(len(tokenized["ids"]))
            key = row["options"][row["label"]]["key"]
            class_counts[task][key] += 1
            position_counts[task][row["label"]] += 1
            hash_inputs.update((row["input_sha256"] + "\n").encode())

    return {
        "status": "CPU projection screen only; student TRAIN not admitted",
        "publisher_train_only": True,
        "publisher_dev_test_labels_opened": False,
        "source": provenance,
        "protected_inventory_sha256": inventory_receipt["inventory_sha256"],
        "protected_overlap": {
            "matched_source_groups": len(blocked),
            "roles": overlap["role_group_counts"],
            "protected_leaves_over_800_chars_not_near_scanned": overlap[
                "protected_snippets_over_800_chars_not_near_scanned"
            ],
        },
        "quarantined_rows": sum(pair.group in blocked for pair in pairs),
        "duplicate_pair_rows_removed": duplicate_rows,
        "conflicting_label_groups_removed": len(conflicting_groups),
        "candidate_original_pairs": len(candidate_pairs),
        "candidate_independent_premise_groups": len(
            {pair.group for pair in candidate_pairs}
        ),
        "native_rows_by_type": {
            "choice": len(candidate_pairs),
            "noul": 2 * len(candidate_pairs),
            "score": 0,
        },
        "full_native_prompt_tokens": {
            task: {
                **_summary(values),
                "over_limit": sum(value > max_length for value in values),
            }
            for task, values in sorted(length_by_task.items())
        },
        "native_max_length": max_length,
        "gold_keys": {
            task: dict(sorted(counts.items()))
            for task, counts in sorted(class_counts.items())
        },
        "gold_option_positions": {
            task: dict(sorted(counts.items()))
            for task, counts in sorted(position_counts.items())
        },
        "input_payload_sha256": hash_inputs.hexdigest(),
        "tokenizer_files": token_files,
        "admission_gates_still_open": [
            "blind source-label-to-native-rubric review",
            "semantic near-neighbor review including long protected leaves",
            "matched-arm row/token/padded-exposure and Score-retention preregistration",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publisher-repo", type=Path, required=True)
    parser.add_argument("--protected-inventory", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(
        args.publisher_repo,
        protected_inventory=args.protected_inventory,
        tokenizer_path=args.tokenizer_path,
        max_length=args.max_length,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
