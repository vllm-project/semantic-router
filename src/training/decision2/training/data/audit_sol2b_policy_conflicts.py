"""CPU quality and exact-token screens for a private policy-conflict candidate."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from training.data.build_sol2b_policy_conflicts import FAMILIES, SEED, build_all
from training.model.data import canonical, load_partition

CONTROL_TRAIN_SHA256 = (
    "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755"
)
CONTROL_TOKENS = 4_194_465
CONTROL_ROWS = 7_455
NEW_SOURCE_MIN_SHARE = 0.20
NEW_SOURCE_MAX_SHARE = 0.25
TOTAL_TOLERANCE = 0.01


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_blind_packet(blind: list[dict[str, Any]]) -> dict[str, Any]:
    if len(blind) != 48:
        raise ValueError("Expected exactly 48 blind review groups")
    counts = collections.Counter(item.get("family") for item in blind)
    if counts != dict.fromkeys(FAMILIES, 12):
        raise ValueError("Blind review groups are not 12 per family")
    if len({item.get("group_id") for item in blind}) != 48:
        raise ValueError("Repeated blind review group")
    forbidden = ("label", "answer", "oracle", "fact_digest", "audit_metadata")
    for item in blind:
        encoded = canonical(item).lower()
        if any(f'"{field}"' in encoded for field in forbidden):
            raise ValueError("Blind packet exposes a label or oracle field")
        questions = item.get("questions")
        if not isinstance(questions, list) or len(questions) != 3:
            raise ValueError("Blind packet needs three native questions per group")
        if {question.get("task_type") for question in questions} != {
            "choice",
            "noul",
            "score",
        }:
            raise ValueError("Blind packet lost a native question type")
        if any(
            not isinstance(question.get("state"), str) or not question["state"]
            for question in questions
        ):
            raise ValueError("Blind question has no context")
    return {
        "blind_groups": len(blind),
        "blind_groups_per_family": 12,
        "answer_blind": True,
    }


def _features(row: dict[str, Any]) -> collections.Counter[str]:
    # Deliberately withhold state, group ID, source, render_template and metadata.
    prompt = (
        row["instructions"]
        + " "
        + " ".join(str(option["description"]) for option in row["options"])
    )
    return collections.Counter(re.findall(r"[a-z]+", prompt.lower()))


def _class_key(row: dict[str, Any]) -> str:
    return row["options"][row["label"]]["key"]


def shortcut_probe(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Four-fold, group-disjoint bag-of-words probe without the state text."""
    by_type: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_type[row["task_type"]].append(row)
    result = {}
    for task_type, samples in sorted(by_type.items()):
        label_counts = collections.Counter(_class_key(row) for row in samples)
        labels = sorted(label_counts)
        successes = 0
        for fold in range(4):

            def assigned(row: dict[str, Any]) -> int:
                return int(hashlib.sha256(row["group_id"].encode()).hexdigest(), 16) % 4

            train = [row for row in samples if assigned(row) != fold]
            test = [row for row in samples if assigned(row) == fold]
            if not train or not test:
                raise ValueError("Empty shortcut-probe fold")
            priors = collections.Counter(_class_key(row) for row in train)
            vocab = set()
            features = {}
            totals = collections.Counter()
            for row in train:
                label = _class_key(row)
                counts = _features(row)
                vocab.update(counts)
                bucket = features.setdefault(label, collections.Counter())
                bucket.update(counts)
                totals[label] += sum(counts.values())
            for row in test:
                tokens = _features(row)
                scores = {
                    label: math.log((priors[label] + 1) / (len(train) + len(labels)))
                    + sum(
                        count
                        * math.log(
                            (features[label][term] + 1) / (totals[label] + len(vocab))
                        )
                        for term, count in tokens.items()
                    )
                    for label in labels
                }
                predicted = sorted(labels, key=lambda label: (-scores[label], label))[0]
                successes += predicted == _class_key(row)
        accuracy = successes / len(samples)
        majority_accuracy = max(label_counts.values()) / len(samples)
        result[task_type] = {
            "n": len(samples),
            "majority_accuracy": majority_accuracy,
            "state_removed_accuracy": accuracy,
            "delta": accuracy - majority_accuracy,
            "gate_pass": accuracy <= majority_accuracy + 0.05,
        }
    return result


def _native_tokens(row: dict[str, Any], tokenizer: Any) -> int:
    # Import only in the model runtime. This exactly follows the selected
    # DecisionModel segmented, no-truncation native encoding path.
    from training.model.decision_model import segments

    prefix, options, suffix = segments(row)
    return sum(
        len(tokenizer.encode(part, add_special_tokens=False))
        for part in (prefix, *options, suffix)
    )


def token_feasibility(
    candidate: list[dict[str, Any]], control: list[dict[str, Any]], tokenizer: Any
) -> dict[str, Any]:
    new_lengths = {row["id"]: _native_tokens(row, tokenizer) for row in candidate}
    type_tokens = collections.Counter()
    for row in candidate:
        type_tokens[row["task_type"]] += new_lengths[row["id"]]
    original_lengths = {row["id"]: _native_tokens(row, tokenizer) for row in control}
    new_total = sum(new_lengths.values())
    original_total = sum(original_lengths.values())
    if original_total != CONTROL_TOKENS or len(control) != CONTROL_ROWS:
        raise ValueError("Exact archived Sol control token/row budget changed")
    new_share = new_total / original_total
    retained = [
        row
        for row in control
        if row["source"] == "google_goemotions_official_train"
        or row["task_type"] == "score"
    ]
    retained_ids = {row["id"] for row in retained}
    replaceable = [row for row in control if row["id"] not in retained_ids]
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in replaceable:
        groups[row["group_id"]].append(row)
    all_group_rows = collections.Counter(row["group_id"] for row in control)
    whole_groups = [
        rows for key, rows in groups.items() if len(rows) == all_group_rows[key]
    ]
    whole_rows = sum(map(len, whole_groups))
    whole_tokens = sum(
        original_lengths[row["id"]] for group in whole_groups for row in group
    )
    return {
        "source_native_rows": len(candidate),
        "source_native_tokens": new_total,
        "source_token_share": new_share,
        "source_share_gate_pass": NEW_SOURCE_MIN_SHARE
        <= new_share
        <= NEW_SOURCE_MAX_SHARE,
        "source_tokens_by_type": dict(sorted(type_tokens.items())),
        "source_type_token_shares": {
            kind: count / new_total for kind, count in sorted(type_tokens.items())
        },
        "source_type_balance_gate_pass": len(type_tokens) == 3
        and all(
            abs(count / new_total - 1 / 3) <= 0.05 for count in type_tokens.values()
        ),
        "source_max_length": max(new_lengths.values()),
        "control_native_rows": len(control),
        "control_native_tokens": original_total,
        "retained_human_rows": sum(
            row["source"] == "google_goemotions_official_train" for row in retained
        ),
        "retained_old_score_rows": sum(row["task_type"] == "score" for row in retained),
        "whole_replaceable_groups": len(whole_groups),
        "whole_replaceable_rows": whole_rows,
        "whole_replaceable_tokens": whole_tokens,
        "replacement_row_capacity_pass": whole_rows >= len(candidate),
        "replacement_token_capacity_pass": whole_tokens
        >= new_total - CONTROL_TOKENS * TOTAL_TOLERANCE,
        "exact_replacement_roster_selected": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    if (args.control is None) != (args.tokenizer is None):
        raise ValueError("Control and tokenizer must be provided together")
    rows = load_partition(args.candidate, "train")
    blind = _jsonl(args.blind)
    diagnostic, _ = build_all(groups_per_family=32, seed=SEED + "-diagnostic")
    if {row["input_sha256"] for row in rows} & {
        row["input_sha256"] for row in diagnostic
    }:
        raise ValueError("Shortcut diagnostic shares a training input")
    report: dict[str, Any] = {
        "candidate_sha256": file_sha256(args.candidate),
        "blind_sha256": file_sha256(args.blind),
        "blind": verify_blind_packet(blind),
        "diagnostic_groups": len({row["group_id"] for row in diagnostic}),
        "shortcut": shortcut_probe(diagnostic),
    }
    if args.control is not None:
        if file_sha256(args.control) != CONTROL_TRAIN_SHA256:
            raise ValueError("Frozen control TRAIN SHA changed")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, local_files_only=True, trust_remote_code=False
        )
        report["tokens"] = token_feasibility(
            rows, load_partition(args.control, "train"), tokenizer
        )
    print(canonical(report))


if __name__ == "__main__":
    main()
