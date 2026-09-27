"""CPU-only, preregistered ANLI TRAIN source screen; never trains or infers."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from training.data.audit_27b_full_input_overlap import (
    _normalize,
    full_input_overlap_rows,
    input_spans,
)
from training.data.audit_anli_score_diagnostic import (
    INPUT_COLUMNS,
    SOURCE_ID,
    SOURCE_LABELS,
    SOURCE_LICENSE,
    SOURCE_REVISION,
    TOKENIZER_FILES,
    Pair,
    load_projected_roles,
    normalize,
    percentile,
    score_row,
    write_once,
)
from training.data.audit_anli_score_diagnostic import (
    ROUND_FILES as DEV_FILES,
)
from training.data.plan_goldfree_inventory import CORE_ROLES
from training.model.data import file_sha256
from training.model.decision_model import PROMPT_VERSION, encode

TRAIN_FILES = {
    1: (
        "train_r1-00000-of-00001.parquet",
        16946,
        "de2d038ae67f1fb1872073490b9e7685e9114d5f278ddd4631905fe0a4ecbcff",
    ),
    2: (
        "train_r2-00000-of-00001.parquet",
        45460,
        "209f4a15bf77224c62ffbde5f150fda928a7e2f5175366f4cacc3c7588aab13d",
    ),
    3: (
        "train_r3-00000-of-00001.parquet",
        100459,
        "c1d3f614d673888ac56b9ab62324e21583c98a11c4fef84e938d0f8fc414b29a",
    ),
}
PROTECTED_MANIFEST_SHA256 = (
    "26bbaf82eb1c30c0f2093c70d27718fab6731ea8c80e9a451547b03bd30897e1"
)
SAMPLE_SALT = "decision2-anli-score-train-v1"
MAX_ROWS_PER_ROUND = 4000
MAX_ROWS_TOTAL = 12000
MAX_NATIVE_TOKENS = 3_600_000
MAX_REQUEST_TOKENS = 4096
CUES = (
    "not",
    "no",
    "never",
    "always",
    "some",
    "all",
    "only",
    "before",
    "after",
    "because",
    "may",
    "might",
    "must",
)
LENGTH_BUCKETS = ((0, 80), (81, 160), (161, None))


@dataclass(frozen=True)
class TrainRow:
    round: int
    position: int
    premise: str
    hypothesis: str
    label: int
    reason_present: bool

    @property
    def group(self) -> str:
        return normalize(self.premise)


def read_train(directory: Path) -> tuple[list[TrainRow], dict[str, str]]:
    rows = []
    hashes = {}
    uids = set()
    for round_id, (filename, expected_rows, expected_sha) in TRAIN_FILES.items():
        path = directory / filename
        if not path.is_file() or file_sha256(path) != expected_sha:
            raise ValueError("Pinned ANLI TRAIN bytes are missing or changed")
        table = pq.read_table(path)
        if table.column_names != INPUT_COLUMNS or table.num_rows != expected_rows:
            raise ValueError("ANLI TRAIN schema or count changed")
        hashes[f"r{round_id}"] = expected_sha
        for position, (uid, premise, hypothesis, label, reason) in enumerate(
            zip(*(table[name].to_pylist() for name in INPUT_COLUMNS), strict=True)
        ):
            if (
                not isinstance(uid, str)
                or not uid
                or uid in uids
                or not isinstance(premise, str)
                or not premise.strip()
                or not isinstance(hypothesis, str)
                or not hypothesis.strip()
                or type(label) is not int
                or label not in SOURCE_LABELS
                or not isinstance(reason, str)
            ):
                raise ValueError("ANLI TRAIN row is malformed or repeated")
            uids.add(uid)
            rows.append(
                TrainRow(
                    round_id, position, premise, hypothesis, label, bool(reason.strip())
                )
            )
    return rows, hashes


def read_dev_inputs(directory: Path) -> set[str]:
    groups = set()
    for filename, expected_rows, expected_sha in DEV_FILES.values():
        path = directory / filename
        if not path.is_file() or file_sha256(path) != expected_sha:
            raise ValueError("Pinned ANLI dev bytes are missing or changed")
        metadata = pq.read_metadata(path)
        if (
            metadata.num_rows != expected_rows
            or pq.read_schema(path).names != INPUT_COLUMNS
        ):
            raise ValueError("ANLI dev schema or count changed")
        premise = pq.read_table(path, columns=["premise"])["premise"].to_pylist()
        if any(not isinstance(value, str) or not value.strip() for value in premise):
            raise ValueError("ANLI dev premise is malformed")
        groups.update(normalize(value) for value in premise)
    return groups


def _group_rank(group: str) -> tuple[bytes, str]:
    return (
        hashlib.sha256((SAMPLE_SALT + "\0" + group).encode()).digest(),
        group,
    )


def protected_exact_groups(
    rows: list[TrainRow], roles: dict[str, list[dict[str, Any]]]
) -> tuple[set[str], dict[str, dict[str, int]]]:
    """Exact screen of the whole source; only aggregate group counts escape."""
    reference_raw = collections.defaultdict(set)
    reference_norm = collections.defaultdict(set)
    for role, prompts in roles.items():
        for prompt in prompts:
            for span in input_spans(prompt):
                reference_raw[span].add(role)
                reference_norm[_normalize(span)].add(role)
    by_role: dict[str, dict[str, set[str]]] = {
        role: {"exact_raw": set(), "exact_normalized": set()} for role in roles
    }
    for row in rows:
        for value in (row.premise, row.hypothesis):
            for role in reference_raw.get(value, ()):
                by_role[role]["exact_raw"].add(row.group)
            for role in reference_norm.get(_normalize(value), ()):
                by_role[role]["exact_normalized"].add(row.group)
    excluded = set().union(
        *(groups for views in by_role.values() for groups in views.values())
    )
    return excluded, {
        role: {name: len(groups) for name, groups in sorted(views.items())}
        for role, views in sorted(by_role.items())
    }


def sample_whole_groups(
    rows: list[TrainRow],
    dev_groups: set[str],
    protected_exact: set[str],
    tokenizer: Any,
) -> tuple[list[TrainRow], dict[str, Any], list[int]]:
    by_group: dict[str, list[TrainRow]] = collections.defaultdict(list)
    for row in rows:
        by_group[row.group].append(row)
    crossing = {
        group
        for group, items in by_group.items()
        if len({item.round for item in items}) > 1
    }
    excluded = set(dev_groups) | protected_exact | crossing
    selected = []
    lengths = []
    token_total = 0
    by_round = collections.Counter()
    skipped_budget = collections.Counter()
    skipped_overlength = collections.Counter()
    for round_id in TRAIN_FILES:
        ordered = sorted(
            (
                group
                for group, items in by_group.items()
                if items[0].round == round_id and group not in excluded
            ),
            key=_group_rank,
        )
        for group in ordered:
            items = by_group[group]
            if (
                by_round[round_id] + len(items) > MAX_ROWS_PER_ROUND
                or len(selected) + len(items) > MAX_ROWS_TOTAL
            ):
                skipped_budget[round_id] += 1
                continue
            group_lengths = [
                len(
                    encode(
                        score_row(
                            Pair(
                                item.round,
                                item.premise,
                                item.hypothesis,
                                item.label,
                                item.reason_present,
                            ),
                            item.position,
                        ),
                        tokenizer,
                        max_length=1_000_000,
                    )["ids"]
                )
                for item in items
            ]
            if any(length > MAX_REQUEST_TOKENS for length in group_lengths):
                skipped_overlength[round_id] += 1
                continue
            if token_total + sum(group_lengths) > MAX_NATIVE_TOKENS:
                skipped_budget[round_id] += 1
                continue
            selected.extend(items)
            lengths.extend(group_lengths)
            token_total += sum(group_lengths)
            by_round[round_id] += len(items)
    return (
        selected,
        {
            "sample_salt_sha256": hashlib.sha256(SAMPLE_SALT.encode()).hexdigest(),
            "max_rows_per_round": MAX_ROWS_PER_ROUND,
            "max_rows_total": MAX_ROWS_TOTAL,
            "max_native_tokens": MAX_NATIVE_TOKENS,
            "max_request_tokens": MAX_REQUEST_TOKENS,
            "selected_rows": len(selected),
            "selected_groups": len({row.group for row in selected}),
            "selected_rows_by_round": {
                f"r{round_id}": by_round[round_id] for round_id in TRAIN_FILES
            },
            "selected_native_tokens": token_total,
            "excluded_dev_exact_groups": len(set(by_group) & dev_groups),
            "excluded_protected_exact_groups": len(protected_exact),
            "excluded_cross_round_groups": len(crossing),
            "skipped_row_or_token_budget_groups_by_round": dict(skipped_budget),
            "skipped_overlength_groups_by_round": dict(skipped_overlength),
        },
        lengths,
    )


def _cue_names(text: str) -> set[str]:
    words = set(re.findall(r"[a-z]+", text.casefold()))
    found = set(words & set(CUES))
    if any(char.isdigit() for char in text):
        found.add("number_or_date")
    length = len(text)
    for lower, upper in LENGTH_BUCKETS:
        if length >= lower and (upper is None or length <= upper):
            found.add(f"length_{lower}_{upper if upper is not None else 'plus'}")
            break
    return found


def shortcut_profile(rows: list[TrainRow]) -> dict[str, Any]:
    results = {}
    for round_id in TRAIN_FILES:
        subset = [row for row in rows if row.round == round_id]
        labels = collections.Counter(row.label for row in subset)
        majority = max(labels.values()) / len(subset)
        positions = {}
        for modulus in (3, 9):
            slots = collections.defaultdict(list)
            for row in subset:
                slots[row.position % modulus].append(row.label)
            correct = sum(
                collections.Counter(slot).most_common(1)[0][1]
                for slot in slots.values()
            )
            positions[str(modulus)] = round(correct / len(subset), 6)
        cues: dict[str, collections.Counter[int]] = collections.defaultdict(
            collections.Counter
        )
        for row in subset:
            for cue in _cue_names(row.hypothesis):
                cues[cue][row.label] += 1
        evaluated = {}
        for cue, counts in sorted(cues.items()):
            support = sum(counts.values())
            if support < 50:
                continue
            lift = max(
                (counts[label] / support) / (labels[label] / len(subset))
                for label in SOURCE_LABELS
                if labels[label]
            )
            evaluated[cue] = {"support": support, "max_label_lift": round(lift, 4)}
        results[f"r{round_id}"] = {
            "rows": len(subset),
            "majority_accuracy": round(majority, 6),
            "file_position_modulo_accuracy": positions,
            "hypothesis_only_cue_associations": evaluated,
            "shortcut_review_trigger": (
                majority > 0.5
                or any(value > majority + 0.05 for value in positions.values())
                or any(value["max_label_lift"] > 2.5 for value in evaluated.values())
            ),
        }
    return results


def source_profile(rows: list[TrainRow]) -> dict[str, Any]:
    by_group = collections.defaultdict(set)
    groups_by_round = collections.defaultdict(set)
    pair_labels = collections.defaultdict(set)
    by_round = {}
    for row in rows:
        by_group[row.group].add(row.round)
        groups_by_round[row.round].add(row.group)
        pair_labels[(row.group, normalize(row.hypothesis))].add(row.label)
    for round_id in TRAIN_FILES:
        subset = [row for row in rows if row.round == round_id]
        counts = collections.Counter(SOURCE_LABELS[row.label] for row in subset)
        by_round[f"r{round_id}"] = {
            "rows": len(subset),
            "premise_groups": len(groups_by_round[round_id]),
            "labels": {name: counts[name] for name in SOURCE_LABELS.values()},
            "reason_present": sum(row.reason_present for row in subset),
            "pair_characters": {
                "median": percentile(
                    [len(row.premise) + len(row.hypothesis) for row in subset], 50
                ),
                "p99": percentile(
                    [len(row.premise) + len(row.hypothesis) for row in subset], 99
                ),
                "max": max(len(row.premise) + len(row.hypothesis) for row in subset),
            },
        }
    return {
        "rows": len(rows),
        "normalized_premise_groups": len(by_group),
        "cross_round_premise_groups": sum(
            len(value) > 1 for value in by_group.values()
        ),
        "repeated_normalized_pair_rows": len(rows) - len(pair_labels),
        "normalized_pairs_with_conflicting_labels": sum(
            len(value) > 1 for value in pair_labels.values()
        ),
        "rounds": by_round,
    }


def input_rows(rows: list[TrainRow], prefix: str) -> list[dict[str, str]]:
    return [
        {
            "id": f"{prefix}-{index}",
            "state": row.premise,
            "instructions": row.hypothesis,
        }
        for index, row in enumerate(rows)
    ]


def audit(
    train_directory: Path,
    dev_directory: Path,
    tokenizer_directory: Path,
    protected_manifest: Path,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    rows, source_hashes = read_train(train_directory)
    dev_groups = read_dev_inputs(dev_directory)
    roles = load_projected_roles(
        protected_manifest, expected_sha256=PROTECTED_MANIFEST_SHA256
    )
    if set(roles) != CORE_ROLES:
        raise ValueError("Eight-role protected input inventory incomplete")
    for name, expected in TOKENIZER_FILES.items():
        path = tokenizer_directory / name
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError("Pinned native tokenizer changed")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_directory, local_files_only=True, trust_remote_code=False
    )
    protected_exact, exact_by_role = protected_exact_groups(rows, roles)
    selected, selection, lengths = sample_whole_groups(
        rows, dev_groups, protected_exact, tokenizer
    )
    if not selected:
        raise ValueError("ANLI candidate selection is empty")
    selected_input = input_rows(selected, "anli-train")
    near_by_role = {
        role: full_input_overlap_rows(selected_input, prompts)
        for role, prompts in sorted(roles.items())
    }
    dev_input_rows = []
    for filename, _, _ in DEV_FILES.values():
        table = pq.read_table(
            dev_directory / filename, columns=["premise", "hypothesis"]
        )
        for premise, hypothesis in zip(
            table["premise"].to_pylist(), table["hypothesis"].to_pylist(), strict=True
        ):
            dev_input_rows.append({"premise": premise, "hypothesis": hypothesis})
    open_dev_near = full_input_overlap_rows(
        selected_input,
        [
            {
                "id": f"anli-dev-{index}",
                "state": row["premise"],
                "instructions": row["hypothesis"],
            }
            for index, row in enumerate(dev_input_rows)
        ],
    )
    selection["labels_by_round"] = {
        f"r{round_id}": {
            name: sum(
                row.round == round_id and SOURCE_LABELS[row.label] == name
                for row in selected
            )
            for name in SOURCE_LABELS.values()
        }
        for round_id in TRAIN_FILES
    }
    selection["native_prompt_tokens_by_round_and_label"] = {
        f"r{round_id}": {
            name: {
                "rows": len(subset),
                "median": percentile(subset, 50) if subset else None,
                "p99": percentile(subset, 99) if subset else None,
                "max": max(subset) if subset else None,
            }
            for name in SOURCE_LABELS.values()
            for subset in [
                [
                    length
                    for row, length in zip(selected, lengths, strict=True)
                    if row.round == round_id and SOURCE_LABELS[row.label] == name
                ]
            ]
        }
        for round_id in TRAIN_FILES
    }
    selected_class_floor_pass = all(
        count >= 0.15 * sum(counts.values())
        for counts in selection["labels_by_round"].values()
        for count in counts.values()
    )
    shortcuts = shortcut_profile(rows)
    selected_shortcuts = shortcut_profile(selected)
    overlap_found = any(any(item["counts"].values()) for item in near_by_role.values())
    dev_overlap_found = any(open_dev_near["counts"].values())
    return {
        "schema": "decision2-anli-score-train-source-screen/1",
        "source": SOURCE_ID,
        "source_revision": SOURCE_REVISION,
        "source_license": SOURCE_LICENSE,
        "source_sha256": source_hashes,
        "source_profile": source_profile(rows),
        "selection": selection,
        "native_prompt_version": PROMPT_VERSION,
        "tokenizer_files_sha256": TOKENIZER_FILES,
        "protected_manifest_sha256": PROTECTED_MANIFEST_SHA256,
        "protected_exact_matched_groups_by_role": exact_by_role,
        "selected_bounded_overlap_by_role": near_by_role,
        "selected_open_dev_bounded_overlap": open_dev_near,
        "shortcuts": shortcuts,
        "selected_shortcuts": selected_shortcuts,
        "gates": {
            "selected_class_floor_pass": selected_class_floor_pass,
            "selected_core_input_overlap_zero": not overlap_found,
            "selected_open_dev_input_overlap_zero": not dev_overlap_found,
            "whole_source_protected_exact_zero": not bool(protected_exact),
            "shortcut_review_clear": not any(
                item["shortcut_review_trigger"]
                for item in [*shortcuts.values(), *selected_shortcuts.values()]
            ),
            "neutral_rubric_blind_review": "PENDING",
            "original_corpus_and_rights_review": "PENDING",
        },
        "admission": "HOLD_NO_TRAINING_OR_REDISTRIBUTION",
        "limitation": "Bounded lexical overlap is not source/semantic independence; public ANLI dev is never an untouched release test.",
        "gpu_hours": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-directory", type=Path, required=True)
    parser.add_argument("--dev-directory", type=Path, required=True)
    parser.add_argument("--tokenizer-directory", type=Path, required=True)
    parser.add_argument("--protected-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.train_directory,
        args.dev_directory,
        args.tokenizer_directory,
        args.protected_inventory,
    )
    write_once(args.output, result)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "rows": result["source_profile"]["rows"],
                "candidate_rows": result["selection"]["selected_rows"],
                "admission": result["admission"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
