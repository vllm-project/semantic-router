"""Audit and assemble matched, private Score v7p A/B/C pilot arms.

The protected inventory accepts only gold-free prompt files. It is an
exclusion input, never a source of labels or training text. Outputs remain
private candidate arms until the separate blind quality gate is signed.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from training.data import build_pilot as pilot
from training.data import build_targeted_candidate as targeted
from training.data.score_v7p_build import _oracle_two
from training.model.data import (
    check_partition_isolation,
    file_sha256,
    load_partition,
)
from training.model.decision_model import encode
from training.model.infer import load_prompts

SCHEMA = "decision2-score-v7p-matched-arms/1"
PARENT_SHA = "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755"
SELECT_SHA = "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6"
CAL_SHA = "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a"
MAX_LENGTH = 1024
SCORE_ROWS = 240
REPLAY_PER_TYPE = 1024
REQUIRED_PROTECTED = {
    "typed_dev",
    "css_pilot",
    "typed_final_goldfree",
    "css15_goldfree",
    "jevbench_public231",
}


def _groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        result[row["group_id"]].append(row)
    return dict(result)


def _flat(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [row for group in sorted(groups) for row in groups[group]]


def _new_groups(rows: list[dict[str, Any]], role: str, count: int) -> None:
    groups = _groups(rows)
    if len(rows) != count * 3 or len(groups) != count:
        raise ValueError(f"{role}: missing group-complete triplets")
    for index, (group, triplet) in enumerate(sorted(groups.items())):
        if group != f"d2score_v7p_{role}_{index:04d}":
            raise ValueError(f"{role}: group sequence changed")
        if {row["label"] for row in triplet} != {0, 1, 2}:
            raise ValueError(f"{role}: group levels changed")
        cases = {
            (row["audit_metadata"]["case_id"], row["audit_metadata"]["review_day"])
            for row in triplet
        }
        if len(cases) != 1:
            raise ValueError(f"{role}: triplet entity or review day changed")
        for row in triplet:
            metadata = row["audit_metadata"]
            if row["task_type"] != "score" or row["language"] != "en":
                raise ValueError(f"{role}: wrong type or language")
            if row["source"] != "decision2-score-v7p-evidence-state/1":
                raise ValueError(f"{role}: wrong source")
            try:
                rendered_level = _oracle_two(
                    row["state"], metadata["case_id"], metadata["review_day"]
                )
            except ValueError as exc:
                raise ValueError(
                    f"{role}: rendered document cannot be verified"
                ) from exc
            if rendered_level != row["label"]:
                raise ValueError(f"{role}: rendered oracle does not support label")


def _protected(
    path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    inventory = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(inventory, list):
        raise ValueError("Protected inventory must be a list")
    roles = [entry["role"] for entry in inventory]
    if len(roles) != len(set(roles)) or not set(roles).issuperset(REQUIRED_PROTECTED):
        raise ValueError("Protected inventory is incomplete or has duplicate roles")
    references = {}
    receipts = []
    for entry in inventory:
        file = Path(entry["path"])
        if not file.name.endswith(("prompts.jsonl", "packet.jsonl")):
            raise ValueError(
                f"{entry['role']}: protected input must be gold-free prompts"
            )
        actual = file_sha256(file)
        if actual != entry["sha256"]:
            raise ValueError(f"{entry['role']}: protected prompt SHA differs")
        prompts = load_prompts(file)
        rows = [
            {
                "id": item["id"],
                "group_id": None,
                "state": item["state"],
                "instructions": pilot.canonical(item["questions"]),
                "options": [],
                "task_type": "context",
                "input_sha256": None,
            }
            for item in prompts
        ]
        references[entry["role"]] = rows
        receipts.append({"role": entry["role"], "sha256": actual, "rows": len(rows)})
    return references, sorted(receipts, key=lambda item: item["role"])


def _overlap(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    ids = {row["id"] for row in right}
    groups = {row.get("group_id") for row in right}
    inputs = {row.get("input_sha256") for row in right}
    states = {targeted.text_hashes(row["state"]) for row in right}
    exact = [
        row["id"]
        for row in left
        if row["id"] in ids
        or row["group_id"] in groups
        or row["input_sha256"] in inputs
        or targeted.text_hashes(row["state"]) in states
    ]
    near_context = pilot.near_duplicates(
        targeted.context_rows(left), targeted.context_rows(right), collect_left_ids=True
    )
    near_full = pilot.near_duplicates(left, right, collect_left_ids=True)
    matched = (
        set(exact) | set(near_context.pop("left_ids")) | set(near_full.pop("left_ids"))
    )
    return {
        "matched_rows": len(matched),
        "matched_groups": len(
            {row["group_id"] for row in left if row["id"] in matched}
        ),
        "near_context": near_context,
        "near_full": near_full,
    }


def _tokens(rows: list[dict[str, Any]], tokenizer: Any) -> dict[str, int]:
    result = {}
    for row in rows:
        try:
            result[row["id"]] = len(encode(row, tokenizer, MAX_LENGTH)["ids"])
        except ValueError as exc:
            if "exceeds max_length" not in str(exc):
                raise
    return result


def _score_control(
    parent: list[dict[str, Any]], lengths: dict[str, int], target: int
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in parent
        if row["task_type"] == "score"
        and row["language"] == "en"
        and row["id"] in lengths
    ]
    by_group = _groups(eligible)
    # A partial legacy group is excluded, even when its surviving row is short.
    full_sizes = collections.Counter(row["group_id"] for row in parent)
    by_group = {
        key: value for key, value in by_group.items() if len(value) == full_sizes[key]
    }
    singles = [value for value in by_group.values() if len(value) == 1]
    doubles = [value for value in by_group.values() if len(value) == 2]
    if len(singles) + 2 * len(doubles) < SCORE_ROWS:
        raise ValueError("Too few eligible, whole parent Score groups")
    score = lambda group: sum(lengths[row["id"]] for row in group)
    singles.sort(key=lambda group: (-score(group), group[0]["group_id"]))
    doubles.sort(key=lambda group: (-score(group), group[0]["group_id"]))
    best: tuple[int, list[list[dict[str, Any]]]] | None = None
    for count_two in range(len(doubles) + 1):
        count_one = SCORE_ROWS - 2 * count_two
        if not 0 <= count_one <= len(singles):
            continue
        selected = doubles[:count_two] + singles[:count_one]
        distance = abs(sum(map(score, selected)) - target)
        if best is None or distance < best[0]:
            best = (distance, selected)
    if best is None:
        raise ValueError("Cannot select exactly 240 whole Score rows")
    if best[0] > target * 0.01:
        raise ValueError("No top-length whole-group Score control meets 1% budget")
    return [row for group in best[1] for row in group]


def _replay(
    parent: list[dict[str, Any]], lengths: dict[str, int]
) -> list[dict[str, Any]]:
    by_group = _groups(parent)
    selected = []
    for kind in ("choice", "noul"):
        eligible = []
        for rows in by_group.values():
            # A parent group may also have other languages or task types in
            # TRAIN. Preserve the entire eligible English projection, rather
            # than require the irrelevant rows to enter this English pilot.
            projection = [
                row
                for row in rows
                if row["task_type"] == kind and row["language"] == "en"
            ]
            if projection and all(row["id"] in lengths for row in projection):
                eligible.append(projection)
        eligible.sort(
            key=lambda rows: hashlib.sha256(
                f"v7p-replay/1/{rows[0]['group_id']}".encode()
            ).hexdigest()
        )
        # Exact cardinality with complete source groups. The frozen hash order
        # is independent of token length, labels, SELECT, and model outcomes.
        mask = (1 << (REPLAY_PER_TYPE + 1)) - 1
        reachable = 1
        snapshots = [reachable]
        for group in eligible:
            reachable = (reachable | (reachable << len(group))) & mask
            snapshots.append(reachable)
        if not (reachable >> REPLAY_PER_TYPE) & 1:
            raise ValueError(f"Cannot select 1024 complete {kind} replay rows")
        remaining = REPLAY_PER_TYPE
        chosen: list[list[dict[str, Any]]] = []
        for index in range(len(eligible) - 1, -1, -1):
            if (snapshots[index] >> remaining) & 1:
                continue
            group = eligible[index]
            chosen.append(group)
            remaining -= len(group)
        if remaining:
            raise AssertionError("Group subset reconstruction failed")
        selected.extend(row for group in reversed(chosen) for row in group)
    return selected


def _exposure(rows: list[dict[str, Any]], lengths: dict[str, int]) -> dict[str, int]:
    raw = sum(lengths[row["id"]] for row in rows)
    padded = sum((lengths[row["id"]] + 7) // 8 * 8 for row in rows)
    return {
        "rows": len(rows),
        "raw_native_tokens": raw,
        "padded_native_tokens": padded,
        "max_native_tokens": max(lengths[row["id"]] for row in rows),
    }


def _write(path: Path, rows: list[dict[str, Any]]) -> str:
    with path.open("x", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    path.chmod(0o600)
    return file_sha256(path)


def admit(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if args.output_dir.exists():
        raise FileExistsError("Candidate output directory already exists")
    pinned = {
        "parent_train": (args.parent_train, PARENT_SHA),
        "parent_select": (args.parent_select, SELECT_SHA),
        "parent_cal": (args.parent_cal, CAL_SHA),
    }
    for name, (path, digest) in pinned.items():
        if file_sha256(path) != digest:
            raise ValueError(f"{name} identity changed")
    parent = load_partition(args.parent_train, "train")
    select = load_partition(args.parent_select, "select")
    cal = load_partition(args.parent_cal, "cal")
    new_train = load_partition(args.new_train, "train")
    new_select = load_partition(args.new_select, "select")
    if (len(parent), len(select), len(cal)) != (7455, 700, 700):
        raise ValueError("Parent cardinality changed")
    _new_groups(new_train, "train", 120)
    _new_groups(new_select, "select", 80)
    check_partition_isolation(
        {"train": [*parent, *new_train], "select": [*select, *new_select], "cal": cal}
    )
    references, protected_receipts = _protected(args.protected_list)
    overlaps = {}
    for name, rows in {
        "parent_select": select,
        "parent_cal": cal,
        "new_select": new_select,
        **references,
    }.items():
        finding = _overlap(new_train, rows)
        overlaps[name] = finding
        if finding["matched_rows"]:
            raise ValueError(
                f"New TRAIN overlap with {name}: {finding['matched_groups']} groups"
            )
    for name, rows in references.items():
        finding = _overlap(new_select, rows)
        overlaps[f"new_select_vs_{name}"] = finding
        if finding["matched_rows"]:
            raise ValueError(
                f"New SELECT overlap with {name}: {finding['matched_groups']} groups"
            )
    tokenizer = AutoTokenizer.from_pretrained(
        args.source_tokenizer, local_files_only=True
    )
    new_lengths = _tokens([*new_train, *new_select], tokenizer)
    if len(new_lengths) != 600:
        raise ValueError("A new Score item exceeds 1024 native tokens")
    parent_lengths = _tokens(parent, tokenizer)
    train_groups = _groups(new_train)
    chosen = _flat(
        {key: value for key, value in train_groups.items() if int(key[-4:]) < 80}
    )
    if len(chosen) != SCORE_ROWS:
        raise ValueError("A TRAIN is not 80 whole new groups")
    score_target = sum(new_lengths[row["id"]] for row in chosen)
    control = _score_control(parent, parent_lengths, score_target)
    replay = _replay(parent, parent_lengths)
    if {row["group_id"] for row in control} & {row["group_id"] for row in replay}:
        raise ValueError("Parent Score and replay groups overlap")
    arm_a = [*chosen, *replay]
    arm_b = [*control, *replay]
    combined_lengths = {**parent_lengths, **new_lengths}
    exposure_a = _exposure(arm_a, combined_lengths)
    exposure_b = _exposure(arm_b, combined_lengths)
    if exposure_a["rows"] != 2288 or exposure_b["rows"] != 2288:
        raise ValueError("Matched arm cardinality differs")
    if (
        abs(exposure_a["raw_native_tokens"] - exposure_b["raw_native_tokens"])
        > exposure_a["raw_native_tokens"] * 0.01
    ):
        raise ValueError("A/B native token budgets differ by >1%")
    if (
        abs(exposure_a["padded_native_tokens"] - exposure_b["padded_native_tokens"])
        > exposure_a["padded_native_tokens"] * 0.05
    ):
        raise ValueError("A/B padded token budgets differ by >5%")
    output = args.output_dir
    output.mkdir(parents=True)
    hashes = {
        "A": _write(output / "arm-A.jsonl", arm_a),
        "B": _write(output / "arm-B.jsonl", arm_b),
        "C": _write(output / "arm-C.jsonl", arm_a),
    }
    manifest = {
        "schema_version": SCHEMA,
        "status": "CANDIDATE_PENDING_BLIND_QA_AND_ZERO_STEP",
        "parent_sha256": {name: digest for name, (_, digest) in pinned.items()},
        "new_train_sha256": file_sha256(args.new_train),
        "new_select_sha256": file_sha256(args.new_select),
        "source_tokenizer_sha256": file_sha256(
            args.source_tokenizer / "tokenizer.json"
        ),
        "protected_inventory_sha256": file_sha256(args.protected_list),
        "protected": protected_receipts,
        "overlap": overlaps,
        "arm_sha256": hashes,
        "arm_exposure": {"A": exposure_a, "B": exposure_b, "C": exposure_a},
        "score_group_count": {
            "A": 80,
            "B": len(_groups(control)),
            "selector": 80,
            "buffer_unused": 40,
        },
        "score_rows": {"A": 240, "B": 240, "selector": 240},
        "replay_rows_per_type": REPLAY_PER_TYPE,
        "score_level_counts": dict(collections.Counter(row["label"] for row in chosen)),
        "selected_ids_sha256": {
            name: hashlib.sha256(
                pilot.canonical([row["id"] for row in rows]).encode()
            ).hexdigest()
            for name, rows in (("A", arm_a), ("B", arm_b))
        },
        "limitations": [
            "One synthetic template family; no independent blind review yet",
            "Approximate SimHash near matching is not a complete semantic proof",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "manifest.json").chmod(0o600)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "parent-train",
        "parent-select",
        "parent-cal",
        "new-train",
        "new-select",
        "protected-list",
        "source-tokenizer",
        "output-dir",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    result = admit(parser.parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "arm_sha256": result["arm_sha256"],
                "exposure": result["arm_exposure"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
