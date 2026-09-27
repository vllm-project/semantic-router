"""Prepare a label-blind review packet and verify Score v7p evidence necessity.

The generated packet is not itself an independent review. An independent
reviewer must return decisions before the A/B optimizer may start.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from training.data.score_v7p_build import _oracle_two
from training.model.data import file_sha256, load_partition

VERSION = "decision2-score-v7p-quality-packet/1"


def _groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    return dict(groups)


def _verdict(row: dict[str, Any], state: str) -> int:
    meta = row["audit_metadata"]
    return _oracle_two(state, meta["case_id"], meta["review_day"])


def _source_witnesses(row: dict[str, Any]) -> dict[str, bool]:
    if row["label"] != 2:
        raise ValueError("Source-necessity witnesses require the positive case")
    lines = row["state"].splitlines()
    if len(lines) != 3:
        raise ValueError("Expected the rendered procedure and two documents")
    case_id = row["audit_metadata"]["case_id"]
    bundle = json.loads(lines[1].split(": ", 1)[1])
    registry = json.loads(lines[2].split(": ", 1)[1])
    a_rows = [entry for entry in bundle["ledger"] if entry["case"] == case_id]
    b_rows = [entry for entry in registry if entry["case"] == case_id]
    vetoes = [
        notice
        for notice in bundle["notices"]
        if notice["state"] == "current" and notice["kind"] == "disqualifier"
    ]
    archived_notices = [
        notice for notice in bundle["notices"] if notice["state"] == "archived"
    ]
    if (
        len(a_rows) != 1
        or len(b_rows) != 1
        or len(vetoes) != 1
        or len(archived_notices) != 1
    ):
        raise ValueError("Cannot construct source-necessity perturbation")

    def rendered(new_bundle: dict[str, Any], new_registry: list[dict[str, Any]]) -> str:
        return "\n".join(
            (
                lines[0],
                lines[1].split(": ", 1)[0]
                + ": "
                + json.dumps(new_bundle, sort_keys=True),
                lines[2].split(": ", 1)[0]
                + ": "
                + json.dumps(new_registry, sort_keys=True),
            )
        )

    changed_a = copy.deepcopy(bundle)
    next(entry for entry in changed_a["ledger"] if entry["case"] == case_id)[
        "requirement_A"
    ] = "unresolved"
    changed_b = copy.deepcopy(registry)
    next(entry for entry in changed_b if entry["case"] == case_id)[
        "requirement_B"
    ] = "unresolved"
    changed_scope = copy.deepcopy(registry)
    next(entry for entry in changed_scope if entry["case"] == case_id)["scope"] = (
        vetoes[0]["scope"]
    )
    without_archived = copy.deepcopy(bundle)
    without_archived["notices"] = [
        notice
        for notice in without_archived["notices"]
        if notice["state"] != "archived"
    ]
    return {
        "document_a_required": _verdict(row, rendered(changed_a, registry)) == 1,
        "document_b_required": _verdict(row, rendered(bundle, changed_b)) == 1,
        "cross_document_veto_required": _verdict(row, rendered(bundle, changed_scope))
        == 0,
        "archived_control_invariant": _verdict(
            row, rendered(without_archived, registry)
        )
        == 2,
    }


def audit(rows: list[dict[str, Any]], role: str) -> dict[str, Any]:
    groups = _groups(rows)
    if len(rows) != 240 or len(groups) != 80:
        raise ValueError(f"{role}: 80 complete triplets are required")
    witnesses = collections.Counter()
    label_lengths: dict[int, list[int]] = collections.defaultdict(list)
    for group, triplet in groups.items():
        if len(triplet) != 3 or {row["label"] for row in triplet} != {0, 1, 2}:
            raise ValueError(f"{role}: incomplete group {group}")
        for row in triplet:
            if _verdict(row, row["state"]) != row["label"]:
                raise ValueError(f"{role}: row oracle mismatch")
            label_lengths[row["label"]].append(len(row["state"]))
        for name, passed in _source_witnesses(
            next(row for row in triplet if row["label"] == 2)
        ).items():
            witnesses[name] += passed
    if any(
        witnesses[name] != 80
        for name in (
            "document_a_required",
            "document_b_required",
            "cross_document_veto_required",
            "archived_control_invariant",
        )
    ):
        raise ValueError(f"{role}: evidence necessity or archived control failed")
    return {
        "role": role,
        "rows": 240,
        "independent_groups": 80,
        "source_necessity_witnesses": dict(witnesses),
        "state_char_range_by_level": {
            str(level): [min(values), max(values)]
            for level, values in label_lengths.items()
        },
        "limitation": "Programmatic witnesses check the rule; they do not replace independent blind review",
    }


def _blind_packet(
    rows: list[dict[str, Any]], role: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    packet, key = [], {}
    for group, triplet in sorted(_groups(rows).items()):
        # Neither exported group nor item ID contains the answer level.
        group_alias = hashlib.sha256(f"{VERSION}/{role}/{group}".encode()).hexdigest()[
            :16
        ]
        order = sorted(
            triplet,
            key=lambda row: hashlib.sha256(
                f"{VERSION}/{row['id']}".encode()
            ).hexdigest(),
        )
        items = []
        for index, row in enumerate(order):
            alias = f"{group_alias}-{index + 1}"
            items.append(
                {
                    "review_id": alias,
                    "state": row["state"],
                    "instructions": row["instructions"],
                    "options": row["options"],
                }
            )
            key[alias] = {
                "source_id": row["id"],
                "label": row["label"],
                "group_id": group,
            }
        packet.append({"review_group": group_alias, "items": items})
    return packet, key


def write(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError("Blind review packet output already exists")
    train = [
        row
        for row in load_partition(args.new_train, "train")
        if int(row["group_id"][-4:]) < 80
    ]
    select = load_partition(args.new_select, "select")
    report = {
        "schema_version": VERSION,
        "status": "PENDING_INDEPENDENT_BLIND_REVIEW",
        "source_sha256": {
            "new_train": file_sha256(args.new_train),
            "new_select": file_sha256(args.new_select),
        },
        "audit": {"train": audit(train, "train"), "select": audit(select, "select")},
    }
    args.output_dir.mkdir(parents=True)
    for role, rows in (("train", train), ("select", select)):
        packet, key = _blind_packet(rows, role)
        packet_path = args.output_dir / f"{role}-blind-packet.jsonl"
        with packet_path.open("x", encoding="utf-8") as output:
            for item in packet:
                output.write(
                    json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                )
            output.flush()
            os.fsync(output.fileno())
        packet_path.chmod(0o600)
        key_path = args.output_dir / f"{role}-sealed-key.json"
        with key_path.open("x", encoding="utf-8") as output:
            json.dump(key, output, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        key_path.chmod(0o600)
        report[f"{role}_packet_sha256"] = file_sha256(packet_path)
        report[f"{role}_key_sha256"] = file_sha256(key_path)
    report_path = args.output_dir / "quality-report.json"
    with report_path.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    report_path.chmod(0o600)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("new-train", "new-select", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    report = write(parser.parse_args())
    print(
        json.dumps(
            {"status": report["status"], "audit": report["audit"]}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
