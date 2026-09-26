"""Validate the sealed Score v3 blind review before opening the private join.

The output is a private aggregate. Never write row text, row IDs, source IDs,
per-row gold, or the alias salt to it.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED = {
    "train": "c4ea3294247022fdf2359e6ed74c0abec0bc6de0295fad06af6271c37a9d0fb7",
    "candidate_manifest": "5ab30da57bd93f7e2045c68d68e607bd5d9fca62aa987d3594ae07f0311c89a7",
    "packet": "3e63de13630ce6833417e5a6f23d0aa011cb8b3e3dd123faa07a7062334df9e0",
    "packet_manifest": "ff9f1655e49325b04ef08dbffb4c4670861c2615a490093336f7ba4925613c8c",
    "derive": "1521b61701cc6700c4b8955675ade75f21b583f0521240ba204bc751f996809e",
    "row_judgments": "d200089e8a0890368b9878d550859682a23aa2527a1f5d25513ef6b5d336142c",
    "group_judgments": "07a9f9020c70cf672bd8abd3f2f7c0e1aa5eacf5b9704599f53f4121ad6c92bf",
    "summary": "1df22660630b31b8202720feab6578f5420248a9d70c8ef801bd9e234b40546d",
    "seal": "12faccf4209abdbd7a03be6685b9c95b7760e198ecdddf160a7d9efa8b0a59fb",
    "private_map": "49a8098abc076e5a583acb848d054305527fa51a14db26fc3f9f7be8ba579c7a",
    "private_salt": "4e26b885bb8859e5bd9cd2b2494c116a3b7dc853a0927931d32e0f21cbe213f5",
}
FAMILIES = (
    "score_obligation_review",
    "score_weighted_points",
    "score_route_depth",
    "score_timely_streak",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def verify_public_seal(args: argparse.Namespace) -> dict[str, Any]:
    """Check every gold-free commitment and chronology; do not touch the key."""
    for role in (
        "train",
        "candidate_manifest",
        "packet",
        "packet_manifest",
        "derive",
        "row_judgments",
        "group_judgments",
        "summary",
        "seal",
    ):
        _require(
            _sha(getattr(args, role)) == EXPECTED[role], f"Frozen {role} SHA mismatch"
        )

    manifest = json.loads(args.packet_manifest.read_text(encoding="utf-8"))
    candidate_manifest = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    seal = json.loads(args.seal.read_text(encoding="utf-8"))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    packet = _jsonl(args.packet)
    rows = _jsonl(args.row_judgments)
    groups = _jsonl(args.group_judgments)

    _require(
        manifest["train_sha256"] == EXPECTED["train"]
        and manifest["packet_sha256"] == EXPECTED["packet"]
        and manifest["private_map_sha256"] == EXPECTED["private_map"]
        and manifest["private_salt_sha256"] == EXPECTED["private_salt"]
        and candidate_manifest["outputs"]["score_curriculum.train.jsonl"]["sha256"]
        == EXPECTED["train"],
        "Packet and TRAIN manifests do not bind the frozen candidate",
    )
    _require(
        seal["source_packet_sha256"] == EXPECTED["packet"]
        and seal["source_manifest_sha256"] == EXPECTED["packet_manifest"]
        and seal["editorial_decision"] == "BLOCK_FOR_TRAINING"
        and seal["private_file_sha256"]
        == {
            "derive.py": EXPECTED["derive"],
            "row_judgments.jsonl": EXPECTED["row_judgments"],
            "group_judgments.jsonl": EXPECTED["group_judgments"],
            "summary.json": EXPECTED["summary"],
        }
        and summary["source_packet_sha256"] == EXPECTED["packet"]
        and summary["source_manifest_sha256"] == EXPECTED["packet_manifest"]
        and summary["editorial_decision"] == "BLOCK_FOR_TRAINING",
        "Blind seal does not bind the packet, verdict and review files",
    )
    source_mtime = max(
        args.packet.stat().st_mtime_ns, args.packet_manifest.stat().st_mtime_ns
    )
    derive_mtime = args.derive.stat().st_mtime_ns
    judgment_mtimes = [
        args.row_judgments.stat().st_mtime_ns,
        args.group_judgments.stat().st_mtime_ns,
        args.summary.stat().st_mtime_ns,
    ]
    seal_mtime = args.seal.stat().st_mtime_ns
    sealed_at = datetime.fromisoformat(seal["sealed_at_utc"])
    _require(
        source_mtime < derive_mtime <= min(judgment_mtimes)
        and max(judgment_mtimes) < seal_mtime
        and sealed_at.tzinfo is not None
        and int(sealed_at.timestamp())
        == int(datetime.fromtimestamp(seal_mtime / 1e9, tz=timezone.utc).timestamp()),
        "Packet, review and seal chronology failed",
    )

    _require(
        len(packet) == len(rows) == 144 and len(groups) == 48, "Blind counts differ"
    )
    _require(
        len({row["review_id"] for row in packet}) == 144
        and len({row["group_id"] for row in packet}) == 48,
        "Packet aliases are not unique",
    )
    for index, (prompt, judgment) in enumerate(zip(packet, rows), 1):
        _require(
            judgment["row_index"] == index
            and judgment["review_id"] == prompt["review_id"]
            and judgment["group_id"] == prompt["group_id"]
            and judgment["family"] == prompt["family"]
            and judgment["language"] == prompt["language"]
            and judgment["derived_level"] in (0, 1, 2),
            "Blind row judgment does not match packet order and aliases",
        )
    packet_by_group: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    rows_by_group: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for prompt in packet:
        packet_by_group[prompt["group_id"]].append(prompt)
    for row in rows:
        rows_by_group[row["group_id"]].append(row)
    _require(
        len({group["group_id"] for group in groups}) == 48
        and {group["group_id"] for group in groups} == set(packet_by_group),
        "Blind group judgments do not cover all source groups",
    )
    for group in groups:
        alias = group["group_id"]
        prompts = packet_by_group[alias]
        judgments = rows_by_group[alias]
        _require(
            len(prompts) == len(judgments) == 3
            and {row["derived_level"] for row in judgments} == {0, 1, 2}
            and len({row["family"] for row in prompts}) == 1
            and len({row["language"] for row in prompts}) == 1
            and group["family"] == prompts[0]["family"]
            and group["language"] == prompts[0]["language"]
            and group["case"] == prompts[0]["state"]["case"]
            and group["coherent_triplet"]
            and not group["ambiguity"]
            and group["review_ids_by_level"]
            == {str(row["derived_level"]): row["review_id"] for row in judgments},
            "Blind group judgment or triplet differs from its packet rows",
        )
    flags = collections.Counter(
        flag for group in groups for flag in group["shortcut_flags"]
    )
    _require(
        summary["row_count"] == 144
        and summary["group_count"] == 48
        and summary["derived_level_counts"]
        == dict(collections.Counter(str(row["derived_level"]) for row in rows))
        and summary["family_row_counts"]
        == dict(collections.Counter(row["family"] for row in rows))
        and summary["group_shortcut_flag_counts"] == dict(flags)
        and summary["ambiguous_rows"] == sum(bool(row["ambiguity"]) for row in rows)
        and summary["ambiguous_groups"]
        == sum(bool(group["ambiguity"]) for group in groups)
        and summary["incoherent_triplets"]
        == sum(not group["coherent_triplet"] for group in groups),
        "Blind summary does not match sealed row/group judgments",
    )
    return {
        "packet": packet,
        "rows": rows,
        "groups": groups,
        "flags": dict(flags),
        "sealed_at_utc": sealed_at.isoformat(),
    }


def _shortcut_counts(source_groups: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    counts = collections.Counter()
    for variants in source_groups.values():
        by_level = {row["label"]: row["state"] for row in variants}
        family = variants[0]["family"]
        if family == "score_obligation_review":
            accepted = [
                sum(
                    review["scope"] == "core" and review["assessment"] == "accepted"
                    for review in by_level[level]["reviews"]
                )
                for level in range(3)
            ]
            counts["accepted_core_count_exact_proxy"] += accepted == [2, 3, 4]
        elif family == "score_timely_streak":
            adjacent = []
            for level in range(3):
                days = {day["day"]: day["on_time"] for day in by_level[level]["days"]}
                adjacent.append(
                    sum(days[index] and days[index + 1] for index in range(1, 8))
                )
            counts["adjacent_pair_count_exact_proxy"] += (
                adjacent[0] == 0 and adjacent[1] in (1, 2) and adjacent[2] == 3
            )
        elif family == "score_weighted_points":
            signals = [by_level[level]["signals"] for level in range(3)]
            counts["local_single_signal_rank_proxy"] += any(
                signals[0][index]["mark"]
                < signals[1][index]["mark"]
                < signals[2][index]["mark"]
                for index in range(len(signals[0]))
            )
    return dict(sorted(counts.items()))


def audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)

    # The private map, salt and gold are untouched until this completes.
    public = verify_public_seal(args)

    for role in ("private_map", "private_salt"):
        _require(
            _sha(getattr(args, role)) == EXPECTED[role], f"Frozen {role} SHA mismatch"
        )
    _require(len(args.private_salt.read_bytes()) == 32, "Private salt size differs")
    mapping = _jsonl(args.private_map)
    train = _jsonl(args.train)
    _require(len(mapping) == 144 and len(train) == 960, "Private corpus count differs")
    by_alias = {row["review_id"]: row for row in mapping}
    by_source = {row["id"]: row for row in train}
    _require(
        len(by_alias) == 144
        and len(by_source) == 960
        and set(by_alias) == {row["review_id"] for row in public["packet"]},
        "Private join is not one-to-one with the packet",
    )
    family_correct = collections.defaultdict(lambda: {"correct": 0, "total": 0})
    source_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    selected_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for prompt, judgment in zip(public["packet"], public["rows"]):
        joined = by_alias[prompt["review_id"]]
        source = by_source[joined["source_id"]]
        _require(
            joined["group_id"] == prompt["group_id"]
            and source["group_id"] == joined["source_group_id"]
            and source["family"] == prompt["family"]
            and source["language"] == prompt["language"]
            and all(
                source[field] == prompt[field]
                for field in ("state", "instructions", "options")
            ),
            "Private join changed a sealed model-facing input",
        )
        family_correct[source["family"]]["total"] += 1
        family_correct[source["family"]]["correct"] += (
            source["label"] == judgment["derived_level"]
        )
        selected_groups[joined["source_group_id"]].append(source)
    for source in train:
        source_groups[source["group_id"]].append(source)
    _require(
        len(selected_groups) == 48
        and all(
            len(group) == 3 and {row["label"] for row in group} == {0, 1, 2}
            for group in selected_groups.values()
        ),
        "Private selected groups are not complete three-level triplets",
    )
    _require(
        len(source_groups) == 320
        and all(
            len(group) == 3 and {row["label"] for row in group} == {0, 1, 2}
            for group in source_groups.values()
        ),
        "Private TRAIN groups are not complete three-level triplets",
    )
    report = {
        "schema_version": "decision20-score-v3-postkey-aggregate/1",
        "status": "HOLD_BLOCK_FOR_TRAINING",
        "input_sha256": dict(EXPECTED),
        "sealed_order_verified": True,
        "sealed_at_utc": public["sealed_at_utc"],
        "rows": len(public["packet"]),
        "groups": len(selected_groups),
        "by_family": dict(sorted(family_correct.items())),
        "reviewer_shortcut_flag_groups": public["flags"],
        "independent_shortcut_groups_in_packet": _shortcut_counts(selected_groups),
        "independent_shortcut_groups_in_full_candidate": _shortcut_counts(
            source_groups
        ),
        "no_row_text_ids_or_labels": True,
    }
    args.output.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    _require(
        args.output.parent.stat().st_mode & 0o077 == 0,
        "Private aggregate directory is accessible to other users",
    )
    descriptor = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for role in EXPECTED:
        parser.add_argument(f"--{role.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = audit(parser.parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "rows": result["rows"],
                "groups": result["groups"],
                "by_family": result["by_family"],
                "shortcuts": result["independent_shortcut_groups_in_packet"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
