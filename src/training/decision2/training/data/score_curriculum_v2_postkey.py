"""Verify sealed Score v2 blind review before a private aggregate key comparison.

The source IDs, labels, mapping and salt are opened only after all public
packet/review/supplement seals and their ordering have been checked. The
output contains aggregates and hashes, never row text or per-row keys.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

EXPECTED = {
    "train": "8fc287dc56281ae32a1d7428f019535fbc1b992d8e85a3c7fa1f6263663e6cee",
    "packet": "09a24684ccee0b07c0258bc37971ed7643a78b264fb7b3c3a2a436dec8bce11e",
    "manifest": "7eacf928d2b5250de2e205b3d91f03b6d0644f1c9f04a3b6a3785843981d40ba",
    "review": "807477cb2de3d03af9bacd72442a494f37750d1746810e43dd0b3a1d276db12c",
    "receipt": "25006a1ae7acaea22d6615cf2db4a07ba3f7ce52b939988fb82619bacaddd811",
    "supplement": "597733315cf348092844c976f455e3d8a3a197022f6b4f16b050a32153c6d492",
    "supplement_receipt": "4b4f7677ead0470b0adeb184225040cc1a269aacdc4f6ee288ea5a83b10bc9f4",
    "private_map": "69b770b7603df1dc0b04a3640b74719b44f23dc81bcc02648c5126215baa7fff",
    "private_salt": "d350df48f0d0f9df61996b4312234c39248d0a654f00f0b9dcbf366275c2c3ee",
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "train": args.train,
        "packet": args.packet,
        "manifest": args.manifest,
        "review": args.review,
        "receipt": args.receipt,
        "supplement": args.supplement,
        "supplement_receipt": args.supplement_receipt,
        "private_map": args.private_map,
        "private_salt": args.private_salt,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    for role in (
        "train",
        "packet",
        "manifest",
        "review",
        "receipt",
        "supplement",
        "supplement_receipt",
    ):
        _require(_hash(paths[role]) == EXPECTED[role], f"Frozen {role} SHA mismatch")

    # Stage one: verify the sealed, gold-free materials and chronology.
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    review = json.loads(args.review.read_text(encoding="utf-8"))
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    supplement = json.loads(args.supplement.read_text(encoding="utf-8"))
    supplement_receipt = json.loads(args.supplement_receipt.read_text(encoding="utf-8"))
    _require(
        manifest["packet_sha256"] == EXPECTED["packet"]
        and manifest["train_sha256"] == EXPECTED["train"]
        and manifest["private_map_sha256"] == EXPECTED["private_map"]
        and manifest["private_salt_sha256"] == EXPECTED["private_salt"],
        "Packet manifest does not bind frozen inputs",
    )
    _require(
        receipt["packet_sha256"] == EXPECTED["packet"]
        and receipt["manifest_sha256"] == EXPECTED["manifest"]
        and receipt["review_sha256"] == EXPECTED["review"]
        and receipt["row_count"] == 144
        and receipt["group_count"] == 48
        and receipt["overall_gate"] == "HOLD",
        "First blind seal does not bind its inputs and result",
    )
    _require(
        supplement_receipt["packet_sha256"] == EXPECTED["packet"]
        and supplement_receipt["manifest_sha256"] == EXPECTED["manifest"]
        and supplement_receipt["prior_review_sha256"] == EXPECTED["review"]
        and supplement_receipt["supplement_sha256"] == EXPECTED["supplement"]
        and supplement["prior_sealed_review_sha256"] == EXPECTED["review"],
        "Second blind seal does not bind the first",
    )
    allowed = {
        "packet.jsonl": EXPECTED["packet"],
        "manifest.json": EXPECTED["manifest"],
    }
    _require(
        review["allowed_source_sha256"] == allowed
        and supplement["allowed_source_sha256"] == allowed,
        "Reviewer sources include an unapproved file",
    )
    packet_time = args.packet.stat().st_mtime_ns
    review_time = args.review.stat().st_mtime_ns
    supplement_time = args.supplement.stat().st_mtime_ns
    _require(
        packet_time < review_time < supplement_time
        and args.receipt.stat().st_mtime_ns == review_time
        and args.supplement_receipt.stat().st_mtime_ns == supplement_time,
        "Packet, review and supplement filesystem ordering failed",
    )
    first_utc = datetime.fromisoformat(review["reviewed_utc"])
    second_utc = datetime.fromisoformat(supplement["reviewed_utc"])
    _require(first_utc < second_utc, "Blind review timestamps are not ordered")

    packet = _jsonl(args.packet)
    blind_rows = review["row_reviews"]
    _require(len(packet) == len(blind_rows) == 144, "Blind row count differs")
    _require(
        all(
            int(verdict["row_number"]) == index
            and verdict["review_id"] == packet[index - 1]["review_id"]
            and verdict["group_id"] == packet[index - 1]["group_id"]
            and verdict["family"] == packet[index - 1]["family"]
            and verdict["language"] == packet[index - 1]["language"]
            for index, verdict in enumerate(blind_rows, 1)
        ),
        "Blind review does not cover packet in order",
    )
    _require(
        len({row["review_id"] for row in packet}) == 144
        and len({row["group_id"] for row in packet}) == 48,
        "Packet alias or group coverage differs",
    )

    # Stage two: only now open private mapping and candidate labels.
    for role in ("private_map", "private_salt"):
        _require(_hash(paths[role]) == EXPECTED[role], f"Frozen {role} SHA mismatch")
    _require(
        len(args.private_salt.read_bytes()) == 32, "Private alias salt size differs"
    )
    mapping = _jsonl(args.private_map)
    train = _jsonl(args.train)
    _require(len(mapping) == 144 and len(train) == 960, "Private row count differs")
    by_alias = {row["review_id"]: row for row in mapping}
    by_source = {row["id"]: row for row in train}
    _require(
        len(by_alias) == 144
        and len(by_source) == 960
        and set(by_alias) == {row["review_id"] for row in packet},
        "Private join is not one-to-one",
    )
    by_family = collections.defaultdict(lambda: {"correct": 0, "total": 0})
    by_group: dict[str, set[int]] = collections.defaultdict(set)
    weighted_rule_correct = 0
    for prompt, verdict in zip(packet, blind_rows):
        source = by_source[by_alias[prompt["review_id"]]["source_id"]]
        _require(
            source["group_id"] == prompt["group_id"]
            and source["family"] == prompt["family"]
            and source["language"] == prompt["language"],
            "Private join changed packet lineage",
        )
        level = int(source["label"])
        inferred = int(verdict["inferred_level"])
        family = source["family"]
        by_family[family]["total"] += 1
        by_family[family]["correct"] += inferred == level
        by_group[prompt["group_id"]].add(level)
        if family == "score_weighted_points":
            unweighted = sum(item["mark"] for item in prompt["state"]["signals"])
            shortcut_level = 0 if unweighted < 5 else (1 if unweighted < 8 else 2)
            weighted_rule_correct += shortcut_level == level
    _require(
        len(by_group) == 48
        and all(levels == {0, 1, 2} for levels in by_group.values()),
        "Private labels do not span three levels per group",
    )
    reviewer_shortcut = review["shortcut_audit"]["weighted_unweighted_mark_sum"]
    _require(
        weighted_rule_correct == reviewer_shortcut["correct_of_36"] == 30
        and reviewer_shortcut["leave_one_group_out_threshold_correct_of_36"] == 28,
        "Blind shortcut counts fail private confirmation",
    )
    report = {
        "schema_version": "decision20-score-v2-postkey-aggregate/1",
        "status": "HOLD_BLOCK_FOR_TRAINING",
        "input_sha256": dict(EXPECTED),
        "sealed_order_verified": True,
        "reviewed_utc": [first_utc.isoformat(), second_utc.isoformat()],
        "rows": len(packet),
        "groups": len(by_group),
        "by_family": dict(sorted(by_family.items())),
        "weighted_unweighted_mark_sum": {
            "global_threshold_correct": weighted_rule_correct,
            "global_threshold_total": 36,
            "blind_leave_one_group_out_correct": 28,
            "blind_leave_one_group_out_total": 36,
        },
        "no_row_text_or_per_row_labels": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in EXPECTED:
        parser.add_argument(f"--{name.replace('_', '-')}", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    result = audit(parser.parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "rows": result["rows"],
                "groups": result["groups"],
                "by_family": result["by_family"],
                "weighted_shortcut_correct": result["weighted_unweighted_mark_sum"][
                    "global_threshold_correct"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
