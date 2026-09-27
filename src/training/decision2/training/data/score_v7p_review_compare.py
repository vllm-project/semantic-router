"""Compare a sealed independent v7p blind review with separately sealed keys.

This post-review step may read keys. It never changes the training candidate,
review receipt, or the preregistered realism gate; it writes a private audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from training.model.data import file_sha256

SCHEMA = "decision2-score-v7p-blind-key-comparison/1"
ROLES = ("train", "select")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}: expected an object")
    return value


def _packet(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    items, groups = {}, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        group = json.loads(line)
        alias = group["review_group"]
        if alias in groups or len(group["items"]) != 3:
            raise ValueError("Blind packet has duplicate/incomplete groups")
        groups[alias] = []
        for item in group["items"]:
            review_id = item["review_id"]
            if review_id in items:
                raise ValueError("Blind packet has duplicate items")
            items[review_id] = item
            groups[alias].append(review_id)
    if len(groups) != 80 or len(items) != 240:
        raise ValueError("Blind packet cardinality differs")
    return items, groups


def compare(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError("Blind-key comparison receipt already exists")
    if file_sha256(args.review) != args.review_sha256:
        raise ValueError("Independent sealed-review receipt SHA differs")
    if file_sha256(args.manual_spot) != args.manual_spot_sha256:
        raise ValueError("Sealed manual spot receipt SHA differs")
    quality = _read_json(args.quality_report)
    review = _read_json(args.review)
    manual = _read_json(args.manual_spot)
    if (
        quality.get("schema_version") != "decision2-score-v7p-quality-packet/1"
        or review.get("schema") != "decision2-score-v7p-independent-blind-review/1"
        or manual.get("schema") != "decision2-score-v7p-manual-spot/1"
        or manual.get("primary_receipt_sha256") != args.review_sha256
        or review.get("reviewer_script_sha256") != file_sha256(args.reviewer_script)
        or review.get("decision") != "HOLD_REALISM_REVIEW"
    ):
        raise ValueError("Review lineage or prospectively recorded status differs")
    packets, keys, packet_groups = {}, {}, {}
    for role in ROLES:
        packet_file = args.packet_dir / f"{role}-blind-packet.jsonl"
        key_file = args.packet_dir / f"{role}-sealed-key.json"
        if (
            file_sha256(packet_file) != quality[f"{role}_packet_sha256"]
            or file_sha256(key_file) != quality[f"{role}_key_sha256"]
            or review.get("input_sha256", {}).get(role) != file_sha256(packet_file)
        ):
            raise ValueError(f"{role}: blind packet or key changed after review")
        packets[role], packet_groups[role] = _packet(packet_file)
        keys[role] = _read_json(key_file)
        if set(keys[role]) != set(packets[role]):
            raise ValueError(f"{role}: key and packet IDs differ")
    if len(review.get("items", [])) != 480 or len(review.get("groups", [])) != 160:
        raise ValueError("Independent reviewer did not cover all fixed items/groups")
    seen_items, mismatches, item_flags = set(), [], []
    for item in review["items"]:
        role, review_id = item["split"], item["review_id"]
        if (
            role not in ROLES
            or review_id not in packets[role]
            or (role, review_id) in seen_items
        ):
            raise ValueError("Independent review has duplicate or foreign item")
        seen_items.add((role, review_id))
        original = packets[role][review_id]
        expected_text_sha = hashlib.sha256(
            json.dumps(original, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if item.get("blind_text_sha256") != expected_text_sha:
            raise ValueError("Independent review item text differs from sealed packet")
        if (
            item["group"] not in packet_groups[role]
            or review_id not in packet_groups[role][item["group"]]
        ):
            raise ValueError("Independent review item/group mapping differs")
        expected = keys[role][review_id]["label"]
        answer = item.get("answer")
        if type(expected) is not int or answer not in {"0", "1", "2"}:
            raise ValueError("Malformed sealed key or reviewer answer")
        if int(answer) != expected:
            mismatches.append(
                {
                    "split": role,
                    "review_id": review_id,
                    "reviewed": answer,
                    "key": expected,
                }
            )
        if (
            item.get("ambiguous") is not False
            or item.get("evidence_sufficient") is not True
        ):
            item_flags.append({"split": role, "review_id": review_id})
    if len(seen_items) != 480:
        raise ValueError("Independent review item coverage differs")
    seen_groups, realism = set(), []
    for group in review["groups"]:
        role, alias = group["split"], group["group"]
        if (
            role not in ROLES
            or alias not in packet_groups[role]
            or (role, alias) in seen_groups
        ):
            raise ValueError("Independent review has duplicate/foreign group")
        seen_groups.add((role, alias))
        if group.get("answerable") is not True or group.get("ambiguous") is not False:
            item_flags.append({"split": role, "review_group": alias})
        if group.get("realism") != "adequate":
            realism.append(
                {"split": role, "review_group": alias, "finding": group.get("realism")}
            )
        item_answers = {
            item["review_id"]: item["answer"]
            for item in review["items"]
            if item["split"] == role and item["group"] == alias
        }
        ordered = [item_answers[review_id] for review_id in packet_groups[role][alias]]
        if group.get("answers_in_packet_order") != ordered:
            raise ValueError("Reviewer item/group answer order differs")
    if len(seen_groups) != 160:
        raise ValueError("Independent review group coverage differs")
    manual_checked = 0
    for alias, answers in manual.get("groups", {}).items():
        matching = [
            (role, group)
            for role, groups in packet_groups.items()
            for group in groups
            if group == alias
        ]
        if len(matching) != 1:
            raise ValueError("Manual spot group not in blind packet")
        role, _ = matching[0]
        reviewed = next(
            group
            for group in review["groups"]
            if group["split"] == role and group["group"] == alias
        )
        if answers != reviewed["answers_in_packet_order"]:
            raise ValueError("Manual spot answers disagree with sealed review")
        manual_checked += len(answers)
    if manual_checked != manual.get("item_count") or manual_checked != 24:
        raise ValueError("Manual spot coverage differs")
    status = (
        "HOLD_ANSWER_MISMATCH"
        if mismatches
        else "HOLD_AMBIGUITY" if item_flags else "HOLD_REALISM" if realism else "PASS"
    )
    result = {
        "schema_version": SCHEMA,
        "status": status,
        "review_sha256": args.review_sha256,
        "reviewer_script_sha256": file_sha256(args.reviewer_script),
        "manual_spot_sha256": args.manual_spot_sha256,
        "quality_report_sha256": file_sha256(args.quality_report),
        "packet_sha256": {
            role: file_sha256(args.packet_dir / f"{role}-blind-packet.jsonl")
            for role in ROLES
        },
        "key_sha256": {
            role: file_sha256(args.packet_dir / f"{role}-sealed-key.json")
            for role in ROLES
        },
        "item_count": len(seen_items),
        "group_count": len(seen_groups),
        "answer_mismatch_count": len(mismatches),
        "answer_mismatches": mismatches,
        "ambiguous_or_unsupported_count": len(item_flags),
        "ambiguous_or_unsupported": item_flags,
        "realism_concern_groups": len(realism),
        "realism_concern_examples": realism[:12],
        "manual_spot_items": manual_checked,
        "decision": "No optimizer authorization unless status is PASS; preserve this v7p outcome",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, sort_keys=True, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    args.output.chmod(0o600)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "packet-dir",
        "quality-report",
        "review",
        "manual-spot",
        "reviewer-script",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--manual-spot-sha256", required=True)
    result = compare(parser.parse_args())
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "item_count",
                    "group_count",
                    "answer_mismatch_count",
                    "ambiguous_or_unsupported_count",
                    "realism_concern_groups",
                    "manual_spot_items",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
