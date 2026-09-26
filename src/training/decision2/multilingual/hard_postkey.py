"""Seal-first private aggregate for the frozen multilingual hard r6 review.

The blind files and packet are validated, and a receipt is written, before
this tool opens any target. Neither output contains row text or answer keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

EXPECTED = {
    "packet": "9b16c9ba50f51c52108732f0fdb3221580f892758e855f410839503160087824",
    "row": "1bf6b4429177fd76df00364c962811b6f985d2c11675e9d08633cc05e215c61b",
    "group": "4d52a64506a8bd7731780f57d77406ae41e0cd815a548e894278c0e5829147ce",
    "summary": "ed4746cb82a8a03b4becb31c5e7b44d1af416974b7e2462f48731679790e1800",
    "seal": "d5f4aab20460b4ff3a3044e484e3a83478ad7f2f541c9d08b48b5944529ffaa5",
    "manifest": "ffc31421654ddf685d7a31bbdb5a4d1d30d82da8e571c8f56fa6b75f79c1e218",
    "target": "a18ea8d2262dd058400202547c1af2c9c1867f1301858709ba60269be8b1b989",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def verify_blind(packet: Path, review_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    """This function must complete before any private target access."""
    paths = {
        "packet": packet,
        "row": review_dir / "row-judgments.jsonl",
        "group": review_dir / "group-judgments.jsonl",
        "summary": review_dir / "summary.json",
        "seal": review_dir / "seal.json",
    }
    actual = {name: sha256(path) for name, path in paths.items()}
    for name, value in actual.items():
        if value != EXPECTED[name]:
            raise ValueError(f"Frozen blind {name} SHA mismatch")
    seal = json.loads(paths["seal"].read_text(encoding="utf-8"))
    if (
        seal.get("packet_sha256") != EXPECTED["packet"]
        or seal.get("row_judgments_sha256") != EXPECTED["row"]
        or seal.get("group_judgments_sha256") != EXPECTED["group"]
        or seal.get("summary_sha256") != EXPECTED["summary"]
        or seal.get("disposition") != "BLOCK_FOR_INFERENCE"
        or Path(seal.get("packet_path", "")).resolve() != packet.resolve()
    ):
        raise ValueError("Blind seal content differs from frozen files")
    sealed_at = datetime.fromisoformat(seal["sealed_at_utc"])
    if sealed_at.tzinfo is None or sealed_at > datetime.now(timezone.utc):
        raise ValueError("Blind seal UTC chronology invalid")
    for name in ("row", "group", "summary"):
        modified_at = datetime.fromtimestamp(paths[name].stat().st_mtime, timezone.utc)
        if modified_at > sealed_at:
            raise ValueError(f"Blind {name} modified after seal")
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    if (
        summary.get("packet_sha256") != EXPECTED["packet"]
        or summary.get("key_access") != "none"
        or summary.get("disposition") != "BLOCK_FOR_INFERENCE"
        or summary.get("rows") != 72
        or summary.get("bases") != 18
        or summary.get("task_type_counts") != {"choice": 24, "noul": 24, "score": 24}
    ):
        raise ValueError("Blind summary disagrees with frozen protocol")
    packet_rows = rows(packet)
    row_reviews = rows(paths["row"])
    group_reviews = rows(paths["group"])
    packet_ids = {row["id"] for row in packet_rows}
    review_ids = {row["id"] for row in row_reviews}
    if (
        len(packet_rows) != 72
        or len(row_reviews) != 72
        or len(group_reviews) != 18
        or len(packet_ids) != 72
        or len(review_ids) != 72
        or packet_ids != review_ids
        or len({row["base_id"] for row in group_reviews}) != 18
    ):
        raise ValueError("Blind review identity or count mismatch")
    for packet_row, review_row in zip(packet_rows, row_reviews, strict=True):
        if any(
            packet_row[key] != review_row[key]
            for key in ("id", "base_id", "language", "task_type")
        ):
            raise ValueError("Blind row order/metadata mismatch")
    return (
        {
            "sealed_at_utc": seal["sealed_at_utc"],
            "blind_verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "blind_file_sha256": actual,
            "row_count": len(row_reviews),
            "group_count": len(group_reviews),
            "review_disposition": seal["disposition"],
            "key_access_before_verification": summary["key_access"],
        },
        row_reviews,
        group_reviews,
    )


def compare_postkey(
    panel: Path, row_reviews: list[dict], group_reviews: list[dict]
) -> dict:
    """Only call after verify_blind and its separate receipt have completed."""
    manifest_path = panel / "manifest.json"
    target_path = panel / "targets.private.jsonl"
    if (
        sha256(manifest_path) != EXPECTED["manifest"]
        or sha256(target_path) != EXPECTED["target"]
    ):
        raise ValueError("Private target/manifest SHA mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["private_targets_sha256"] != EXPECTED["target"]:
        raise ValueError("Manifest target commitment mismatch")
    targets = rows(target_path)
    by_id = {row["id"]: row for row in targets}
    by_group = {}
    for target in targets:
        group = target["base_id"]
        if (
            group in by_group
            and by_group[group]["semantic_gold"] != target["semantic_gold"]
        ):
            raise ValueError("Target variants disagree inside a base group")
        by_group[group] = target
    if len(targets) != 72 or len(by_id) != 72 or len(by_group) != 18:
        raise ValueError("Frozen target identity/count mismatch")
    row_judgments = Counter()
    row_types = Counter()
    row_languages = Counter()
    row_match = 0
    choice_key_match = 0
    choice_key_reviewed = 0
    for review in row_reviews:
        target = by_id[review["id"]]
        if any(
            target[key] != review[key] for key in ("base_id", "language", "task_type")
        ):
            raise ValueError("Blind review/target metadata mismatch")
        row_judgments[review["judgment"]] += 1
        row_types[target["task_type"]] += 1
        row_languages[target["language"]] += 1
        row_match += review.get("independent_answer") == target["semantic_gold"]
        if target["task_type"] == "choice" and review.get("option_key") is not None:
            choice_key_reviewed += 1
            choice_key_match += review["option_key"] == target["gold"]
    group_judgments = Counter()
    group_match = 0
    for review in group_reviews:
        target = by_group[review["base_id"]]
        if target["task_type"] != review["task_type"]:
            raise ValueError("Blind group/target type mismatch")
        group_judgments[review["judgment"]] += 1
        group_match += review.get("independent_answer") == target["semantic_gold"]
    choice_distribution = Counter(
        str(target["semantic_gold"])
        for target in by_group.values()
        if target["task_type"] == "choice"
    )
    return {
        "schema_version": "decision2-multilingual-hard-r6-postkey/1",
        "scope": "private aggregate; no model inference or release score",
        "key_opened_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": EXPECTED["manifest"],
        "targets_sha256": EXPECTED["target"],
        "review_rows": len(row_reviews),
        "review_groups": len(group_reviews),
        "blind_row_judgments": dict(row_judgments),
        "blind_group_judgments": dict(group_judgments),
        "independent_answer_matches_oracle_rows": row_match,
        "independent_answer_matches_oracle_groups": group_match,
        "choice_option_key_matches": choice_key_match,
        "choice_option_key_reviewed": choice_key_reviewed,
        "rows_by_type": dict(row_types),
        "rows_by_language": dict(row_languages),
        "choice_semantic_distribution_by_base": dict(choice_distribution),
        "release_status": "BLOCK_FOR_INFERENCE",
        "training_approved": False,
        "publication_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--blind-receipt", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    args = parser.parse_args()
    if args.blind_receipt.exists() or args.aggregate.exists():
        raise FileExistsError("Do not overwrite a blind/post-key receipt")
    verification, row_reviews, group_reviews = verify_blind(
        args.packet, args.review_dir
    )
    args.blind_receipt.write_text(
        json.dumps(verification, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    args.blind_receipt.chmod(0o600)
    result = compare_postkey(args.panel, row_reviews, group_reviews)
    result["blind_receipt_sha256"] = sha256(args.blind_receipt)
    args.aggregate.write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    args.aggregate.chmod(0o600)
    print(
        json.dumps(
            {
                "blind_receipt_sha256": result["blind_receipt_sha256"],
                "aggregate_sha256": sha256(args.aggregate),
                "review_rows": result["review_rows"],
                "review_groups": result["review_groups"],
                "independent_answer_matches_oracle_rows": result[
                    "independent_answer_matches_oracle_rows"
                ],
                "independent_answer_matches_oracle_groups": result[
                    "independent_answer_matches_oracle_groups"
                ],
                "release_status": result["release_status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
