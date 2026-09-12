"""Training-only risk supervision repair; source labels and evaluation stay intact."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from ..sequence_repair.data import file_receipts
from .vela_data import LABELS
from .vela_hazard import read_rows
from .vela_hazard_supervision import supervision_counts

QUARANTINED_POSITIVES = {"specialized_advice", "misinformation"}
REVIEWED_SOURCE = "reviewed_safety_hazard_training_v2"


def reviewed_rows(raw, review):
    """Apply explicit text reviews only to exact source rows, never translations."""
    by_id = {row["id"]: row for row in raw}
    if len(by_id) != len(raw):
        raise ValueError("Raw source IDs must be unique")
    output, groups, seen = [], set(), set()
    for item in review["items"]:
        if item["id"] in seen or item["id"] not in by_id:
            raise ValueError("Duplicate or missing reviewed source row")
        seen.add(item["id"])
        row = by_id[item["id"]]
        if (
            row.get("source_split") != "train"
            or row["group_id"] != item["group_id"]
            or hashlib.sha256(row["text"].encode()).hexdigest() != item["text_sha256"]
        ):
            raise ValueError("Reviewed source text, group or split changed")
        groups.add(row["group_id"])
        binary = item["binary_label"]
        positive, unknown = set(item["positive_labels"]), set(item["unknown_labels"])
        if (
            positive & unknown
            or (positive | unknown) - set(LABELS)
            or binary not in {None, "safe", "unsafe"}
            or (binary == "safe" and (positive or unknown))
            or (binary == "unsafe" and not positive)
            or (binary is None and (positive or unknown))
        ):
            raise ValueError("Invalid reviewed risk contract")
        if binary is None:
            continue
        output.append(
            {
                **row,
                "source": REVIEWED_SOURCE,
                "raw_source": row["source"],
                "raw_binary_label": row["label"],
                "raw_targets": list(row["targets"]),
                "raw_label_mask": list(row["label_mask"]),
                "label": binary,
                "targets": [int(label in positive) for label in LABELS],
                "label_mask": [int(label not in unknown) for label in LABELS],
                "review_reason": item["reason"],
                "review_protocol": review["protocol"],
                "reviewer": review["reviewer"],
            }
        )
    return output, groups


def repair_weak_rows(rows, reviewed_groups, *, task):
    """Quarantine incompatible weak positives; never turn an unsafe label safe."""
    if task not in {"safety", "hazard"}:
        raise ValueError("Unknown risk task")
    output, excluded = [], []
    for row in rows:
        if row.get("source_split") != "train":
            raise ValueError("Only training rows may be repaired")
        reason = None
        if row["group_id"] in reviewed_groups:
            reason = "reviewed source group supersedes all unreviewed variants"
        elif row.get("source_tag") == "jailbreaking":
            reason = "generated jailbreak source labels are quarantined"
        positive = {
            label for label, value in zip(LABELS, row["targets"], strict=True) if value
        }
        if (
            reason is None
            and task == "safety"
            and row["label"] == "unsafe"
            and positive
            and positive <= QUARANTINED_POSITIVES
        ):
            reason = "unsafe weak label depends only on incompatible risk categories"
        if reason:
            excluded.append(
                {"id": row["id"], "group_id": row["group_id"], "reason": reason}
            )
            continue
        targets, mask = list(row["targets"]), list(row["label_mask"])
        if task == "hazard":
            for index, label in enumerate(LABELS):
                if label in QUARANTINED_POSITIVES and targets[index]:
                    targets[index], mask[index] = 0, 0
            if not any(mask):
                excluded.append(
                    {
                        "id": row["id"],
                        "group_id": row["group_id"],
                        "reason": "no observed labels after weak-positive quarantine",
                    }
                )
                continue
        output.append(
            {
                **row,
                "raw_targets_before_review_repair": list(row["targets"]),
                "raw_mask_before_review_repair": list(row["label_mask"]),
                "targets": targets,
                "label_mask": mask,
                "supervision_protocol": "vela-reviewed-risk-supervision-v5",
                "annotation_provenance": "Remaining source labels are weak, not individually reviewed; quarantined positives are unknown, not negative",
            }
        )
    return output, excluded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aegis", type=Path, required=True)
    parser.add_argument("--culture", type=Path, required=True)
    parser.add_argument("--culture-hazard", nargs="+", required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing overwrite of versioned supervision")
    review = json.loads(args.review.read_text())
    for name, path in [("aegis", args.aegis), ("culture", args.culture)]:
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != review["source_sha256"][name]
        ):
            raise ValueError("Raw training source differs from reviewed revision")
    aegis = read_rows([args.aegis], len(LABELS))
    culture = read_rows([args.culture], len(LABELS))
    projected_culture = read_rows(args.culture_hazard, len(LABELS))
    raw_culture = {row["id"]: row for row in culture}
    for row in projected_culture:
        raw = raw_culture.get(row["id"])
        if raw is None or any(
            row[key] != raw[key] for key in ["text", "group_id", "label"]
        ):
            raise ValueError("Projected CultureGuard does not match raw training")
    strong, groups = reviewed_rows(aegis + culture, review)
    files = {"reviewed-train": strong}
    exclusions = {}
    for name, rows, task in [
        ("safety-aegis", aegis, "safety"),
        ("safety-culture", culture, "safety"),
        ("hazard-aegis", aegis, "hazard"),
        ("hazard-culture", projected_culture, "hazard"),
    ]:
        files[name], exclusions[name] = repair_weak_rows(rows, groups, task=task)
    args.output.mkdir(parents=True)
    manifest = {
        "protocol": "vela-reviewed-risk-supervision-v5",
        "raw_source_and_evaluation_unchanged": True,
        "reviewed_groups_removed_from_all_weak_sources": len(groups),
        "reviewed_rows": len(strong),
        "reviewed_rows_are_not_propagated_to_unreviewed_translations": True,
        "quarantined_weak_positive_labels": sorted(QUARANTINED_POSITIVES),
        "review_sha256": hashlib.sha256(args.review.read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "source_files": file_receipts(
            [str(args.aegis), str(args.culture)] + args.culture_hazard
        ),
        "test_used": False,
        "files": {},
        "exclusions": exclusions,
    }
    for name, rows in files.items():
        path = args.output / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows)
        )
        manifest["files"][name] = {
            "rows": len(rows),
            "groups": len({row["group_id"] for row in rows}),
            "sources": dict(Counter(row["source"] for row in rows)),
            "labels": dict(Counter(row["label"] for row in rows)),
            "supervision": supervision_counts(rows),
            "observed_count_distribution": dict(
                Counter(sum(row["label_mask"]) for row in rows)
            ),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "exclusions"}
        )
    )


if __name__ == "__main__":
    main()
