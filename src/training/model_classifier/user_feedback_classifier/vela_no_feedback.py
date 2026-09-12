"""Materialize reviewed non-feedback user turns while retaining source partitions."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from ..sequence_repair.data import assert_disjoint
from .vela_contract import VELA_ID2LABEL, VELA_LABEL2ID
from .vela_data import TRAIN_AND_VALIDATION_PERCENT, TRAIN_PERCENT


def project_review(review, annotations):
    wanted = {row["source_index"]: row for row in review["items"]}
    if len(wanted) != len(review["items"]):
        raise ValueError("Review indices must be unique")
    splits = {"train": [], "validation": []}
    conversation = -1
    for index, source in enumerate(annotations):
        if source["UtterranceId"] == 0:
            conversation += 1
        if index not in wanted:
            continue
        item = wanted.pop(index)
        text = source["Content"].strip()
        group = f"wildfeedback:conversation:{conversation}"
        partition = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
        split = (
            "train"
            if partition < TRAIN_PERCENT
            else "validation" if partition < TRAIN_AND_VALIDATION_PERCENT else "test"
        )
        if (
            source["Role"] != "User"
            or source["State"] != "NEWTOPIC"
            or group != item["group_id"]
            or split != item["split"]
            or split == "test"
            or hashlib.sha256(text.encode()).hexdigest() != item["text_sha256"]
        ):
            raise ValueError("Reviewed source text, state or partition changed")
        if item["label"] is None:
            continue
        if item["label"] != "NO_FEEDBACK":
            raise ValueError("This review only supplies non-feedback labels")
        splits[split].append(
            {
                "id": f"wildfeedback-no-feedback:{index}",
                "group_id": group,
                "text": text,
                "label": "NO_FEEDBACK",
                "source": "wildfeedback_reviewed_no_feedback_v1",
                "source_split": split,
                "source_state": source["State"],
                "language": item["language"],
                "length_bucket": "natural_short",
                "label_provenance": review["review"],
            }
        )
    if wanted:
        raise ValueError("Reviewed source indices are absent")
    assert_disjoint(splits["train"], splits["validation"])
    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--review",
        type=Path,
        default=Path(__file__).parent / "configs/vela-no-feedback-review-v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing overwrite of frozen review material")
    review = json.loads(args.review.read_text())
    source = args.source.read_bytes()
    if hashlib.sha256(source).hexdigest() != review["source_sha256"]:
        raise ValueError("Source differs from reviewed revision")
    splits = project_review(review, json.loads(source))
    args.output.mkdir(parents=True)
    manifest = {
        "source": review["source"],
        "revision": review["revision"],
        "review_sha256": hashlib.sha256(args.review.read_bytes()).hexdigest(),
        "source_sha256": review["source_sha256"],
        "reviewed_rows": len(review["items"]),
        "excluded_rows": sum(row["label"] is None for row in review["items"]),
        "review": review["review"],
        "policy": review["policy"],
        "files": {},
        "test_used": False,
    }
    for split, rows in splits.items():
        path = args.output / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows)
        )
        manifest["files"][split] = {
            "rows": len(rows),
            "languages": dict(Counter(row["language"] for row in rows)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (args.output / "contract.json").write_text(
        json.dumps(
            {
                "id2label": VELA_ID2LABEL,
                "label2id": VELA_LABEL2ID,
                "problem_type": "single_label_classification",
            },
            indent=2,
        )
        + "\n"
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
