"""Append reviewed training-only non-feedback requests to a frozen corpus."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from ..sequence_repair.data import assert_disjoint, normalized_text
from .vela_no_feedback import project_review


def append_reviewed_negatives(previous, development, projected):
    if projected.get("validation"):
        raise ValueError("Natural-negative expansion must be training only")
    seen = {normalized_text(row["text"]): row for row in previous}
    accepted, duplicates = [], []
    for source_row in projected["train"]:
        if (
            source_row["label"] != "NO_FEEDBACK"
            or source_row.get("source_split") != "train"
        ):
            raise ValueError("Expected reviewed NO_FEEDBACK training rows")
        row = {**source_row, "source": "wildfeedback_reviewed_no_feedback_v3"}
        key = normalized_text(row["text"])
        if key in seen:
            prior = seen[key]
            if prior["label"] != row["label"]:
                raise ValueError("Reviewed text conflicts with an existing label")
            duplicates.append(
                {
                    "id": row["id"],
                    "reason": "same-label normalized text duplicate",
                    "existing_id": prior["id"],
                }
            )
            continue
        seen[key] = row
        accepted.append(row)
    combined = previous + accepted
    assert_disjoint(combined, development)
    return combined, accepted, duplicates


def read_rows(path):
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--wildfeedback", type=Path, required=True)
    parser.add_argument(
        "--review",
        type=Path,
        default=Path(__file__).parent / "configs/vela-no-feedback-review-v3.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite a versioned corpus")
    review = json.loads(args.review.read_text())
    source = args.wildfeedback.read_bytes()
    if hashlib.sha256(source).hexdigest() != review["source_sha256"]:
        raise ValueError("WildFeedback bytes differ from the reviewed revision")
    projected = project_review(review, json.loads(source))
    previous = read_rows(args.previous / "train.jsonl")
    development = read_rows(args.previous / "validation.jsonl")
    combined, accepted, duplicates = append_reviewed_negatives(
        previous, development, projected
    )
    args.output.mkdir(parents=True)
    for name, rows in [("train", combined), ("new-negative-training", accepted)]:
        (args.output / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows)
        )
    for name in ["validation.jsonl", "contract.json"]:
        (args.output / name).write_bytes((args.previous / name).read_bytes())
    manifest = {
        "purpose": "Reviewed natural NO_FEEDBACK expansion; unchanged development",
        "source_revision": review["revision"],
        "source_sha256": review["source_sha256"],
        "review_sha256": hashlib.sha256(args.review.read_bytes()).hexdigest(),
        "reviewed": len(review["items"]),
        "semantic_excluded": sum(row["label"] is None for row in review["items"]),
        "normalized_duplicates": duplicates,
        "new_rows": len(accepted),
        "new_languages": dict(Counter(row["language"] for row in accepted)),
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "final_accessed": False,
        "files": {},
    }
    for name in ["train", "validation", "new-negative-training"]:
        path = args.output / f"{name}.jsonl"
        rows = read_rows(path)
        manifest["files"][name] = {
            "rows": len(rows),
            "groups": len({row["group_id"] for row in rows}),
            "labels": dict(Counter(row["label"] for row in rows)),
            "sources": dict(Counter(row["source"] for row in rows)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
