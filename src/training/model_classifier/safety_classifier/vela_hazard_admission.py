"""Admit explicit text reviews, independently of archived source crosswalks.

The input JSONL remains immutable and its annotations are preserved as provenance.
Each review binds the entire input file, rubric, visible text and source identity.
Missing annotations never supply negative labels. Legacy reviewed corpora can use
``preserve_existing`` to reject any unplanned target or mask change; a source name
alone never qualifies a row as reviewed.
"""

import argparse
import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

from .vela_data import LABELS

PROTOCOL = "vela-hazard-semantic-admission-v1"
IDENTITY_FIELDS = ("id", "group_id", "source", "source_revision", "source_split")
PARTITIONS = ("positive_labels", "negative_labels", "unknown_labels")


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def _required_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid {name}")
    return value


def _partition(item):
    assigned = set()
    partitions = {}
    for name in PARTITIONS:
        values = item.get(name)
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise ValueError(f"Explicit list required for {name}")
        labels = set(values)
        if len(labels) != len(values) or assigned & labels:
            raise ValueError("Duplicate or overlapping reviewed labels")
        partitions[name] = labels
        assigned.update(labels)
    if assigned != set(LABELS):
        raise ValueError("Review must partition all twelve labels exactly")
    unknown = partitions["unknown_labels"]
    reasons = item.get("unknown_reasons")
    if not isinstance(reasons, dict) or set(reasons) != unknown:
        raise ValueError("Every unknown label needs its own reason")
    for reason in reasons.values():
        _required_text(reason, "unknown reason")
    positive = partitions["positive_labels"]
    label = "unsafe" if positive else "unknown" if unknown else "safe"
    if item.get("binary_label") != label:
        raise ValueError("Binary label disagrees with reviewed observations")
    targets = [int(name in positive) for name in LABELS]
    mask = [int(name not in unknown) for name in LABELS]
    return targets, mask, label


def _validate_review(source_bytes, review, rubric_bytes):
    rubric = json.loads(rubric_bytes)
    if tuple(rubric.get("labels", ())) != LABELS:
        raise ValueError("Rubric label order differs from the Hazard contract")
    if review.get("labels") != list(LABELS):
        raise ValueError("Review label order differs from the Hazard contract")
    if review.get("input_sha256") != sha256(source_bytes):
        raise ValueError("Reviewed input file changed")
    if review.get("rubric_sha256") != sha256(rubric_bytes):
        raise ValueError("Reviewed rubric changed")
    for name in ("protocol", "reviewer", "reviewer_kind"):
        _required_text(review.get(name), name)
    if not isinstance(review.get("items"), list):
        raise ValueError("Review items must be an explicit list")


def _validate_item(row, item):
    for key in IDENTITY_FIELDS:
        _required_text(row.get(key), key)
        if item.get(key) != row[key]:
            raise ValueError(f"Reviewed source identity changed: {key}")
    text = _required_text(row.get("text"), "visible text")
    if item.get("text_sha256") != sha256(text.encode("utf-8")):
        raise ValueError("Reviewed visible text changed")
    _required_text(item.get("reason"), "review reason")
    return _partition(item)


def _counts(rows):
    return {
        label: {
            state: {
                "rows": sum(predicate(row, index) for row in rows),
                "groups": len(
                    {row["group_id"] for row in rows if predicate(row, index)}
                ),
            }
            for state, predicate in (
                ("positive", lambda row, i: bool(row["targets"][i])),
                (
                    "negative",
                    lambda row, i: bool(row["label_mask"][i]) and not row["targets"][i],
                ),
                ("unknown", lambda row, i: not row["label_mask"][i]),
            )
        }
        for index, label in enumerate(LABELS)
    }


def admit_corpus(source_bytes, review, *, rubric_bytes, preserve_existing=False):
    """Return admitted rows, exclusions and receipts without altering any input.

    The review must cover every input ID, including fully unknown exclusions.
    Partial negative supervision is retained with ``label=unknown`` and cannot
    enter a binary-safe false-positive denominator. ``preserve_existing`` checks
    existing targets/masks against this explicit review, not source categories.
    It does not infer the authenticity or semantic correctness of a review.
    """
    _validate_review(source_bytes, review, rubric_bytes)
    rows = [
        json.loads(line)
        for line in source_bytes.decode("utf-8").split("\n")
        if line.strip()
    ]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Input must contain nonempty JSONL records")
    by_id = {}
    for row in rows:
        identifier = _required_text(row.get("id"), "input ID")
        if identifier in by_id:
            raise ValueError("Duplicate input ID")
        by_id[identifier] = row
    items = {}
    for item in review["items"]:
        if not isinstance(item, dict):
            raise ValueError("Review items must be objects")
        identifier = _required_text(item.get("id"), "review ID")
        if identifier in items:
            raise ValueError("Duplicate review ID")
        items[identifier] = item
    if set(items) != set(by_id):
        raise ValueError("Review must cover exactly the input IDs")

    review_sha = sha256(
        json.dumps(review, sort_keys=True, ensure_ascii=True).encode("utf-8")
    )
    output, excluded, reviewed_rows, seen_text = [], [], [], {}
    for row in rows:
        item = items[row["id"]]
        targets, mask, label = _validate_item(row, item)
        if preserve_existing and (
            row.get("targets") != targets or row.get("label_mask") != mask
        ):
            raise ValueError("Existing reviewed supervision would change")
        signature = (tuple(targets), tuple(mask))
        previous = seen_text.setdefault(row["text"], signature)
        if previous != signature:
            raise ValueError("Conflicting reviews for identical visible text")
        reviewed_rows.append(
            {"group_id": row["group_id"], "targets": targets, "label_mask": mask}
        )
        provenance = {
            "protocol": PROTOCOL,
            "input_sha256": review["input_sha256"],
            "rubric_sha256": review["rubric_sha256"],
            "review_canonical_sha256": review_sha,
            "review_protocol": review["protocol"],
            "reviewer": review["reviewer"],
            "reviewer_kind": review["reviewer_kind"],
            "item": deepcopy(item),
        }
        if not any(mask):
            excluded.append(
                {
                    **{key: row[key] for key in IDENTITY_FIELDS},
                    "text_sha256": item["text_sha256"],
                    "reason": "No observed labels; never convert unknown to safe",
                    "current_supervision": provenance,
                }
            )
            continue
        output.append(
            {
                **deepcopy(row),
                "label": label,
                "binary_observed": label != "unknown",
                "targets": targets,
                "label_mask": mask,
                "source_annotations": {
                    "admission_basis": "Input annotations are provenance only; explicit text review supplies supervision",
                    "input_metadata": deepcopy(
                        {key: value for key, value in row.items() if key != "text"}
                    ),
                    "text_sha256": item["text_sha256"],
                },
                "annotation_provenance": PROTOCOL,
                "current_supervision": provenance,
            }
        )
    sources = sorted({row["source"] for row in rows})
    manifest = {
        "protocol": PROTOCOL,
        "input_sha256": sha256(source_bytes),
        "rubric_sha256": sha256(rubric_bytes),
        "review_canonical_sha256": review_sha,
        "labels": list(LABELS),
        "preserve_existing": preserve_existing,
        "input_rows": len(rows),
        "admitted_rows": len(output),
        "excluded_rows": len(excluded),
        "reserved_groups": sorted({row["group_id"] for row in rows}),
        "binary_labels": dict(Counter(row["label"] for row in output)),
        "observed_count_distribution": dict(
            Counter(sum(row["label_mask"]) for row in output)
        ),
        "supervision": _counts(output),
        "reviewed_supervision_including_exclusions": _counts(reviewed_rows),
        "sources": {
            source: _counts([row for row in output if row["source"] == source])
            for source in sources
        },
        "raw_annotations_supply_gold": False,
        "limitation": "Hashes bind review identity, not reviewer accuracy; no label is inferred from source metadata",
    }
    return {"rows": output, "excluded": excluded, "manifest": manifest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preserve-existing", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite a versioned admission result")
    review_bytes = args.review.read_bytes()
    result = admit_corpus(
        args.input.read_bytes(),
        json.loads(review_bytes),
        rubric_bytes=args.rubric.read_bytes(),
        preserve_existing=args.preserve_existing,
    )
    args.output.mkdir(parents=True)
    for name in ("rows", "excluded"):
        path = args.output / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in result[name])
        )
        result["manifest"][f"{name}_sha256"] = sha256(path.read_bytes())
    result["manifest"]["review_file_sha256"] = sha256(review_bytes)
    result["manifest"]["implementation_sha256"] = sha256(Path(__file__).read_bytes())
    (args.output / "manifest.json").write_text(
        json.dumps(result["manifest"], indent=2) + "\n"
    )
    print(json.dumps({name: len(result[name]) for name in ("rows", "excluded")}))


if __name__ == "__main__":
    main()
