"""Training-only partial observation for inherited CultureGuard risk categories.

This v4 projection preserves v3 and the raw development/test data. Binary safety
consistency does not establish exhaustive fine-grained annotation after prompt
generation. Generic unsafe rows retain only positive categories independently
traceable to the original training prompt; missing categories are unknown.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from ..sequence_repair.data import file_receipts, normalized_text
from .vela_data import LABELS, SOURCE_SHA256, hazard_target
from .vela_hazard import read_rows
from .vela_hazard_supervision import supervision_counts

PROTOCOL = "hazard-train-partial-observation-v4"
SAFE_SOURCE = "cultureguard_safe"
POSITIVE_SOURCE = "cultureguard_generic_partial_positive"


def original_training_annotations(rows):
    """Intersect attributable response variants; never infer a missing category."""
    grouped = defaultdict(list)
    for row in rows:
        fingerprint = hashlib.sha256(
            normalized_text(row["prompt"].strip()).encode()
        ).hexdigest()
        targets, mask, reason = hazard_target(row)
        if not reason and row["prompt_label"] == "unsafe":
            grouped[f"aegis:{fingerprint}"].append(
                {
                    label
                    for label, value, observed in zip(
                        LABELS, targets, mask, strict=True
                    )
                    if value and observed
                }
            )
    return {
        group: set.intersection(*annotations) for group, annotations in grouped.items()
    }


def project_training_rows(rows, original):
    result, excluded = [], []
    for row in rows:
        if row.get("source_split") != "train" or row.get("source") != "cultureguard":
            raise ValueError(
                "Only the explicit CultureGuard training projection is allowed"
            )
        tag = row.get("source_tag")
        if tag not in {"generic", "adapted", "jailbreaking"}:
            raise ValueError("Unreviewed generation provenance")
        if row.get("label") not in {"safe", "unsafe"}:
            raise ValueError("Unknown binary safety label")
        reason = None
        if tag == "jailbreaking":
            reason = (
                "generated jailbreak binary and category annotations are unverified"
            )
        elif row["label"] == "safe":
            if any(row["targets"]) or not all(row["label_mask"]):
                raise ValueError(
                    "Safe examples must have explicit all-category negatives"
                )
            targets, mask, source = [0] * len(LABELS), [1] * len(LABELS), SAFE_SOURCE
        elif tag != "generic":
            reason = "adaptation preserves binary label but categories are not independently verified"
        elif row["group_id"] not in original:
            reason = "no attributable original AEGIS training prompt"
        else:
            raw_positive = {
                label
                for label, value, observed in zip(
                    LABELS, row["targets"], row["label_mask"], strict=True
                )
                if value and observed
            }
            positive = raw_positive & original[row["group_id"]]
            if not positive:
                reason = "no positive category agreed by generic and original prompt annotations"
            else:
                targets = [int(label in positive) for label in LABELS]
                mask, source = list(targets), POSITIVE_SOURCE
        if reason:
            excluded.append(
                {"id": row["id"], "group_id": row["group_id"], "reason": reason}
            )
            continue
        result.append(
            {
                **row,
                "source": source,
                "raw_source": row["source"],
                "raw_targets": list(row["targets"]),
                "raw_label_mask": list(row["label_mask"]),
                "targets": targets,
                "label_mask": mask,
                "supervision_protocol": PROTOCOL,
                "annotation_provenance": (
                    "weak source binary safe label; projected all-category negatives, not individually verified"
                    if source == SAFE_SOURCE
                    else "generic mapped positives intersect original training prompt; all absent labels unknown"
                ),
            }
        )
    return result, excluded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--aegis-train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing overwrite of versioned training data")
    payload = args.aegis_train.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256["train"]:
        raise ValueError("Original training data differs from pinned AEGIS revision")
    rows = read_rows([args.input], len(LABELS))
    original = original_training_annotations(json.loads(payload))
    projected, excluded = project_training_rows(rows, original)
    args.output.mkdir(parents=True)
    files = {}
    for source in [SAFE_SOURCE, POSITIVE_SOURCE]:
        selected = [row for row in projected if row["source"] == source]
        if not selected:
            raise ValueError(f"Empty supervision source: {source}")
        path = args.output / f"{source}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in selected)
        )
        files[path.name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(selected),
            "groups": len({row["group_id"] for row in selected}),
            "supervision": supervision_counts(selected),
            "observed_count_distribution": dict(
                Counter(sum(row["label_mask"]) for row in selected)
            ),
        }
    manifest = {
        "protocol": PROTOCOL,
        "source_files": file_receipts([args.input, str(args.aegis_train)]),
        "files": files,
        "excluded": excluded,
        "exclusion_counts": dict(Counter(row["reason"] for row in excluded)),
        "raw_targets_preserved": True,
        "development_or_test_modified": False,
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "limitation": "Generic translations retain inherited positive labels, not independent per-language adjudication. Generic/adapted safe and unchanged AEGIS supervision remain source labels, not individually verified clean negatives. Source sampling weights do not equal effective per-label gradient weights.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "excluded"}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
