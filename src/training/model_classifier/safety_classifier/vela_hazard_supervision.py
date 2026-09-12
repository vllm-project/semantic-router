"""Versioned training-only exclusion for inherited generated hazard annotations.

CultureGuard's generated jailbreak prompt can change which fine-grained risks
are observable. The binary safety annotation does not validate every inherited
category. Keep the original corpus and all development/test labels unchanged.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from ..sequence_repair.data import file_receipts
from .vela_data import LABELS
from .vela_hazard import read_rows

PROTOCOL = "hazard-train-supervision-v3"


def select_training_rows(rows):
    kept, excluded = [], []
    for row in rows:
        if row.get("source_split") != "train":
            raise ValueError("This protocol only accepts explicit training rows")
        if row.get("source") != "cultureguard":
            raise ValueError("Expected the pinned CultureGuard training projection")
        if row.get("source_tag") not in {"generic", "adapted", "jailbreaking"}:
            raise ValueError("Unknown CultureGuard generation provenance")
        if row.get("label") == "unsafe" and row["source_tag"] == "jailbreaking":
            excluded.append(
                {
                    "id": row["id"],
                    "group_id": row["group_id"],
                    "reason": "generated attack has no independent fine-grained prompt annotation",
                }
            )
        else:
            kept.append(row)
    return kept, excluded


def supervision_counts(rows):
    return {
        label: {
            "positive": sum(row["targets"][index] for row in rows),
            "negative": sum(
                row["label_mask"][index] and not row["targets"][index] for row in rows
            ),
            "unknown": sum(not row["label_mask"][index] for row in rows),
        }
        for index, label in enumerate(LABELS)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing overwrite of a versioned training corpus")
    rows = read_rows([args.input], len(LABELS))
    kept, excluded = select_training_rows(rows)
    args.output.mkdir(parents=True)
    path = args.output / "train.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in kept))
    manifest = {
        "protocol": PROTOCOL,
        "source": file_receipts([args.input]),
        "rows_before": len(rows),
        "rows_after": len(kept),
        "groups_after": len({row["group_id"] for row in kept}),
        "language_label_counts": dict(
            Counter(f"{row['language']}:{row['label']}" for row in kept)
        ),
        "source_tag_counts": dict(Counter(row["source_tag"] for row in kept)),
        "supervision": supervision_counts(kept),
        "excluded": excluded,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "development_or_test_modified": False,
        "limitations": "Generic and adapted labels remain synthetic source projections, not independent human annotations.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "excluded"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
