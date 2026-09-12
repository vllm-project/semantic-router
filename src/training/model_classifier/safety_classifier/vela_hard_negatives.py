"""Original content-safety hard negatives with fixed semantic-family splits."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from ..sequence_repair.data import assert_disjoint, normalized_text
from .vela_data import LABELS


def build(registry):
    result = {"train": [], "validation": []}
    families, texts = set(), set()
    for label, split, family, english, chinese in registry["families"]:
        if label not in LABELS or split not in result or family in families:
            raise ValueError("Every family needs one valid label and immutable split")
        families.add(family)
        for language, text in [("en", english), ("zh", chinese)]:
            normalized = normalized_text(text)
            if not normalized or normalized in texts:
                raise ValueError("Empty or duplicate authored text")
            texts.add(normalized)
            result[split].append(
                {
                    "id": f"vela-hazard-negative:{family}:{language}",
                    "group_id": f"vela-hazard-negative:{family}",
                    "text": text,
                    "source": "authored_hazard_hard_negatives_v1",
                    "source_split": split,
                    "language": language,
                    "length_bucket": "authored_short",
                    "label": "safe",
                    "targets": [0] * len(LABELS),
                    "label_mask": [1] * len(LABELS),
                    "contrast_category": label,
                    "supervision_protocol": registry["version"],
                    "annotation_provenance": "Original assistant-authored safe contrast; not independent natural-user evidence",
                }
            )
    assert_disjoint(result["train"], result["validation"])
    for split, rows in result.items():
        if {row["contrast_category"] for row in rows} != set(LABELS):
            raise ValueError(f"Every contrast category needs explicit {split} families")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing overwrite of a frozen corpus")
    registry_path = Path(__file__).parent / "configs/hazard-hard-negatives-v1.json"
    registry = json.loads(registry_path.read_text())
    result = build(registry)
    args.output.mkdir(parents=True)
    files = {}
    for split, rows in result.items():
        path = args.output / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows)
        )
        files[path.name] = {
            "rows": len(rows),
            "groups": len({row["group_id"] for row in rows}),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "category_rows": dict(Counter(row["contrast_category"] for row in rows)),
        }
    manifest = {
        "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        "rubric_sha256": hashlib.sha256(
            (registry_path.parent / "hazard-rubric-v1.json").read_bytes()
        ).hexdigest(),
        "files": files,
        "split_policy": registry["split_policy"],
        "authorship": registry["authorship"],
        "final_tests_read": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
