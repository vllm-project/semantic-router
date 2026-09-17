"""Frozen task corpus receipts and source-group partitioning."""

import hashlib
import json
from pathlib import Path

from .data import assert_disjoint, normalized_text


def group_split(group, seed=20260913):
    bucket = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "dev" if bucket < 85 else "test"  # noqa: PLR2004


def write_corpus(output, splits, contract, provenance):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite a frozen corpus")
    unique = {}
    for split, rows in splits.items():
        seen, retained, retained_keys = {}, [], set()
        for row in rows:
            key = normalized_text(row["text"])
            previous = seen.setdefault(key, row["label"])
            if previous != row["label"]:
                raise ValueError("Contradictory labels for identical normalized text")
            if key in unique and unique[key] != split:
                raise ValueError("Duplicate text crosses corpus splits")
            unique[key] = split
            if key not in retained_keys:
                retained.append(row)
                retained_keys.add(key)
        splits[split] = retained
    for first, second in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        assert_disjoint(splits[first], splits[second])
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, rows in splits.items():
        content = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        )
        (output / f"{split}.jsonl").write_text(content)
        files[split] = {
            "rows": len(rows),
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
        }
    (output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    manifest = {
        "files": files,
        "provenance": provenance,
        "test_used_for_selection": False,
        "split_policy": "Group-disjoint before sampling; exact NFKC/case/whitespace overlap rejected",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
