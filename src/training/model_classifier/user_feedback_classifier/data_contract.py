"""Strict labels, split isolation and artifact names for feedback training."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

LABEL2ID = {"SAT": 0, "NEED_CLARIFICATION": 1, "WRONG_ANSWER": 2, "WANT_DIFFERENT": 3}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}


def fingerprint(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def parse_example(row: dict[str, Any]) -> dict[str, Any]:
    """Keep provenance and reject unknown labels instead of treating them as SAT."""
    text = row.get("text", row.get("followup"))
    label = row.get("label_name", row.get("label"))
    if isinstance(label, int) and not isinstance(label, bool):
        label = ID2LABEL.get(label)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Feedback examples require nonempty text")
    if label not in LABEL2ID:
        raise ValueError(f"Unknown feedback label: {label!r}")
    return {**row, "text": text, "label": LABEL2ID[label], "label_name": label}


def validate_splits(
    train: list[dict[str, Any]], validation: list[dict[str, Any]]
) -> dict[str, Any]:
    """Require all classes and isolation by text and available source groups.

    A group ID must identify the original conversation or template family, not
    an augmented row. Text checks alone cannot establish semantic disjointness;
    the returned manifest records whether group provenance was provided.
    """
    manifest: dict[str, Any] = {"contract_version": 2, "splits": {}}
    fingerprints: dict[str, set[str]] = {}
    groups: dict[str, set[str]] = {}
    for name, rows in (("train", train), ("validation", validation)):
        if not rows:
            raise ValueError(f"A separate nonempty {name} split is required")
        counts = Counter(row["label"] for row in rows)
        missing = set(ID2LABEL) - set(counts)
        if missing:
            raise ValueError(
                f"{name} is missing classes: {[ID2LABEL[i] for i in sorted(missing)]}"
            )
        labels_by_text: dict[str, int] = {}
        for row in rows:
            key = fingerprint(row["text"])
            previous = labels_by_text.setdefault(key, row["label"])
            if previous != row["label"]:
                raise ValueError(f"Conflicting feedback labels within {name}")
        fingerprints[name] = set(labels_by_text)
        groups[name] = {
            f"{row.get('source', '')}:{row['group_id']}"
            for row in rows
            if row.get("group_id") is not None
        }
        content = json.dumps(
            rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        manifest["splits"][name] = {
            "rows": len(rows),
            "normalized_unique": len(labels_by_text),
            "label_counts": {ID2LABEL[i]: counts[i] for i in ID2LABEL},
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "rows_with_group": sum(row.get("group_id") is not None for row in rows),
        }
    if fingerprints["train"] & fingerprints["validation"]:
        raise ValueError("Training and validation contain normalized duplicate text")
    if groups["train"] & groups["validation"]:
        raise ValueError("Training and validation share conversation/template groups")
    return manifest


def balanced_class_weights(labels: list[int]) -> list[float]:
    """Index weights by the actual output ID, including noncontiguous subsets."""
    if not labels or any(label not in ID2LABEL for label in labels):
        raise ValueError("Class weights require known feedback labels")
    counts = Counter(labels)
    return [
        len(labels) / (len(counts) * counts[i]) if counts[i] else 0.0 for i in ID2LABEL
    ]


def merged_output_directory(adapter_directory: str | Path) -> Path:
    path = Path(adapter_directory)
    for suffix in ("-lora", "_lora"):
        if path.name.endswith(suffix):
            return path.with_name(path.name[: -len(suffix)] + "-merged")
    return path.with_name(path.name + "-merged")


def write_label_mapping(directory: str | Path) -> None:
    payload = {"label2id": LABEL2ID, "id2label": ID2LABEL}
    (Path(directory) / "label_mapping.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
