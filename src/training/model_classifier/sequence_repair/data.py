"""Dependency-light sequence data contracts and classification metrics."""

import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path


def normalized_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def load_contract(path):
    config = json.loads(Path(path).read_text())
    id_to_label = {int(index): label for index, label in config["id2label"].items()}
    if set(id_to_label) != set(range(len(id_to_label))):
        raise ValueError("Label IDs must be contiguous from zero")
    if config["label2id"] != {label: index for index, label in id_to_label.items()}:
        raise ValueError("id2label and label2id must be a bijection")
    return config["label2id"], id_to_label


def read_records(paths, label_to_id):
    records, ids, texts = [], set(), {}
    for path in paths:
        for line in Path(path).read_text().split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError("A classification row needs nonempty text")
            if not row.get("id") or row["id"] in ids:
                raise ValueError("Missing or duplicate row ID")
            if not row.get("group_id"):
                raise ValueError("Each row needs a source/translation/template group")
            if row.get("label") not in label_to_id:
                raise ValueError("Unknown classification label")
            normalized = normalized_text(row["text"])
            previous = texts.setdefault(normalized, row["label"])
            if previous != row["label"]:
                raise ValueError("Identical text has contradictory labels")
            ids.add(row["id"])
            records.append(row)
    if not records:
        raise ValueError(
            "Empty datasets cannot produce training or evaluation evidence"
        )
    return records


def assert_disjoint(train, dev):
    for key, transform in [("group_id", str), ("text", normalized_text), ("id", str)]:
        overlap = {transform(row[key]) for row in train} & {
            transform(row[key]) for row in dev
        }
        if overlap:
            raise ValueError(f"Training/development overlap in {key}: {len(overlap)}")


def file_receipts(paths):
    return [
        {
            "file": Path(path).name,
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        }
        for path in paths
    ]


def classification_metrics(gold, predicted, id_to_label):
    if not gold or len(gold) != len(predicted):
        raise ValueError("Empty or mismatched prediction evidence")
    count = len(id_to_label)
    confusion = [[0] * count for _ in range(count)]
    for index, expected in enumerate(gold):
        actual = predicted[index]
        if expected not in id_to_label or actual not in id_to_label:
            raise ValueError("Prediction outside the frozen label contract")
        confusion[expected][actual] += 1
    per_label = {}
    for index, label in id_to_label.items():
        tp = confusion[index][index]
        support = sum(confusion[index])
        selected = sum(row[index] for row in confusion)
        per_label[label] = {
            "support": support,
            "tp": tp,
            "fp": selected - tp,
            "fn": support - tp,
            "precision": tp / selected if selected else 0.0,
            "recall": tp / support if support else 0.0,
            "f1": 2 * tp / (selected + support) if selected + support else 0.0,
        }
    present = [row for row in per_label.values() if row["support"]]
    return {
        "rows": len(gold),
        "accuracy": sum(confusion[index][index] for index in range(count)) / len(gold),
        "macro_f1": sum(row["f1"] for row in per_label.values()) / count,
        "macro_f1_present_labels": sum(row["f1"] for row in present) / len(present),
        "per_label": per_label,
        "confusion": confusion,
    }


def label_counts(records):
    return dict(Counter(row["label"] for row in records))
