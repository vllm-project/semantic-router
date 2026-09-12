"""Validated character spans and strict entity metrics for PII training.

Span offsets are Unicode codepoints. Tokenizers may include whitespace in a
token span; decoded entities trim that whitespace and adjust both offsets.
"""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_label_contract(path):
    config = json.loads(Path(path).read_text())
    id_to_label = {int(index): label for index, label in config["id2label"].items()}
    label_to_id = config["label2id"]
    if set(id_to_label) != set(range(len(id_to_label))):
        raise ValueError("Label IDs must be contiguous from zero")
    if {label: index for index, label in id_to_label.items()} != label_to_id:
        raise ValueError("id2label and label2id disagree")
    if "O" not in label_to_id:
        raise ValueError("The label contract must include O")
    for label in label_to_id:
        if label == "O":
            continue
        if not label.startswith(("B-", "I-")):
            raise ValueError(f"Expected a BIO label, got {label}")
        if "B-" + label[2:] not in label_to_id or "I-" + label[2:] not in label_to_id:
            raise ValueError(f"Missing BIO pair for {label}")
    return label_to_id, id_to_label


def validated_spans(record, label_to_id):
    text = record["full_text"]
    spans = sorted(
        record["spans"], key=lambda span: (span["start_position"], span["end_position"])
    )
    previous_end = 0
    for span in spans:
        start, end = span["start_position"], span["end_position"]
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(text)
        ):
            raise ValueError("Invalid character span")
        if start < previous_end:
            raise ValueError("Overlapping entities need an explicit annotation policy")
        if text[start:end] != span["entity_value"]:
            raise ValueError("Annotation value does not match its character span")
        if "B-" + span["entity_type"] not in label_to_id:
            raise ValueError(f"Unknown entity type: {span['entity_type']}")
        if not text[start:end].strip():
            raise ValueError("Whitespace cannot be an entity")
        previous_end = end
    return spans


def align_record(record, tokenizer, label_to_id, max_length):
    """Label every subword without silently truncating or overwriting spans."""
    spans = validated_spans(record, label_to_id)
    encoding = tokenizer(
        record["full_text"],
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_offsets_mapping=True,
    )
    if len(encoding["input_ids"]) > max_length:
        raise ValueError("Input exceeds the explicitly configured training budget")
    offsets = encoding["offset_mapping"]
    labels = [-100 if start == end else label_to_id["O"] for start, end in offsets]
    for span in spans:
        indices = [
            index
            for index, (start, end) in enumerate(offsets)
            if start != end
            and start < span["end_position"]
            and end > span["start_position"]
        ]
        if not indices:
            raise ValueError("An annotated entity has no token coverage")
        first, last = offsets[indices[0]][0], offsets[indices[-1]][1]
        if record["full_text"][first:last].strip() != span["entity_value"].strip():
            raise ValueError(
                "An entity boundary crosses a token containing non-whitespace context"
            )
        for position, index in enumerate(indices):
            if labels[index] != label_to_id["O"]:
                raise ValueError("A token overlaps multiple annotated entities")
            prefix = "B-" if position == 0 else "I-"
            labels[index] = label_to_id[prefix + span["entity_type"]]
    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": labels,
        "offset_mapping": offsets,
    }


def decode_entities(text, offsets, labels):
    """Mirror permissive BIO grouping; never join different entity types."""
    if len(offsets) != len(labels):
        raise ValueError("Token offsets and labels differ in length")
    entities = []
    current = None

    def finish():
        if current is None:
            return
        entity_type, start, end = current
        value = text[start:end]
        trimmed = value.strip()
        if trimmed:
            start += len(value) - len(value.lstrip())
            entities.append((entity_type, start, start + len(trimmed)))

    for (start, end), label in zip(offsets, labels):  # noqa: B905
        if start == end:
            continue
        entity_type = label[2:] if label.startswith(("B-", "I-")) else None
        if label.startswith("I-") and current is not None and current[0] == entity_type:
            current = (entity_type, current[1], end)
            continue
        finish()
        current = (entity_type, start, end) if entity_type else None
    finish()
    return entities


def gold_entities(record):
    result = []
    for span in record["spans"]:
        start, end = span["start_position"], span["end_position"]
        value = record["full_text"][start:end]
        start += len(value) - len(value.lstrip())
        result.append((span["entity_type"], start, start + len(value.strip())))
    return result


def entity_metrics(records, predictions, entity_types):
    if len(records) != len(predictions):
        raise ValueError("Prediction count differs from evaluation row count")
    counts = {entity_type: Counter(tp=0, fp=0, fn=0) for entity_type in entity_types}
    negative_documents = false_positive_documents = complete_email = total_email = 0
    for record, predicted in zip(records, predictions):  # noqa: B905
        expected = set(gold_entities(record))
        actual = {tuple(entity) for entity in predicted}
        for kind, entities in [
            ("tp", actual & expected),
            ("fp", actual - expected),
            ("fn", expected - actual),
        ]:
            for entity_type, _, _ in entities:
                if entity_type not in counts:
                    raise ValueError(
                        f"Prediction contains unknown entity type: {entity_type}"
                    )
                counts[entity_type][kind] += 1
        emails = {entity for entity in expected if entity[0] == "EMAIL_ADDRESS"}
        total_email += len(emails)
        complete_email += len(emails & actual)
        if not expected:
            negative_documents += 1
            false_positive_documents += bool(actual)

    def scores(count):
        tp, fp, fn = count["tp"], count["fp"], count["fn"]
        return {
            **dict(count),
            "support": tp + fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        }

    total = Counter(tp=0, fp=0, fn=0)
    for count in counts.values():
        total.update(count)
    return {
        "rows": len(records),
        "micro": scores(total),
        "per_type": {
            entity_type: scores(count) for entity_type, count in counts.items()
        },
        "complete_email": {
            "correct": complete_email,
            "total": total_email,
            "recall": complete_email / total_email if total_email else None,
        },
        "negative_documents": {
            "total": negative_documents,
            "with_false_positive": false_positive_documents,
        },
    }


def assert_split_isolation(splits):
    """Reject text, family, source-document, or entity-value reuse across splits."""
    owners = defaultdict(dict)
    for split, records in splits.items():
        for record in records:
            identities = [
                ("text", hashlib.sha256(record["full_text"].encode()).hexdigest())
            ]
            for field in ("template_family", "source_group"):
                if record.get(field):
                    identities.append((field, record[field]))
            identities.extend(
                ("entity_value", span["entity_value"].casefold())
                for span in record["spans"]
            )
            for field, value in identities:
                previous = owners[field].setdefault(value, split)
                if previous != split:
                    raise ValueError(
                        f"Cross-split {field} reuse: {previous} and {split}"
                    )


def read_jsonl(path):
    return [
        json.loads(line) for line in Path(path).read_text().split("\n") if line.strip()
    ]
