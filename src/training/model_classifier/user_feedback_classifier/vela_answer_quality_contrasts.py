"""Freeze authored current-user feedback contrasts before development prediction."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from ..sequence_repair.data import assert_disjoint, normalized_text
from .vela_contract import VELA_ID2LABEL, VELA_LABEL2ID

DEFAULT_REGISTRY = (
    Path(__file__).parent / "configs" / "vela-answer-quality-families-v1.json"
)
FAMILY_COUNTS = {"train": 20, "dev": 10}
LANGUAGES = ("en", "zh")
SOURCE = "authored_feedback_answer_quality_v1"


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def family_plan(registry):
    return [
        {key: family[key] for key in ("family_id", "split", "intent")}
        for family in registry["families"]
    ]


def build(registry):
    """Validate annotations and grouping; this does not infer gold from wording."""
    labels = list(VELA_ID2LABEL.values())
    if set(registry["label_rules"]) != set(labels) or any(
        not isinstance(rule, str) or not rule.strip()
        for rule in registry["label_rules"].values()
    ):
        raise ValueError("The reviewed protocol must define all five feedback labels")
    for field in ("input_protocol", "authorship", "evidence_scope", "precedence"):
        if not isinstance(registry.get(field), str) or not registry[field].strip():
            raise ValueError(
                "Authored contrasts require explicit protocol and provenance"
            )
    if (
        registry.get("final_data_read") is not False
        or registry.get("model_predictions_used") is not False
    ):
        raise ValueError(
            "Authoring must precede model predictions and exclude final data"
        )
    if registry["family_plan_sha256"] != sha256(json_bytes(family_plan(registry))):
        raise ValueError("Family assignments changed from the frozen plan")
    records = {split: [] for split in FAMILY_COUNTS}
    names, texts, counts = set(), set(), Counter()
    for family in registry["families"]:
        name, split = family["family_id"], family["split"]
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]*", name)
            or name in names
            or split not in records
            or not isinstance(family.get("intent"), str)
            or not family["intent"].strip()
        ):
            raise ValueError("Each distinct intent family needs one explicit partition")
        names.add(name)
        counts[split] += 1
        if [variant["label"] for variant in family["variants"]] != labels:
            raise ValueError(
                "Each family must contain every ordered feedback label once"
            )
        for variant in family["variants"]:
            if (
                not isinstance(variant.get("rationale"), str)
                or not variant["rationale"].strip()
            ):
                raise ValueError("Every label requires an explicit full-text rationale")
            for language in LANGUAGES:
                text = variant.get(language)
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(
                        "Every intention requires both English and Chinese"
                    )
                normalized = normalized_text(text)
                if normalized in texts:
                    raise ValueError(
                        "Authored texts must be distinct after normalization"
                    )
                texts.add(normalized)
                records[split].append(
                    {
                        "id": f"vela-feedback-quality:{name}:{variant['label']}:{language}",
                        "group_id": f"vela-feedback-quality:{name}",
                        "text": text,
                        "text_sha256": sha256(text.encode("utf-8")),
                        "label": variant["label"],
                        "source": SOURCE,
                        "source_split": split,
                        "language": language,
                        "length_bucket": "authored_short",
                        "intent_family": family["intent"],
                        "review_reason": variant["rationale"],
                        "review_protocol": registry["version"],
                        "annotation_provenance": registry["authorship"],
                        "license": registry["license"],
                    }
                )
    if dict(counts) != FAMILY_COUNTS:
        raise ValueError(
            "The frozen corpus requires 20 training and 10 development families"
        )
    assert_disjoint(records["train"], records["dev"])
    return records


def summary(rows):
    return {
        "rows": len(rows),
        "groups": len({row["group_id"] for row in rows}),
        "labels": dict(Counter(row["label"] for row in rows)),
        "languages": dict(Counter(row["language"] for row in rows)),
        "label_language_counts": {
            label: dict(
                Counter(row["language"] for row in rows if row["label"] == label)
            )
            for label in VELA_LABEL2ID
        },
    }


def freeze(output, registry_path=DEFAULT_REGISTRY):
    output, registry_path = Path(output), Path(registry_path)
    if output.exists():
        raise FileExistsError("Refusing to overwrite frozen data or evidence")
    registry_data = registry_path.read_bytes()
    registry = json.loads(registry_data)
    records = build(registry)
    contract = {
        "id2label": VELA_ID2LABEL,
        "label2id": VELA_LABEL2ID,
        "problem_type": "single_label_classification",
        "input_protocol": registry["input_protocol"],
    }
    artifacts = {
        f"{split}.jsonl": "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows
        ).encode("utf-8")
        for split, rows in records.items()
    }
    artifacts["contract.json"] = json_bytes(contract)
    artifacts["family-plan.json"] = json_bytes(family_plan(registry))
    artifacts["review.json"] = json_bytes(
        {
            "reviewer": registry["authorship"],
            "protocol": {
                key: registry[key]
                for key in ("input_protocol", "label_rules", "precedence")
            },
            "rows": [
                {
                    key: row[key]
                    for key in (
                        "id",
                        "group_id",
                        "source_split",
                        "language",
                        "text",
                        "text_sha256",
                        "label",
                        "review_reason",
                    )
                }
                for rows in records.values()
                for row in rows
            ],
        }
    )
    manifest = {
        "version": registry["version"],
        "license": registry["license"],
        "authorship": registry["authorship"],
        "scope": registry["evidence_scope"],
        "split_policy": registry["split_policy"],
        "registry_sha256": sha256(registry_data),
        "builder_sha256": sha256(Path(__file__).read_bytes()),
        "family_plan_sha256": registry["family_plan_sha256"],
        "partitions": {split: summary(rows) for split, rows in records.items()},
        "files": {name: sha256(data) for name, data in artifacts.items()},
        "model_predictions_used": False,
        "final_data_read": False,
        "limitation": "Scenario families are disjoint; broad writing and evaluation concepts can recur across splits. These are authored contrasts, not an external natural benchmark.",
    }
    output.mkdir(parents=True)
    for name, data in artifacts.items():
        (output / name).write_bytes(data)
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.output, args.registry), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
