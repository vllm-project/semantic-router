"""Freeze original bilingual hazard contrasts with explicit full-text annotations."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from ..sequence_repair.data import assert_disjoint, normalized_text
from .vela_data import LABELS

CATEGORIES = ("specialized_advice", "misinformation")
MINIMUM_POSITIVE_FAMILIES = {"train": 20, "validation": 10}
CONFIG_DIR = Path(__file__).parent / "configs"
DEFAULT_REGISTRY = CONFIG_DIR / "hazard-boundary-contrasts-v1.json"
DEFAULT_RUBRIC = CONFIG_DIR / "hazard-rubric-v1.json"


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def family_plan(registry):
    return [
        {key: family[key] for key in ("family_id", "category", "split", "intent")}
        for family in registry["families"]
    ]


def _annotation(variant, category):
    positive = variant["positive_labels"]
    unknown = variant["unknown_labels"]
    if (
        not isinstance(positive, list)
        or not isinstance(unknown, list)
        or len(set(positive)) != len(positive)
        or len(set(unknown)) != len(unknown)
        or set(positive) & set(unknown)
        or (set(positive) | set(unknown)) - set(LABELS)
    ):
        raise ValueError(
            "Every annotation needs valid, distinct positive/unknown labels"
        )
    if (
        not isinstance(variant.get("rationale"), str)
        or not variant["rationale"].strip()
    ):
        raise ValueError("Every authored variant requires a full-text review rationale")
    if set(variant.get("unknown_rationale", {})) != set(unknown) or any(
        not isinstance(reason, str) or not reason.strip()
        for reason in variant.get("unknown_rationale", {}).values()
    ):
        raise ValueError("Every unknown label requires an uncertainty rationale")
    if variant["name"] == "positive":
        if category not in positive:
            raise ValueError("A positive contrast must support its target category")
    elif positive or unknown:
        raise ValueError(
            "Reviewed safe contrasts require all 12 observed negative labels"
        )
    return [int(label in positive) for label in LABELS], [
        int(label not in unknown) for label in LABELS
    ]


def build(registry, rubric):
    if tuple(rubric["labels"]) != LABELS:
        raise ValueError("The rubric label order must match the hazard output contract")
    if registry["family_plan_sha256"] != _sha(_json_bytes(family_plan(registry))):
        raise ValueError("Family assignments differ from the frozen family plan")
    result = {split: [] for split in MINIMUM_POSITIVE_FAMILIES}
    families, texts = set(), set()
    positive_families = {split: Counter() for split in result}
    for family in registry["families"]:
        name, category, split = (
            family["family_id"],
            family["category"],
            family["split"],
        )
        if (
            not re.fullmatch(r"[a-z][a-z0-9_]*", name)
            or name in families
            or category not in CATEGORIES
            or split not in result
            or not family.get("intent", "").strip()
        ):
            raise ValueError(
                "Families need distinct IDs, valid categories and one split"
            )
        families.add(name)
        names = [variant["name"] for variant in family["variants"]]
        if names not in (["positive", "near_miss"], ["safe_anchor"]):
            raise ValueError(
                "A family needs one positive/near-miss pair or one safe anchor"
            )
        if names[0] == "positive":
            positive_families[split][category] += 1
        for variant in family["variants"]:
            targets, mask = _annotation(variant, category)
            for language in ("en", "zh"):
                text = variant.get(language)
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Each variant needs both English and Chinese text")
                normalized = normalized_text(text)
                if normalized in texts:
                    raise ValueError(
                        "Authored texts must be distinct after normalization"
                    )
                texts.add(normalized)
                result[split].append(
                    {
                        "id": f"vela-boundary:{name}:{variant['name']}:{language}",
                        "group_id": f"vela-boundary:{name}",
                        "text": text,
                        "text_sha256": _sha(text.encode("utf-8")),
                        "label": "unsafe" if any(targets) else "safe",
                        "targets": list(targets),
                        "label_mask": list(mask),
                        "source": "authored_hazard_boundary_contrasts_v1",
                        "source_split": split,
                        "language": language,
                        "length_bucket": "authored_short",
                        "contrast_category": category,
                        "contrast_variant": variant["name"],
                        "intent_family": family["intent"],
                        "review_reason": variant["rationale"],
                        "unknown_rationale": dict(variant["unknown_rationale"]),
                        "review_protocol": registry["version"],
                        "annotation_provenance": registry["authorship"],
                        "license": registry["license"],
                    }
                )
    assert_disjoint(result["train"], result["validation"])
    for split, minimum in MINIMUM_POSITIVE_FAMILIES.items():
        if any(positive_families[split][label] < minimum for label in CATEGORIES):
            raise ValueError(
                f"{split} needs {minimum} distinct positive families per category"
            )
    return result


def _summary(rows):
    observed = {
        label: {
            "positive": sum(row["targets"][index] for row in rows),
            "negative": sum(
                row["label_mask"][index] and not row["targets"][index] for row in rows
            ),
            "unknown": sum(not row["label_mask"][index] for row in rows),
        }
        for index, label in enumerate(LABELS)
    }
    return {
        "rows": len(rows),
        "groups": len({row["group_id"] for row in rows}),
        "binary_labels": dict(Counter(row["label"] for row in rows)),
        "languages": dict(Counter(row["language"] for row in rows)),
        "observed_label_counts": observed,
        "observed_count_histogram": dict(
            Counter(sum(row["label_mask"]) for row in rows)
        ),
        "positive_contrast_families": {
            category: len(
                {
                    row["group_id"]
                    for row in rows
                    if row["contrast_category"] == category
                    and row["contrast_variant"] == "positive"
                }
            )
            for category in CATEGORIES
        },
    }


def freeze(output, registry_path=DEFAULT_REGISTRY, rubric_path=DEFAULT_RUBRIC):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite frozen authored data")
    registry_path, rubric_path = Path(registry_path), Path(rubric_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    rubric = json.loads(rubric_path.read_text(encoding="utf-8"))
    result = build(registry, rubric)
    output.mkdir(parents=True)
    files = {}
    reviews = []
    for split, rows in result.items():
        data = "".join(
            json.dumps(row, ensure_ascii=True) + "\n" for row in rows
        ).encode()
        filename = f"{split}.jsonl"
        (output / filename).write_bytes(data)
        files[filename] = {"sha256": _sha(data), **_summary(rows)}
        for row in rows:
            reviews.append(
                {
                    **{
                        key: row[key]
                        for key in (
                            "id",
                            "group_id",
                            "text",
                            "text_sha256",
                            "source_split",
                            "language",
                            "contrast_category",
                            "contrast_variant",
                        )
                    },
                    "binary_label": row["label"],
                    "label_review": {
                        label: (
                            "unknown"
                            if not row["label_mask"][index]
                            else "positive" if row["targets"][index] else "negative"
                        )
                        for index, label in enumerate(LABELS)
                    },
                    "rationale": row["review_reason"],
                    "unknown_rationale": row["unknown_rationale"],
                    "reviewer": registry["authorship"],
                }
            )
    extras = {
        "family-plan.json": family_plan(registry),
        "review.json": reviews,
        "contract.json": {
            "id2label": {str(index): label for index, label in enumerate(LABELS)},
            "label2id": {label: index for index, label in enumerate(LABELS)},
            "problem_type": "multi_label_classification",
            "binary_label2id": {"safe": 0, "unsafe": 1},
            "observed_label_mask_required": True,
        },
    }
    for filename, value in extras.items():
        data = _json_bytes(value)
        (output / filename).write_bytes(data)
        files[filename] = {"sha256": _sha(data)}
    manifest = {
        "version": registry["version"],
        "registry_sha256": _sha(registry_path.read_bytes()),
        "rubric_sha256": _sha(rubric_path.read_bytes()),
        "builder_sha256": _sha(Path(__file__).read_bytes()),
        "family_plan_sha256": registry["family_plan_sha256"],
        "labels": list(LABELS),
        "license": registry["license"],
        "authorship": registry["authorship"],
        "evidence_scope": registry["evidence_scope"],
        "split_policy": registry["split_policy"],
        "review_policy": registry["review_policy"],
        "model_predictions_used_for_authoring": False,
        "final_data_read": False,
        "files": files,
    }
    (output / "manifest.json").write_bytes(_json_bytes(manifest))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC)
    args = parser.parse_args()
    print(json.dumps(freeze(args.output, args.registry, args.rubric)), flush=True)


if __name__ == "__main__":
    main()
