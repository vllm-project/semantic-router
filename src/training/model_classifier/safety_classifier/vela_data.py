"""Vela prompt safety and unconditional multi-label hazard supervision.

AEGIS categories describe a dialogue. Category targets are therefore eligible
only for prompt-only rows or unsafe prompts whose response is labelled safe.
Binary prompt labels do not require this restriction. Source splits are kept;
normalized duplicate prompts belong to the most restrictive held-out split.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from ..sequence_repair.data import normalized_text

MIN_TRAIN_PROMPT_CHARS = 25

LABELS = (
    "violence",
    "criminal_activity",
    "sexual_content",
    "child_exploitation",
    "hate",
    "harassment_abuse",
    "regulated_substances",
    "weapons",
    "self_harm",
    "privacy",
    "specialized_advice",
    "misinformation",
)
MAPPING = {
    "Violence": "violence",
    "Threat": "violence",
    "Criminal Planning/Confessions": "criminal_activity",
    "Illegal Activity": "criminal_activity",
    "Fraud/Deception": "criminal_activity",
    "Malware": "criminal_activity",
    "Sexual": "sexual_content",
    "Sexual (minor)": "child_exploitation",
    "Hate/Identity Hate": "hate",
    "Harassment": "harassment_abuse",
    "Profanity": "harassment_abuse",
    "Controlled/Regulated Substances": "regulated_substances",
    "Guns and Illegal Weapons": "weapons",
    "Suicide and Self Harm": "self_harm",
    "PII/Privacy": "privacy",
    "Unauthorized Advice": "specialized_advice",
    "Political/Misinformation/Conspiracy": "misinformation",
}
UNSUPPORTED = {
    "Needs Caution",
    "Other",
    "Immoral/Unethical",
    "Copyright/Trademark/Plagiarism",
    "High Risk Gov Decision Making",
    "Manipulation",
}
SOURCE_REVISION = "d86bb8bedff51d25ac834ab7838f1cc61acb7a2c"
SOURCE_SHA256 = {
    "train": "154fba82c71d9fa73abd2ca5588a198e693ddc816c83444df180a22f613e02f6",
    "validation": "a97200e226ad4f6ba6a639982f817f675909bc74513859df3e8fc9a92951dfcd",
    "test": "b0a6d602260524866053cb34105194f074f2c2906e3691b68d43b9e6e9318f35",
}


def hazard_target(row):
    """Return observed risk targets, or a reason category attribution is unsafe."""
    categories = {
        part.strip() for part in row["violated_categories"].split(",") if part.strip()
    }
    unknown = categories - set(MAPPING) - UNSUPPORTED
    if unknown:
        raise ValueError(f"Unreviewed AEGIS categories: {sorted(unknown)}")
    if row["prompt_label"] == "safe":
        # Source category mentions can describe topics, including benign quotes.
        # The human prompt-level safe label takes precedence for risk targets.
        return [0] * len(LABELS), [1] * len(LABELS), None
    if row.get("response") and row.get("response_label") != "safe":
        return None, None, "dialogue category cannot be attributed to prompt"
    positives = {MAPPING[category] for category in categories if category in MAPPING}
    if not positives:
        return None, None, "unsafe prompt has no supported category annotation"
    # This source annotates the whole declared taxonomy; absence is an observed
    # negative only inside that scope. Partial-taxonomy sources must supply masks.
    return [int(label in positives) for label in LABELS], [1] * len(LABELS), None


def partial_target(positive_labels, observed_labels):
    positive, observed = set(positive_labels), set(observed_labels)
    if not positive <= observed or not observed <= set(LABELS):
        raise ValueError("Positive labels must be observed and inside the taxonomy")
    if not observed:
        raise ValueError("A masked example needs at least one observed label")
    return [int(label in positive) for label in LABELS], [
        int(label in observed) for label in LABELS
    ]


def build(source_dir):
    binary, hazard = {}, {}
    seen, audit = set(), Counter()
    for split in ["test", "validation", "train"]:
        binary[split], hazard[split] = [], []
        payload = (Path(source_dir) / f"{split}.json").read_bytes()
        if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256[split]:
            raise ValueError(f"AEGIS source does not match pinned revision: {split}")
        for row in json.loads(payload):
            text = row["prompt"].strip()
            if not text or text == "REDACTED":
                audit[f"{split}:empty_or_redacted"] += 1
                continue
            fingerprint = hashlib.sha256(normalized_text(text).encode()).hexdigest()
            if fingerprint in seen:
                audit[f"{split}:normalized_duplicate"] += 1
                continue
            seen.add(fingerprint)
            if row["prompt_label"] not in {"safe", "unsafe"}:
                raise ValueError("Unknown binary label")
            item = {
                "id": f"aegis:{row['id']}",
                "group_id": f"aegis:{fingerprint}",
                "text": text,
                "label": row["prompt_label"],
                "source": "aegis2",
                "source_revision": SOURCE_REVISION,
                "source_split": split,
                "language": "source_unspecified",
                "length_bucket": "natural_short",
            }
            if split != "train" or len(text) >= MIN_TRAIN_PROMPT_CHARS:
                binary[split].append(item)
            else:
                audit["train:short_context_ambiguous_binary"] += 1
            targets, mask, reason = hazard_target(row)
            if reason:
                audit[f"{split}:{reason}"] += 1
                continue
            hazard[split].append({**item, "targets": targets, "label_mask": mask})
    return binary, hazard, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite a materialized corpus")
    binary, hazard, audit = build(args.source)
    manifest = {
        "source": "nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
        "revision": SOURCE_REVISION,
        "license": "cc-by-4.0",
        "taxonomy": "vela-hazard-12-v1",
        "labels": LABELS,
        "unsupported_source_categories": sorted(UNSUPPORTED),
        "exclusions": dict(audit),
        "files": {},
    }
    for task, splits in [("safety", binary), ("hazard", hazard)]:
        output = args.output / task
        output.mkdir(parents=True)
        labels = ["safe", "unsafe"] if task == "safety" else LABELS
        config = {
            "id2label": dict(enumerate(labels)),
            "label2id": {label: i for i, label in enumerate(labels)},
            "problem_type": (
                "single_label_classification"
                if task == "safety"
                else "multi_label_classification"
            ),
        }
        (output / "contract.json").write_text(json.dumps(config, indent=2) + "\n")
        for split, rows in splits.items():
            path = output / f"{split}.jsonl"
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            )
            info = {
                "rows": len(rows),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            if task == "hazard":
                info["supervision"] = {
                    label: {
                        "positive": sum(row["targets"][i] for row in rows),
                        "negative": sum(
                            row["label_mask"][i] and not row["targets"][i]
                            for row in rows
                        ),
                        "unknown": sum(not row["label_mask"][i] for row in rows),
                    }
                    for i, label in enumerate(labels)
                }
            manifest["files"][f"{task}/{split}.jsonl"] = info
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
