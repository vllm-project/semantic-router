"""Stratified long-context training with explicit backgrounds and source groups.

Hazard coverage follows each observed positive risk, not the binary unsafe
label. This builder creates training data only; it never evaluates a model.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .safety_classifier.vela_data import LABELS
from .sequence_repair.context import insert_payload

MAX_PAYLOAD_TOKENS = 768
MIN_BACKGROUNDS = 2


def strata(row, task):
    language = row.get("language", "en")
    if language in {"source_unspecified", "unspecified"}:
        language = "en"
    if language not in {"en", "zh"}:
        return []
    if task == "hazard" and row["label"] != "safe":
        if len(row["targets"]) != len(LABELS) or len(row["label_mask"]) != len(LABELS):
            raise ValueError("Hazard training targets differ from the fixed taxonomy")
        labels = [
            label
            for label, positive, observed in zip(
                LABELS, row["targets"], row["label_mask"], strict=True
            )
            if positive and observed
        ]
    else:
        labels = [row["label"]]
    return [(label, language) for label in labels]


def select_payloads(rows, task, per_stratum, count_tokens):
    if per_stratum <= 0:
        raise ValueError("Positive family budget required")
    counts, selected, family_strata = Counter(), [], set()
    for row in sorted(
        rows, key=lambda item: hashlib.sha256(item["id"].encode()).hexdigest()
    ):
        eligible = [
            key
            for key in strata(row, task)
            if counts[key] < per_stratum and (row["group_id"], key) not in family_strata
        ]
        if not eligible or count_tokens(row["text"]) > MAX_PAYLOAD_TOKENS:
            continue
        selected.append(row)
        for key in eligible:
            counts[key] += 1
            family_strata.add((row["group_id"], key))
    if not selected:
        raise ValueError("No eligible independent training groups")
    return selected, {
        f"{label}:{language}": count
        for (label, language), count in sorted(counts.items())
    }


def main():
    from transformers import AutoTokenizer  # noqa: PLC0415

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--task", choices=["sequence", "hazard"], required=True)
    parser.add_argument("--backgrounds", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--per-stratum", type=int, default=8)
    parser.add_argument(
        "--budgets", nargs="+", type=int, default=[1024, 4096, 8192, 16384, 32768]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite training curriculum")
    backgrounds = json.loads(args.backgrounds.read_text())
    for language in ["en", "zh"]:
        values = backgrounds.get(language)
        if (
            not isinstance(values, list)
            or len(values) < MIN_BACKGROUNDS
            or any(not isinstance(value, str) or not value.strip() for value in values)
        ):
            raise ValueError(
                "At least two explicit nonempty backgrounds per language are required"
            )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def count(text):
        return len(
            tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        )

    rows = [
        json.loads(line)
        for path in args.input
        for line in path.read_text().split("\n")
        if line.strip()
    ]
    selected, coverage = select_payloads(rows, args.task, args.per_stratum, count)
    output = []
    for index, row in enumerate(selected):
        language = strata(row, args.task)[0][1]
        background_id = index % len(backgrounds[language])
        background = backgrounds[language][background_id]
        for budget in args.budgets:
            for position in ["head", "middle", "tail"]:
                built = insert_payload(row["text"], background, position, budget, count)
                output.append(
                    {
                        **row,
                        "id": f'vela-context-v2:{row["id"]}:{background_id}:{budget}:{position}',
                        "text": built["text"],
                        "source": "authored_training_context_v2_on_source_payload",
                        "source_payload_id": row["id"],
                        "source_payload_source": row["source"],
                        "language": language,
                        "background_id": background_id,
                        "length_bucket": str(budget),
                        "position": position,
                        "actual_tokens": built["actual_tokens"],
                        "payload_start": built["payload_start"],
                        "payload_end": built["payload_end"],
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in output)
    )
    receipt = {
        "scope": "Training-only source payloads and explicit original backgrounds; no quality claim",
        "rows": len(output),
        "source_payloads": len(selected),
        "source_groups": len({row["group_id"] for row in selected}),
        "stratum_group_coverage": coverage,
        "per_stratum_requested": args.per_stratum,
        "background_sha256": hashlib.sha256(args.backgrounds.read_bytes()).hexdigest(),
        "input_files": [
            {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in args.input
        ],
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
