"""Freeze Safety context curricula without changing payload labels or truncating them."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from ..sequence_repair.context import insert_payload
from ..sequence_repair.data import (
    assert_disjoint,
    file_receipts,
    normalized_text,
    read_records,
)

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "vela-safety-context-v1.json"
LABELS = {"safe": 0, "unsafe": 1}
POSITIONS = ("head", "middle", "tail")
CONTEXT_SOURCE = "authored_safety_context_v1"
QUOTE_SOURCE = "authored_safety_quote_contrasts_v1"
AUTHORED_ROW_COUNTS = {"train": 32, "dev": 16}
MINIMUM_BACKGROUNDS = 2


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def authored_rows(config):
    plan = [
        {key: family[key] for key in ("family_id", "split")}
        for family in config["families"]
    ]
    if sha256(json_bytes(plan)) != config["family_plan_sha256"]:
        raise ValueError("Authored family assignments changed")
    rows = {"train": [], "dev": []}
    names, texts = set(), set()
    for family in config["families"]:
        name, split = family["family_id"], family["split"]
        if name in names or not name or split not in rows:
            raise ValueError("Distinct families require explicit train/dev assignments")
        names.add(name)
        if [(item["variant"], item["label"]) for item in family["variants"]] != [
            ("endorsed_quote", "unsafe"),
            ("protective_quote", "safe"),
        ]:
            raise ValueError(
                "Every family needs explicit endorsement/protection labels"
            )
        for item in family["variants"]:
            if not item.get("rationale", "").strip():
                raise ValueError("Complete-text labels require review rationales")
            for language in ("en", "zh"):
                text = item.get(language)
                if (
                    not isinstance(text, str)
                    or not text.strip()
                    or normalized_text(text) in texts
                ):
                    raise ValueError("Distinct bilingual texts are required")
                texts.add(normalized_text(text))
                rows[split].append(
                    {
                        "id": f"vela-safety-quote:{name}:{item['variant']}:{language}",
                        "group_id": f"vela-safety-quote:{name}",
                        "text": text,
                        "text_sha256": sha256(text.encode()),
                        "label": item["label"],
                        "language": language,
                        "source": QUOTE_SOURCE,
                        "source_split": split,
                        "length_bucket": "authored_short",
                        "review_reason": item["rationale"],
                        "review_protocol": config["version"],
                        "annotation_provenance": config["authorship"],
                        "license": "CC0-1.0",
                    }
                )
    if {split: len(part) for split, part in rows.items()} != AUTHORED_ROW_COUNTS:
        raise ValueError("Expected eight training and four development families")
    assert_disjoint(rows["train"], rows["dev"])
    return rows


def validate_backgrounds(config):
    backgrounds = config["backgrounds"]
    if len(backgrounds) < MINIMUM_BACKGROUNDS or len(
        {item["family_id"] for item in backgrounds}
    ) != len(backgrounds):
        raise ValueError("Multiple distinct background families are required")
    for item in backgrounds:
        if not all(
            isinstance(item.get(language), str) and item[language].strip()
            for language in ("en", "zh")
        ):
            raise ValueError("Background families need both languages")
    return backgrounds


def contextualize(rows, config, budgets, count_tokens):
    """Label/group-independent placement; complete neutral paragraphs only."""
    backgrounds = validate_backgrounds(config)
    if (
        not budgets
        or len(set(budgets)) != len(budgets)
        or any(type(budget) is not int or budget < 1 for budget in budgets)
    ):
        raise ValueError("Distinct positive token budgets are required")
    result, excluded = [], []
    seen = set()
    for row in rows:
        if (
            row.get("source_split") != "train"
            or row.get("label") not in LABELS
            or not row.get("group_id")
        ):
            raise ValueError(
                "Only explicitly grouped, labelled training payloads may enter a curriculum"
            )
        if row["id"] in seen:
            raise ValueError("Duplicate training payload ID")
        seen.add(row["id"])
        language = row.get("language")
        if language not in {"en", "zh", "source_unspecified"}:
            raise ValueError("Context language must be explicitly supported")
        background_language = "en" if language == "source_unspecified" else language
        payload_tokens = count_tokens(row["text"])
        for budget in budgets:
            key = f"{row['group_id']}:{budget}"
            background = backgrounds[
                int(sha256(("background:" + key).encode()), 16) % len(backgrounds)
            ]
            position = POSITIONS[
                int(sha256(("position:" + key).encode()), 16) % len(POSITIONS)
            ]
            if payload_tokens > budget:
                excluded.append(
                    {
                        "id": row["id"],
                        "budget": budget,
                        "reason": "payload exceeds budget",
                        "payload_tokens": payload_tokens,
                    }
                )
                continue
            built = insert_payload(
                row["text"],
                background[background_language],
                position,
                budget,
                count_tokens,
                fill_to_budget=False,
            )
            if built["actual_tokens"] == payload_tokens:
                excluded.append(
                    {
                        "id": row["id"],
                        "budget": budget,
                        "reason": "no complete background paragraph fits",
                        "payload_tokens": payload_tokens,
                    }
                )
                continue
            result.append(
                {
                    **row,
                    **built,
                    "id": f"safety-context:{budget}:{row['id']}",
                    "text_sha256": sha256(built["text"].encode()),
                    "parent_id": row["id"],
                    "parent_text_sha256": sha256(row["text"].encode()),
                    "parent_source": row["source"],
                    "source": CONTEXT_SOURCE,
                    "length_bucket": str(budget),
                    "position": position,
                    "background_family": background["family_id"],
                    "background_language": background_language,
                    "background_license": "CC0-1.0",
                    "payload_label_policy": "Retain the reviewed training payload label; background is neutral task-irrelevant text. No automatic quotation or label reversal.",
                }
            )
    return result, excluded


def counts(rows):
    return {
        "rows": len(rows),
        "groups": len({row["group_id"] for row in rows}),
        **{
            field: dict(Counter(str(row.get(field, "unspecified")) for row in rows))
            for field in (
                "label",
                "language",
                "source",
                "length_bucket",
                "background_family",
                "position",
            )
        },
    }


def freeze(
    output,
    config_path,
    replay_paths,
    context_paths,
    dev_paths,
    budgets,
    count_tokens,
    replay_budget=2048,
):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite frozen training/development data")
    config_data = Path(config_path).read_bytes()
    config = json.loads(config_data)
    authored = authored_rows(config)
    replay = read_records(replay_paths, LABELS)
    if any(row.get("source_split") != "train" for row in replay):
        raise ValueError("Replay inputs must contain training rows only")
    replay_kept, replay_excluded = [], []
    for row in replay:
        tokens = count_tokens(row["text"])
        if tokens > replay_budget:
            replay_excluded.append(
                {
                    "id": row["id"],
                    "tokens": tokens,
                    "reason": "unchanged short replay eligibility budget",
                }
            )
        else:
            replay_kept.append(row)
    context_payloads = read_records(context_paths, LABELS) + authored["train"]
    context, context_excluded = contextualize(
        context_payloads, config, budgets, count_tokens
    )
    dev = read_records(dev_paths, LABELS)
    assert_disjoint(replay_kept + context_payloads + context, dev + authored["dev"])
    partitions = {
        "replay-train": replay_kept,
        "authored-train": authored["train"],
        "authored-dev": authored["dev"],
        "context-train": context,
    }
    files = {
        f"{name}.jsonl": "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows
        ).encode()
        for name, rows in partitions.items()
    }
    files["exclusions.json"] = json_bytes(
        {
            "replay": replay_excluded,
            "context": context_excluded,
            "policy": "Retain original inputs and gold. These rows do not fit a declared training condition; they are not relabelled or silently truncated.",
        }
    )
    manifest = {
        "version": config["version"],
        "config_sha256": sha256(config_data),
        "builder_sha256": sha256(Path(__file__).read_bytes()),
        "family_plan_sha256": config["family_plan_sha256"],
        "inputs": {
            "replay": file_receipts(replay_paths),
            "context_payloads": file_receipts(context_paths),
            "unchanged_development": file_receipts(dev_paths),
        },
        "replay_budget": replay_budget,
        "context_budgets": budgets,
        "partitions": {name: counts(rows) for name, rows in partitions.items()},
        "files": {name: sha256(data) for name, data in files.items()},
        "license": config["license"],
        "scope": "Original authored context and quotation training contrasts; original payload licences/provenance remain binding. Development groups and texts are excluded from training. Development diagnostics inform curriculum design, but models do not assign gold labels. Repeated neutral paragraphs are synthetic context, not natural long-document benchmark evidence.",
        "candidate_predictions_used_to_assign_labels": False,
        "development_diagnostics_informed_curriculum_design": True,
        "final_data_used": False,
    }
    output.mkdir(parents=True)
    for name, data in files.items():
        (output / name).write_bytes(data)
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    return manifest


def main():
    # Heavy dependency is needed only to measure the actual tokenizer budget.
    from transformers import AutoTokenizer  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--replay-inputs", nargs="+", required=True)
    parser.add_argument("--context-inputs", nargs="+", required=True)
    parser.add_argument("--development-inputs", nargs="+", required=True)
    parser.add_argument(
        "--budgets", nargs="+", type=int, default=[256, 512, 2048, 4096]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def count_tokens(text):
        return len(tokenizer(text, truncation=False, padding=False)["input_ids"])

    manifest = freeze(
        args.output,
        args.config,
        args.replay_inputs,
        args.context_inputs,
        args.development_inputs,
        args.budgets,
        count_tokens,
    )
    print(
        json.dumps(
            {"partitions": manifest["partitions"], "files": manifest["files"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
