# Heavy inference dependencies are lazy so data-contract tests stay dependency-light.
# ruff: noqa: PLC0415
"""Add tokenizer-measured length/position stress cases to frozen PII splits."""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

# Keep the existing direct-script CLI while sharing only the generic context helper.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_repair_data import NEGATIVES, write_jsonl
from sequence_repair.context import insert_payload
from span_data import read_jsonl


def concatenate(records):
    text, spans = "", []
    for record in records:
        if text:
            text += "\n"
        offset = len(text)
        spans.extend(
            {
                **span,
                "start_position": span["start_position"] + offset,
                "end_position": span["end_position"] + offset,
            }
            for span in record["spans"]
        )
        text += record["full_text"]
    return {"full_text": text, "spans": spans}


def place_needle(needle, filler, position, budget, count_tokens):
    result = insert_payload(
        needle["full_text"],
        filler,
        position,
        budget,
        count_tokens,
        fill_to_budget=False,
    )
    offset = result["payload_start"]
    return {
        "full_text": result["text"],
        "spans": [
            {
                **span,
                "start_position": span["start_position"] + offset,
                "end_position": span["end_position"] + offset,
            }
            for span in needle["spans"]
        ],
    }


def pack_records(pool, budget, count_tokens, rng):
    candidates = list(pool)
    rng.shuffle(candidates)
    selected, estimated, cursor = [], 2, 0
    while estimated < budget:
        record = candidates[cursor % len(candidates)]
        estimated += count_tokens(record["full_text"]) + 1
        if estimated <= budget:
            selected.append(record)
        cursor += 1
    if not selected:
        raise ValueError("No complete source record fits in the packing budget")
    result = concatenate(selected)
    while selected and count_tokens(result["full_text"]) > budget:
        selected.pop()
        result = concatenate(selected)
    return result, [record["id"] for record in selected]


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=[4096, 8192, 16384, 32768]
    )
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite frozen long-context data")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def count_tokens(text):
        return len(
            tokenizer(text, add_special_tokens=True, truncation=False, padding=False)[
                "input_ids"
            ]
        )

    args.output.mkdir(parents=True, exist_ok=True)
    files = []
    for split_index, split in enumerate(["train", "dev", "test"]):
        rows = read_jsonl(args.corpus / f"{split}.jsonl")
        for budget in args.budgets:
            result = []
            for language in NEGATIVES:
                pool = [
                    row
                    for row in rows
                    if row["language"] == language and row["kind"] == "positive"
                ]
                needles = [
                    row
                    for row in pool
                    if row["spans"][0]["entity_type"] == "EMAIL_ADDRESS"
                ]
                filler = NEGATIVES[language][split_index]
                for position_index, position in enumerate(["head", "middle", "tail"]):
                    needle = needles[position_index % len(needles)]
                    record = place_needle(
                        needle, filler, position, budget, count_tokens
                    )
                    record.update(
                        id=f"long-{split}-{language}-{budget}-{position}",
                        language=language,
                        position=position,
                        kind="needle-stress",
                        parent_ids=[needle["id"]],
                        source_group=f"long-{split}-{language}",
                        split=split,
                        length_bucket=budget,
                    )
                    result.append(record)
                packed, parents = pack_records(
                    pool,
                    budget,
                    count_tokens,
                    random.Random(args.seed + budget + split_index),
                )
                packed.update(
                    id=f"long-{split}-{language}-{budget}-packed",
                    language=language,
                    position="multiple",
                    kind="packed-synthetic",
                    parent_ids=parents,
                    source_group=f"long-{split}-{language}",
                    split=split,
                    length_bucket=budget,
                )
                result.append(packed)
                empty = {"full_text": filler, "spans": []}
                negative = place_needle(empty, filler, "middle", budget, count_tokens)
                negative.update(
                    id=f"long-{split}-{language}-{budget}-negative",
                    language=language,
                    position="none",
                    kind="negative-stress",
                    parent_ids=[f"negative-{split}-{language}"],
                    source_group=f"long-{split}-{language}",
                    split=split,
                    length_bucket=budget,
                )
                result.append(negative)
            for record in result:
                record["actual_tokens"] = count_tokens(record["full_text"])
                if record["actual_tokens"] > budget:
                    raise ValueError(
                        "Generated input exceeds its measured token budget"
                    )
                # A neutral suffix fills the last few tokens without cutting
                # any annotated entity or claiming padding as attended text.
                record["full_text"] += " note" * (budget - record["actual_tokens"])
                record["actual_tokens"] = count_tokens(record["full_text"])
                if record["actual_tokens"] != budget:
                    raise ValueError(
                        "This tokenizer does not encode the neutral suffix as one token"
                    )
                for span in record["spans"]:
                    if (
                        record["full_text"][
                            span["start_position"] : span["end_position"]
                        ]
                        != span["entity_value"]
                    ):
                        raise ValueError("Concatenation corrupted a span")
            item = write_jsonl(args.output / f"{split}-{budget}.jsonl", result)
            item.update(
                actual_tokens_min=min(record["actual_tokens"] for record in result),
                actual_tokens_max=max(record["actual_tokens"] for record in result),
            )
            files.append(item)
            print(json.dumps(item), flush=True)
    manifest = {
        "version": 1,
        "source_manifest_sha256": hashlib.sha256(
            (args.corpus / "manifest.json").read_bytes()
        ).hexdigest(),
        "seed": args.seed,
        "files": files,
        "split_rule": "Every packed source and length/position variant stays in its original frozen partition; no existing diagnostic samples are included.",
        "limitations": [
            "Repeated synthetic filler is a position/length stress test, not natural long-document accuracy.",
            "Packed examples have dense PII and should be reported separately from sparse needles and negative documents.",
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
