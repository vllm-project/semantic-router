"""Frozen context curriculum and held-out stress variants from task source groups."""

# Chinese punctuation is intentional in language-specific task markers.
# ruff: noqa: PLC0415, RUF001
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .context import insert_payload
from .corpus import write_corpus
from .data import read_records

LANGUAGE_ALIASES = {
    "eng": "en",
    "zho": "zh",
    "spa": "es",
    "fra": "fr",
    "deu": "de",
    "jpn": "ja",
}


def choose_payloads(rows, languages, per_label):
    pools = defaultdict(list)
    for row in rows:
        language = LANGUAGE_ALIASES.get(row.get("language"), row.get("language"))
        if language in languages:
            pools[(language, row["label"])].append(row)
    result = []
    for (language, _label), pool in sorted(pools.items()):
        seen = set()
        for row in sorted(
            pool, key=lambda item: hashlib.sha256(item["id"].encode()).hexdigest()
        ):
            if row["group_id"] in seen:
                continue
            seen.add(row["group_id"])
            result.append({**row, "language": language})
            if len(seen) >= per_label:
                break
    return result


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--backgrounds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-label", type=int, default=3)
    parser.add_argument("--eval-per-label", type=int, default=1)
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=[4096, 8192, 16384, 32768]
    )
    parser.add_argument("--languages", nargs="+", default=["en", "zh"])
    parser.add_argument("--payload-sources", nargs="+")
    args = parser.parse_args()
    if min(args.train_per_label, args.eval_per_label, *args.budgets) <= 0:
        raise ValueError("Context budgets and sample counts must be positive")
    backgrounds = json.loads(args.backgrounds.read_text())
    if len({text for group in backgrounds.values() for text in group.values()}) != sum(
        len(group) for group in backgrounds.values()
    ):
        raise ValueError("Background text must differ across languages and partitions")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    contract = json.loads((args.corpus / "contract.json").read_text())

    def count(text):
        return len(tokenizer(text, truncation=False, padding=False)["input_ids"])

    splits = {}
    for split in ["train", "dev", "test"]:
        rows = read_records([args.corpus / f"{split}.jsonl"], contract["label2id"])
        if args.payload_sources:
            rows = [row for row in rows if row.get("source") in args.payload_sources]
        chosen = choose_payloads(
            rows,
            args.languages,
            args.train_per_label if split == "train" else args.eval_per_label,
        )
        if not chosen:
            raise ValueError(f"No eligible {split} source groups")
        output = []
        for row in chosen:
            language = row["language"]
            prefix = "Requested task:\n" if language == "en" else "需要完成的任务：\n"
            payload = prefix + row["text"]
            for budget in args.budgets:
                for position in ["head", "middle", "tail"]:
                    built = insert_payload(
                        payload, backgrounds[split][language], position, budget, count
                    )
                    output.append(
                        {
                            **row,
                            "id": f"context:{row['id']}:{budget}:{position}",
                            "text": built["text"],
                            "source": "authored-context-on-frozen-source",
                            "parent_source": row["source"],
                            "parent_id": row["id"],
                            "length_bucket": str(budget),
                            "position": position,
                            "actual_tokens": built["actual_tokens"],
                            "payload_start": built["payload_start"],
                            "payload_end": built["payload_end"],
                        }
                    )
        splits[split] = output
        print(
            json.dumps(
                {"split": split, "payloads": len(chosen), "variants": len(output)}
            ),
            flush=True,
        )
    provenance = {
        "parent_manifest_sha256": hashlib.sha256(
            (args.corpus / "manifest.json").read_bytes()
        ).hexdigest(),
        "backgrounds_sha256": hashlib.sha256(args.backgrounds.read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(
            (Path(args.tokenizer) / "tokenizer.json").read_bytes()
        ).hexdigest(),
        "train_payloads_per_label_language": args.train_per_label,
        "eval_payloads_per_label_language": args.eval_per_label,
        "budgets": args.budgets,
        "languages": args.languages,
        "scope": "Authored repeated multi-paragraph context around frozen source requests, not natural long-document accuracy. Label is the explicitly marked requested task. Paired length/position variants retain the parent source group and partition.",
        "test_used": False,
    }
    print(json.dumps(write_corpus(args.output, splits, contract, provenance), indent=2))


if __name__ == "__main__":
    main()
