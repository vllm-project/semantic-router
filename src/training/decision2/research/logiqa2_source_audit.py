"""Aggregate-only CPU screen of the official LogiQA 2.0 MRC source.

The output contains counts and hashes, never source text, row IDs, or gold.
This does not perform protected-prompt overlap or authorize training.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path

from decision2.training.model.decision_model import segments
from transformers import AutoTokenizer

SPLITS = ("train", "dev", "test")
LANGUAGES = {"en": ("", "id"), "zh": ("_zh", "example_id")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def percentiles(values: list[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": (ordered[(len(ordered) - 1) // 2] + ordered[len(ordered) // 2]) / 2,
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
        "p99": ordered[int(0.99 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prompt_hash(row: dict) -> str:
    return fingerprint(
        {
            "state": norm(row["text"]),
            "question": norm(row["question"]),
            "options": [norm(option) for option in row["options"]],
        }
    )


def passage_hash(row: dict) -> str:
    return fingerprint(norm(row["text"]))


def encoded_length(tokenizer: object, row: dict) -> int:
    native = {
        "state": row["text"],
        "task_type": "choice",
        "instructions": row["question"],
        "options": [
            {"key": str(index), "description": option}
            for index, option in enumerate(row["options"])
        ],
    }
    prefix, options, suffix = segments(native)
    return sum(
        len(tokenizer.encode(part, add_special_tokens=False))
        for part in (prefix, *options, suffix)
    )


def audit(source: Path, tokenizer_path: Path, max_length: int) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    rows: dict[tuple[str, str], list[dict]] = {}
    output: dict = {
        "source_revision_expected": "955e1d3df6c59d9bfb44d9913da1e1a27ec14e18",
        "tokenizer_revision_expected": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        "prompt_version": "decision2-segmented-options-global-query-v1",
        "max_length": max_length,
        "files": {},
        "source_internal_overlap": {},
        "en_zh_same_id": {},
        "protected_overlap": "HOLD: a complete pinned gold-free role inventory was not supplied",
    }
    for split in SPLITS:
        for language, (suffix, key) in LANGUAGES.items():
            name = f"{split}{suffix}.txt"
            path = source / "logiqa" / "DATA" / "LOGIQA" / name
            current = read_rows(path)
            rows[(split, language)] = current
            ids = collections.Counter(row.get(key) for row in current)
            answers = collections.Counter(row.get("answer") for row in current)
            errors = collections.Counter()
            lengths = []
            prompt_groups: dict[str, list[dict]] = collections.defaultdict(list)
            types = collections.Counter()
            for row in current:
                options = row.get("options")
                if not isinstance(row.get(key), int):
                    errors["missing_or_noninteger_id"] += 1
                if not all(
                    isinstance(row.get(field), str) and row[field].strip()
                    for field in ("text", "question")
                ):
                    errors["empty_state_or_question"] += 1
                if (
                    not isinstance(options, list)
                    or len(options) != 4
                    or not all(isinstance(x, str) and x.strip() for x in options)
                ):
                    errors["invalid_four_options"] += 1
                    continue
                if len({norm(x) for x in options}) < 4:
                    errors["duplicate_normalized_option"] += 1
                if type(row.get("answer")) is not int or row["answer"] not in range(4):
                    errors["invalid_answer_index"] += 1
                prompt_groups[prompt_hash(row)].append(row)
                lengths.append(encoded_length(tokenizer, row))
                for kind, present in row.get("type", {}).items():
                    if present:
                        types[kind] += 1
            output["files"][name] = {
                "sha256": sha256(path),
                "rows": len(current),
                "unique_ids": len(ids),
                "duplicate_id_groups": sum(n > 1 for n in ids.values()),
                "duplicate_id_rows": sum(n for n in ids.values() if n > 1),
                "answer_histogram": {str(k): v for k, v in sorted(answers.items())},
                "validation_errors": dict(errors),
                "exact_prompt_groups": len(prompt_groups),
                "repeated_prompt_groups": sum(
                    len(v) > 1 for v in prompt_groups.values()
                ),
                "repeated_prompt_conflicting_answers": sum(
                    len({row["answer"] for row in group}) > 1
                    for group in prompt_groups.values()
                ),
                "native_tokens": percentiles(lengths),
                "native_token_rows": len(lengths),
                "native_tokens_total": sum(lengths),
                "over_max_length": sum(n > max_length for n in lengths),
                "reasoning_type_positive_rows": dict(sorted(types.items())),
                "missing_reasoning_type_rows": sum(
                    not isinstance(row.get("type"), dict) for row in current
                ),
            }
        en_by_id: dict[int, list[dict]] = collections.defaultdict(list)
        zh_by_id: dict[int, list[dict]] = collections.defaultdict(list)
        for row in rows[(split, "en")]:
            en_by_id[row["id"]].append(row)
        for row in rows[(split, "zh")]:
            zh_by_id[row["example_id"]].append(row)
        common = set(en_by_id) & set(zh_by_id)
        unique_common = [
            key for key in common if len(en_by_id[key]) == len(zh_by_id[key]) == 1
        ]
        output["en_zh_same_id"][split] = {
            "intersecting_ids": len(common),
            "one_to_one_ids": len(unique_common),
            "one_to_one_answer_agreement": sum(
                en_by_id[key][0]["answer"] == zh_by_id[key][0]["answer"]
                for key in unique_common
            ),
        }
    for language in LANGUAGES:
        for first, second in (("train", "dev"), ("train", "test"), ("dev", "test")):
            left, right = rows[(first, language)], rows[(second, language)]
            left_prompt: dict[str, set[int]] = collections.defaultdict(set)
            right_prompt: dict[str, set[int]] = collections.defaultdict(set)
            for row in left:
                left_prompt[prompt_hash(row)].add(row["answer"])
            for row in right:
                right_prompt[prompt_hash(row)].add(row["answer"])
            matched = set(left_prompt) & set(right_prompt)
            output["source_internal_overlap"][f"{language}:{first}:{second}"] = {
                "exact_full_prompt_groups": len(matched),
                "same_prompt_conflicting_answer": sum(
                    len(left_prompt[key] | right_prompt[key]) > 1 for key in matched
                ),
                "exact_passage_groups": len(
                    {passage_hash(row) for row in left}
                    & {passage_hash(row) for row in right}
                ),
                "same_id_groups": len(
                    {row[LANGUAGES[language][1]] for row in left}
                    & {row[LANGUAGES[language][1]] for row in right}
                ),
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()
    if args.max_length <= 0:
        parser.error("max-length must be positive")
    print(
        json.dumps(
            audit(args.source_root, args.tokenizer, args.max_length), sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
