"""Recompute aggregate native-token exposure for the frozen 0.6B TRAIN.

This CPU-only audit opens TRAIN labels to count their distribution. It does not
open SELECT, CAL or benchmark rows. Output contains no row text or identifiers.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from training.model.data import canonical

TRAIN_SHA256 = "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755"
TOKENIZER_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
RENDERER_SEGMENTS_SHA256 = (
    "464efa93fa25f5a7e9dda6b9011d38507930a282d25972848d88ca2bb446bb10"
)
EXPECTED_ROWS = 7455
EXPECTED_TOKENS = 4094489
EXPECTED_GROUPS = 5392
EXPECTED_SOURCES = {
    "css_flute_official_train",
    "decision2_programmatic_original_v1",
    "decision2_targeted_programmatic_v1",
    "google_goemotions_official_train",
    "legacy:cosmos_qa",
    "legacy:snli",
    "legacy:squad2_answerability",
    "legacy:stage3_replay",
    "legacy:stage4-general-composition-v2",
}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_renderer() -> None:
    source_path = Path(__file__).resolve().parents[1] / "model" / "decision_model.py"
    source = source_path.read_text(encoding="utf-8")
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "segments"
    )
    actual = hashlib.sha256(
        ast.get_source_segment(source, function).encode()
    ).hexdigest()
    if actual != RENDERER_SEGMENTS_SHA256:
        raise ValueError("Native segments renderer changed")


def native_parts(row: dict[str, Any]) -> list[str]:
    def payload(value: Any) -> str:
        return value if isinstance(value, str) else canonical(value)

    prefix = (
        f"Context:\n{payload(row['state'])}\n\n"
        f"Task type: {row['task_type']}\nQuestion:\n{payload(row['instructions'])}\nOptions:"
    )
    options = [
        "\n<option>\n"
        + canonical({"key": option["key"], "description": option["description"]})
        + "\n</option>"
        for option in row["options"]
    ]
    suffix = "\n\nSelect the single option best supported by the context and instructions.\nDecision:"
    return [prefix, *options, suffix]


def summarize(rows: list[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    source_rows: Counter[str] = Counter()
    source_tokens: Counter[str] = Counter()
    source_type_rows: Counter[str] = Counter()
    source_type_tokens: Counter[str] = Counter()
    type_rows: Counter[str] = Counter()
    type_tokens: Counter[str] = Counter()
    language_rows: Counter[str] = Counter()
    language_tokens: Counter[str] = Counter()
    type_language_rows: Counter[str] = Counter()
    type_language_tokens: Counter[str] = Counter()
    score_levels: Counter[int] = Counter()
    score_three_families: Counter[str] = Counter()
    score_three_family_tokens: Counter[str] = Counter()
    score_three_tokens = 0
    groups = set()
    grouped_rows: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    total_tokens = 0
    for row in rows:
        kind = row["task_type"]
        if kind not in {"choice", "noul", "score"}:
            raise ValueError("Unexpected task type")
        label = row["label"]
        if type(label) is not int or not 0 <= label < len(row["options"]):
            raise ValueError("Malformed TRAIN label")
        count = sum(
            len(tokenizer.encode(part, add_special_tokens=False))
            for part in native_parts(row)
        )
        if count > 8192:
            raise ValueError("TRAIN input exceeds the native context cap")
        total_tokens += count
        groups.add(row["group_id"])
        source_rows[row["source"]] += 1
        source_tokens[row["source"]] += count
        source_type_rows[f"{row['source']}/{kind}"] += 1
        source_type_tokens[f"{row['source']}/{kind}"] += count
        type_rows[kind] += 1
        type_tokens[kind] += count
        grouped_rows[(row["source"], row["group_id"])].append(
            (kind, len(row["options"]))
        )
        language = row["language"]
        language_rows[language] += 1
        language_tokens[language] += count
        type_language_rows[f"{kind}/{language}"] += 1
        type_language_tokens[f"{kind}/{language}"] += count
        if kind == "score":
            levels = len(row["options"])
            score_levels[levels] += 1
            if levels == 3:
                score_three_families[row["family"]] += 1
                score_three_family_tokens[row["family"]] += count
                score_three_tokens += count
    non_three_score_group_sizes: Counter[str] = Counter()
    for (source, _), entries in grouped_rows.items():
        if all(kind == "score" and levels != 3 for kind, levels in entries):
            non_three_score_group_sizes[f"{source}/{len(entries)}"] += 1
    return {
        "schema": "decision2-small06-train-balance-audit/1",
        "rows": len(rows),
        "groups": len(groups),
        "native_tokens": total_tokens,
        "type_rows": dict(sorted(type_rows.items())),
        "type_tokens": dict(sorted(type_tokens.items())),
        "source_rows": dict(sorted(source_rows.items())),
        "source_tokens": dict(sorted(source_tokens.items())),
        "source_type_rows": dict(sorted(source_type_rows.items())),
        "source_type_tokens": dict(sorted(source_type_tokens.items())),
        "language_rows": dict(sorted(language_rows.items())),
        "language_tokens": dict(sorted(language_tokens.items())),
        "type_language_rows": dict(sorted(type_language_rows.items())),
        "type_language_tokens": dict(sorted(type_language_tokens.items())),
        "score_level_rows": dict(sorted(score_levels.items())),
        "score_three_family_rows": dict(sorted(score_three_families.items())),
        "score_three_family_tokens": dict(sorted(score_three_family_tokens.items())),
        "score_three_tokens": score_three_tokens,
        "non_three_score_group_sizes": dict(
            sorted(non_three_score_group_sizes.items())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.train.is_symlink() or sha_file(args.train) != TRAIN_SHA256:
        raise ValueError("Frozen TRAIN bytes changed")
    tokenizer_json = args.tokenizer_dir / "tokenizer.json"
    if tokenizer_json.is_symlink() or sha_file(tokenizer_json) != TOKENIZER_SHA256:
        raise ValueError("Official tokenizer bytes changed")
    verify_renderer()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_dir, local_files_only=True, use_fast=True
    )
    with args.train.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    report = summarize(rows, tokenizer)
    if (
        report["rows"] != EXPECTED_ROWS
        or report["groups"] != EXPECTED_GROUPS
        or report["native_tokens"] != EXPECTED_TOKENS
        or set(report["source_rows"]) != EXPECTED_SOURCES
        or set(report["language_rows"]) != {"en", "zh"}
    ):
        raise ValueError("Frozen TRAIN roster or native token exposure changed")
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
