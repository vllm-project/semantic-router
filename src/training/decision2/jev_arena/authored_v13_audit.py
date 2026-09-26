"""Private-only lexical, roster and tokenizer preflight for authored v13.

Only prompt/state fields are extracted from roster JSONL. Gold, labels, model
outputs and private joins are never used or emitted. A clean text screen does
not establish semantic independence or editorial quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

WORD = re.compile(r"[\w]+", re.UNICODE)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def word_tokens(value: str) -> list[str]:
    return [token.lower() for token in WORD.findall(value)]


def prompt_text(row: dict[str, Any]) -> str | None:
    """Access prompt-side fields only, irrespective of other row keys."""
    state = row.get("state")
    if not isinstance(state, str):
        state = row.get("prompt")
    if not isinstance(state, str) or not state.strip():
        return None
    questions = row.get("questions")
    if isinstance(questions, dict):
        instructions = [
            question.get("instructions", "")
            for question in questions.values()
            if isinstance(question, dict)
        ]
        return (
            state
            + "\n"
            + "\n".join(text for text in instructions if isinstance(text, str))
        )
    instructions = row.get("instructions")
    return state + ("\n" + instructions if isinstance(instructions, str) else "")


def shingles(value: str, width: int) -> set[tuple[str, ...]]:
    parts = word_tokens(value)
    return {
        tuple(parts[index : index + width])
        for index in range(max(0, len(parts) - width + 1))
    }


def compare(candidate: str, reference: str) -> dict[str, float | int | bool]:
    left, right = word_tokens(candidate), word_tokens(reference)
    a, b = set(left), set(right)
    three_a, three_b = shingles(candidate, 3), shingles(reference, 3)
    return {
        "exact": left == right,
        "word_jaccard": len(a & b) / len(a | b) if a | b else 0.0,
        "trigram_jaccard": (
            len(three_a & three_b) / len(three_a | three_b)
            if three_a | three_b
            else 0.0
        ),
        "shared_eight_word_spans": len(shingles(candidate, 8) & shingles(reference, 8)),
    }


def read_rows(path: Path) -> list[str]:
    rows: list[str] = []
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            prompt = prompt_text(json.loads(line))
            if prompt is not None:
                rows.append(prompt)
    return rows


def native_prompt_texts(path: Path) -> list[str]:
    """Count the complete native prompt, including criteria and option text."""
    rows: list[str] = []
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            rows.append(
                row["state"]
                + "\n"
                + json.dumps(row["questions"], ensure_ascii=False, sort_keys=True)
            )
    return rows


def audit(
    freeze: Path, roster_list: Path, tokenizer_json: Path, output: Path
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Audit is immutable once written")
    originals = read_rows(freeze / "reviewer-a" / "packet.jsonl")
    variants = read_rows(freeze / "reviewer-b" / "packet.jsonl")
    if not originals or len(originals) != len(variants):
        raise ValueError("Frozen pair count mismatch")
    candidate = originals + variants
    candidate_parts = [word_tokens(text) for text in candidate]
    candidate_three = [shingles(text, 3) for text in candidate]
    candidate_eight = [shingles(text, 8) for text in candidate]
    exact_index: Counter[tuple[str, ...]] = Counter(
        tuple(parts) for parts in candidate_parts
    )
    trigram_index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    eight_index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, (three, eight) in enumerate(zip(candidate_three, candidate_eight)):
        for part in three:
            trigram_index[part].append(index)
        for part in eight:
            eight_index[part].append(index)
    roster_paths = [
        Path(line.strip())
        for line in roster_list.read_text().splitlines()
        if line.strip()
    ]
    if not roster_paths or len(roster_paths) != len(set(roster_paths)):
        raise ValueError("Roster list empty or repeated")
    if any(path.suffix != ".jsonl" for path in roster_paths):
        raise ValueError("Only JSONL prompt rosters accepted")
    exact_hits = 0
    near_hits = 0
    max_trigram = 0.0
    max_eight = 0
    roster_counts: dict[str, int] = {}
    roster_hashes: dict[str, str] = {}
    for path in roster_paths:
        references = read_rows(path)
        roster_counts[str(path)] = len(references)
        roster_hashes[str(path)] = sha(path)
        for reference in references:
            reference_parts = word_tokens(reference)
            reference_three = shingles(reference, 3)
            reference_eight = shingles(reference, 8)
            exact_hits += exact_index[tuple(reference_parts)]
            shared_three: Counter[int] = Counter()
            shared_eight: Counter[int] = Counter()
            for part in reference_three:
                shared_three.update(trigram_index.get(part, ()))
            for part in reference_eight:
                shared_eight.update(eight_index.get(part, ()))
            for index, shared in shared_three.items():
                jaccard = shared / (
                    len(candidate_three[index]) + len(reference_three) - shared
                )
                near_hits += jaccard >= 0.7
                max_trigram = max(max_trigram, jaccard)
            if shared_eight:
                max_eight = max([max_eight, *shared_eight.values()])
    own_max_trigram = 0.0
    for left_index, left in enumerate(originals):
        for right in originals[left_index + 1 :]:
            own_max_trigram = max(
                own_max_trigram, compare(left, right)["trigram_jaccard"]
            )
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    full_prompts = native_prompt_texts(
        freeze / "reviewer-a" / "packet.jsonl"
    ) + native_prompt_texts(freeze / "reviewer-b" / "packet.jsonl")
    lengths = [len(tokenizer.encode(text).ids) for text in full_prompts]
    if max(lengths) > 960:
        raise ValueError("Native prompt exceeds the 1024-token envelope with reserve")
    report = {
        "version": "jevarena-authored-v13-private-preflight/1",
        "originals": len(originals),
        "paired_variants": len(variants),
        "roster_files": len(roster_paths),
        "roster_rows": sum(roster_counts.values()),
        "roster_counts": roster_counts,
        "roster_sha256": roster_hashes,
        "exact_hits": exact_hits,
        "near_hits_trigram_jaccard_at_least_0_7": near_hits,
        "maximum_roster_trigram_jaccard": round(max_trigram, 6),
        "maximum_roster_shared_eight_word_spans": max_eight,
        "maximum_cross_original_trigram_jaccard": round(own_max_trigram, 6),
        "tokenizer_json_sha256": sha(tokenizer_json),
        "token_count_min": min(lengths),
        "token_count_max": max(lengths),
        "token_count_mean": round(sum(lengths) / len(lengths), 2),
        "token_budget": 1024,
        "reserved_tokens": 64,
        "all_rosters_prompt_fields_only": True,
        "no_model_inference": True,
        "limitation": "Text overlap does not prove semantic independence or source necessity; independent blind editorial review remains required.",
    }
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    output.chmod(0o600)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--roster-list", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.freeze, args.roster_list, args.tokenizer_json, args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "originals",
                    "paired_variants",
                    "roster_files",
                    "roster_rows",
                    "exact_hits",
                    "near_hits_trigram_jaccard_at_least_0_7",
                    "maximum_roster_trigram_jaccard",
                    "token_count_max",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
