"""Gold-free overlap and native-token preflight for authored scale v2.

Only candidate prompt fields and prompt-only reference roster fields are read.
The separate answer snapshot is rehashed, never opened. Editorial adequacy
and semantic independence still require blinded human review.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from jev_arena.authored_release_scale_v1 import file_sha, write_private
from jev_arena.authored_v13_audit import (
    native_prompt_texts,
    read_rows,
    shingles,
    word_tokens,
)

VERSION = "jevarena-authored-release-scale-v2-preflight/1"


def _band(tokens: int) -> str:
    if tokens <= 600:
        return "short"
    if tokens <= 2000:
        return "medium"
    return "long"


def _required_gaps(by_type: dict[str, Counter[str]]) -> list[str]:
    reasons = []
    by_band = Counter()
    for counts in by_type.values():
        by_band.update(counts)
    if by_band["short"] < 6 or by_band["medium"] < 2 or by_band["long"] < 1:
        reasons.append("length_allocation")
    return reasons


def audit(
    prepared: Path, roster_list: Path, tokenizer_json: Path, output: Path
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("V2 preflight is immutable once written")
    receipt_path = prepared / "receipt.private.json"
    receipt = json.loads(receipt_path.read_text())
    for name, expected in receipt["components_sha256"].items():
        if file_sha(prepared / f"{name}.private.jsonl") != expected:
            raise ValueError(f"V2 {name} snapshot changed")
    originals_path = prepared / "originals.private.jsonl"
    variants_path = prepared / "variants.private.jsonl"
    originals = native_prompt_texts(originals_path)
    variants = native_prompt_texts(variants_path)
    if (
        not len(originals)
        == len(variants)
        == receipt["independent_original_candidates"]
    ):
        raise ValueError("Original and paired prompt snapshots differ in cardinality")
    candidate = originals + variants
    three = [shingles(text, 3) for text in candidate]
    eight = [shingles(text, 8) for text in candidate]
    exact_index = Counter(tuple(word_tokens(text)) for text in candidate)
    three_index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    eight_index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index in range(len(candidate)):
        for gram in three[index]:
            three_index[gram].append(index)
        for gram in eight[index]:
            eight_index[gram].append(index)
    paths = [
        Path(line.strip())
        for line in roster_list.read_text().splitlines()
        if line.strip()
    ]
    if (
        not paths
        or len(paths) != len(set(paths))
        or any(not path.is_file() for path in paths)
    ):
        raise ValueError("Reference roster list is incomplete or repeated")
    exact_hits = near_hits = shared_eight = 0
    maximum_roster_three = 0.0
    roster_rows = 0
    for path in paths:
        for reference in read_rows(path):
            roster_rows += 1
            tokens = word_tokens(reference)
            exact_hits += exact_index[tuple(tokens)]
            ref_three = shingles(reference, 3)
            ref_eight = shingles(reference, 8)
            counts: Counter[int] = Counter()
            for gram in ref_three:
                counts.update(three_index.get(gram, ()))
            for index, intersection in counts.items():
                union = len(ref_three) + len(three[index]) - intersection
                similarity = intersection / union if union else 0.0
                near_hits += similarity >= 0.7
                maximum_roster_three = max(maximum_roster_three, similarity)
            common_eight: set[int] = set()
            for gram in ref_eight:
                common_eight.update(eight_index.get(gram, ()))
            shared_eight += len(common_eight)
    own_max_three = 0.0
    for index, left in enumerate(three[: len(originals)]):
        for right in three[index + 1 : len(originals)]:
            union = len(left | right)
            own_max_three = max(
                own_max_three, len(left & right) / union if union else 0.0
            )
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    original_lengths = [len(tokenizer.encode(text).ids) for text in originals]
    variant_lengths = [len(tokenizer.encode(text).ids) for text in variants]
    prompt_rows = [
        json.loads(line)
        for line in originals_path.read_text().splitlines()
        if line.strip()
    ]
    by_type: dict[str, Counter[str]] = {
        kind: Counter() for kind in ("choice", "noul", "score")
    }
    for row, length in zip(prompt_rows, original_lengths):
        by_type[row["questions"]["decision"]["type"]][_band(length)] += 1
    source_rows = [
        json.loads(line)
        for line in (prepared / "sources.private.jsonl").read_text().splitlines()
        if line.strip()
    ]
    one_paragraph = sum(
        "\n\n" not in source["document"]
        for case in source_rows
        for source in case["sources"]
    )
    reasons = _required_gaps(by_type)
    if exact_hits or near_hits or shared_eight:
        reasons.append("prompt_overlap")
    if max([*original_lengths, *variant_lengths]) > 3500:
        reasons.append("native_token_envelope")
    if one_paragraph:
        reasons.append("one_paragraph_source")
    report = {
        "version": VERSION,
        "status": (
            "HOLD_BEFORE_BLIND_PACKET"
            if reasons
            else "MECHANICAL_PASS_EDITORIAL_PENDING"
        ),
        "hold_reasons": reasons,
        "source_commit": receipt["source_commit"],
        "candidate_receipt_sha256": file_sha(receipt_path),
        "source_snapshot_sha256": receipt["components_sha256"]["sources"],
        "prompt_snapshot_sha256": receipt["components_sha256"]["originals"],
        "answer_snapshot_sha256": receipt["components_sha256"]["answers"],
        "proof_snapshot_sha256": receipt["components_sha256"]["proofs"],
        "roster_list_sha256": file_sha(roster_list),
        "roster_files": len(paths),
        "roster_rows": roster_rows,
        "exact_hits": exact_hits,
        "near_hits_trigram_jaccard_at_least_0_7": near_hits,
        "shared_eight_word_spans": shared_eight,
        "maximum_roster_trigram_jaccard": round(maximum_roster_three, 6),
        "maximum_cross_original_trigram_jaccard": round(own_max_three, 6),
        "tokenizer_json_sha256": file_sha(tokenizer_json),
        "by_type_and_length": {
            kind: dict(sorted(counts.items())) for kind, counts in by_type.items()
        },
        "native_tokens_original_min": min(original_lengths),
        "native_tokens_original_max": max(original_lengths),
        "native_tokens_variant_max": max(variant_lengths),
        "one_paragraph_sources": one_paragraph,
        "document_families": receipt["document_families"],
        "target_distribution": receipt["target_distribution"],
        "model_inference": False,
        "reviewer_packet_created": False,
        "release_qualified": False,
    }
    write_private(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--roster-list", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.prepared, args.roster_list, args.tokenizer_json, args.output)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "hold_reasons",
                    "by_type_and_length",
                    "native_tokens_original_max",
                    "exact_hits",
                    "near_hits_trigram_jaccard_at_least_0_7",
                    "shared_eight_word_spans",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
