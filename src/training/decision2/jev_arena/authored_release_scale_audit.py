"""Aggregate, gold-free preflight for a private authored scale candidate.

This cannot confer release eligibility. It verifies commitments and exposes
coverage failures before any blinded reviewer packet is made.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from jev_arena.authored_release_scale_v1 import FORM_FAMILIES, file_sha, write_private

VERSION = "jevarena-authored-release-scale-preflight/1"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _prompt(row: dict[str, Any]) -> str:
    return (
        row["state"]
        + "\n"
        + json.dumps(row["questions"], ensure_ascii=False, sort_keys=True)
    )


def _band(tokens: int) -> str:
    if tokens <= 600:
        return "short"
    if tokens <= 2000:
        return "medium"
    return "long"


def audit(
    prepared: Path, overlap_report: Path, tokenizer_json: Path, output: Path
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Immutable scale preflight already exists")
    receipt_path = prepared / "candidate.private.json"
    receipt = json.loads(receipt_path.read_text())
    paths = {
        "casebook": prepared / "casebook.private.json",
        "originals": prepared / "originals.private.jsonl",
        "variants": prepared / "variants.private.jsonl",
        "proofs": prepared / "proofs.private.jsonl",
    }
    for name, path in paths.items():
        if file_sha(path) != receipt[f"{name}_sha256"]:
            raise ValueError(f"{name} changed after preparation")
    cases = json.loads(paths["casebook"].read_text())["cases"]
    originals = _rows(paths["originals"])
    variants = _rows(paths["variants"])
    proofs = _rows(paths["proofs"])
    if not len(cases) == len(originals) == len(variants) == len(proofs):
        raise ValueError("Original, variant or oracle cardinality mismatch")
    overlap = json.loads(overlap_report.read_text())
    if (overlap.get("originals"), overlap.get("paired_variants")) != (
        len(originals),
        len(variants),
    ):
        raise ValueError("Overlap receipt refers to another candidate")
    if overlap.get("tokenizer_json_sha256") != file_sha(tokenizer_json):
        raise ValueError("Tokenizer differs from prompt-only overlap receipt")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    original_lengths = [len(tokenizer.encode(_prompt(row)).ids) for row in originals]
    variant_lengths = [len(tokenizer.encode(_prompt(row)).ids) for row in variants]
    by_type_length: dict[str, Counter[str]] = {
        kind: Counter() for kind in ("choice", "noul", "score")
    }
    for row, length in zip(originals, original_lengths):
        by_type_length[row["questions"]["decision"]["type"]][_band(length)] += 1
    family_counts = Counter(
        source.get("form_family") for case in cases for source in case["sources"]
    )
    raw_label_count = len(
        {source.get("form") for case in cases for source in case["sources"]}
    )
    single_paragraph_sources = sum(
        "\n" not in source["document"] for case in cases for source in case["sources"]
    )
    reasons: list[str] = []
    if (
        overlap.get("exact_hits")
        or overlap.get("near_hits_trigram_jaccard_at_least_0_7")
        or overlap.get("maximum_roster_shared_eight_word_spans")
    ):
        reasons.append("prompt_overlap")
    if max([*original_lengths, *variant_lengths]) > 7500:
        reasons.append("native_token_budget")
    if not all(
        any(counts[band] for counts in by_type_length.values())
        for band in ("short", "medium", "long")
    ):
        reasons.append("missing_length_band")
    if not set(family_counts) <= FORM_FAMILIES or len(family_counts) < 9:
        reasons.append("document_families_unverified")
    if single_paragraph_sources > len(cases):
        reasons.append("one_paragraph_source_dominance")
    # Blind native answers, ambiguity, realism and rights must be verified by
    # independent humans after a passing mechanical preflight.
    if reasons:
        status = "HOLD_BEFORE_BLIND_PACKET"
    else:
        status = "MECHANICAL_PASS_EDITORIAL_PENDING"
    report = {
        "version": VERSION,
        "status": status,
        "hold_reasons": reasons,
        "candidate_receipt_sha256": file_sha(receipt_path),
        "casebook_sha256": file_sha(paths["casebook"]),
        "originals_sha256": file_sha(paths["originals"]),
        "variants_sha256": file_sha(paths["variants"]),
        "overlap_receipt_sha256": file_sha(overlap_report),
        "tokenizer_json_sha256": file_sha(tokenizer_json),
        "original_candidates": len(cases),
        "paired_variants_not_independent": len(variants),
        "by_type_and_length": {
            key: dict(sorted(counts.items())) for key, counts in by_type_length.items()
        },
        "native_tokens_original_min": min(original_lengths),
        "native_tokens_original_max": max(original_lengths),
        "native_tokens_variant_max": max(variant_lengths),
        "semantic_mechanisms": len({case["operation"] for case in cases}),
        "raw_form_labels_not_verified_families": raw_label_count,
        "verified_document_families": len(set(family_counts) & FORM_FAMILIES),
        "single_paragraph_sources": single_paragraph_sources,
        "sources": len(cases) * 2,
        "roster_files": overlap["roster_files"],
        "roster_rows": overlap["roster_rows"],
        "exact_overlap_hits": overlap["exact_hits"],
        "near_overlap_hits": overlap["near_hits_trigram_jaccard_at_least_0_7"],
        "model_inference": False,
        "reviewer_packet_created": False,
        "release_qualified": False,
    }
    write_private(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--overlap-report", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.prepared, args.overlap_report, args.tokenizer_json, args.output)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "hold_reasons",
                    "original_candidates",
                    "paired_variants_not_independent",
                    "by_type_and_length",
                    "semantic_mechanisms",
                    "single_paragraph_sources",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
