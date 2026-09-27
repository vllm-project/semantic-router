"""Gold-free CPU feasibility audit for the own-Eos Score-only substitution.

The command emits aggregate counts and hashes only. It never reads benchmark
answers or exports source text, record identifiers, or private paths.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def token_count(row: dict[str, Any], tokenizer: Any) -> int:
    """Mirror training.model.decision_model.segments/encode token boundaries."""
    state = row["state"] if isinstance(row["state"], str) else canonical(row["state"])
    instructions = (
        row["instructions"]
        if isinstance(row["instructions"], str)
        else canonical(row["instructions"])
    )
    prefix = (
        f"Context:\n{state}\n\nTask type: {row['task_type']}\n"
        f"Question:\n{instructions}\nOptions:"
    )
    options = [
        "\n<option>\n"
        + canonical({"key": option["key"], "description": option["description"]})
        + "\n</option>"
        for option in row["options"]
    ]
    suffix = (
        "\n\nSelect the single option best supported by the context and "
        "instructions.\nDecision:"
    )
    return (
        len(tokenizer.encode(prefix, add_special_tokens=False))
        + sum(len(tokenizer.encode(part, add_special_tokens=False)) for part in options)
        + len(tokenizer.encode(suffix, add_special_tokens=False))
    )


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def control(replay: Path, receipt: Path, tokenizer: Any) -> dict[str, Any]:
    rows = jsonl(replay)
    saved = json.loads(receipt.read_text(encoding="utf-8"))
    if sha256(replay) != saved["replay_sha256"]:
        raise ValueError("Replay bytes differ from archived receipt")
    roster = saved["pool_roster"]
    if len(rows) != 512 or len(roster) != len(rows):
        raise ValueError("Archived replay/roster size changed")
    if any(
        row["id"] != entry["id"] or row["input_sha256"] != entry["input_sha256"]
        for row, entry in zip(rows, roster, strict=True)
    ):
        raise ValueError("Archived replay row identity/order differs from receipt")
    lengths = [token_count(row, tokenizer) for row in rows]
    if sum(lengths) != saved["replay_token_count"]:
        raise ValueError("Native tokenizer/encoder differs from archived receipt")
    counts = collections.Counter(row["task_type"] for row in rows)
    sums = {
        kind: sum(
            n for row, n in zip(rows, lengths, strict=True) if row["task_type"] == kind
        )
        for kind in ("choice", "noul", "score")
    }
    ordered_identities = [
        (
            row["id"],
            row["source"],
            row["group_id"],
            row["input_sha256"],
            row["task_type"],
        )
        for row in rows
    ]
    identity_sha = hashlib.sha256(
        json.dumps(ordered_identities, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "status": "PASS_CONTROL_IDENTITY_AND_TOKENS",
        "replay_sha256": sha256(replay),
        "ordered_identity_sha256": identity_sha,
        "counts_by_type": dict(counts),
        "tokens_by_type": sums,
        "replay_tokens": sum(lengths),
        "unchanged_base_train_tokens": saved["train_token_count"],
        "total_tokens": saved["train_token_count"] + sum(lengths),
        "score_distinct_groups": len(
            {row["group_id"] for row in rows if row["task_type"] == "score"}
        ),
    }


EVIDENCE_OPTIONS = [
    {"key": "0", "description": "The premise refutes the claim."},
    {"key": "1", "description": "The premise does not determine the claim."},
    {"key": "2", "description": "The premise supports the claim."},
]
EVIDENCE_INSTRUCTIONS = (
    "Using only the premise, determine how the evidence bears on the claim."
)


def candidate(
    ocnli: Path,
    expected_sha: str,
    tokenizer: Any,
    k: int,
    old_score_tokens: int,
    old_total_tokens: int,
    tolerance: float,
) -> dict[str, Any]:
    if sha256(ocnli) != expected_sha:
        raise ValueError("OCNLI original TRAIN bytes differ from audited file")
    lengths = []
    labels: collections.Counter[str] = collections.Counter()
    excluded = collections.Counter()
    with ocnli.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("label") == "-":
                excluded["no_consensus"] += 1
                continue
            if not isinstance(record.get("genre"), str) or not isinstance(
                record.get("prem_id"), str
            ):
                excluded["missing_provenance"] += 1
                continue
            if record["genre"].casefold() == "news":
                excluded["news_rights"] += 1
                continue
            if record["label"] not in {"contradiction", "neutral", "entailment"}:
                raise ValueError("Unexpected OCNLI label")
            if not record.get("sentence1") or not record.get("sentence2"):
                raise ValueError("Empty OCNLI text")
            row = {
                "state": {"premise": record["sentence1"], "claim": record["sentence2"]},
                "task_type": "score",
                "instructions": EVIDENCE_INSTRUCTIONS,
                "options": EVIDENCE_OPTIONS,
            }
            lengths.append(token_count(row, tokenizer))
            labels[record["label"]] += 1
    if len(lengths) < k:
        raise ValueError("Fewer OCNLI non-news rows than repeat slots")
    top_k = sum(sorted(lengths, reverse=True)[:k])
    tolerance_tokens = math.floor(tolerance * old_total_tokens)
    minimum_replacement_tokens = old_score_tokens - tolerance_tokens
    # This is deliberately generous: it ignores whole-group uniqueness,
    # class balance, rights, ambiguity and protected-overlap exclusions.
    return {
        "status": (
            "HOLD_TOKEN_UPPER_BOUND"
            if top_k < minimum_replacement_tokens
            else "TOKEN_UPPER_BOUND_ONLY_NOT_ADMITTED"
        ),
        "ocnli_train_sha256": sha256(ocnli),
        "candidate_rows_before_group_and_overlap_filters": len(lengths),
        "candidate_class_counts": dict(labels),
        "excluded_counts": dict(excluded),
        "candidate_native_tokens_max": max(lengths),
        "candidate_native_tokens_top_k_upper_bound": top_k,
        "k": k,
        "old_score_tokens": old_score_tokens,
        "old_total_tokens": old_total_tokens,
        "one_percent_tolerance_tokens": tolerance_tokens,
        "minimum_replacement_tokens": minimum_replacement_tokens,
        "shortfall_even_under_upper_bound": max(0, minimum_replacement_tokens - top_k),
        "upper_bound_ignores_group_uniqueness_and_class_balance": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--replay", type=Path)
    modes.add_argument("--ocnli", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--ocnli-sha256")
    parser.add_argument("--k", type=int)
    parser.add_argument("--old-score-tokens", type=int)
    parser.add_argument("--old-total-tokens", type=int)
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    if args.replay:
        if not args.receipt:
            parser.error("--receipt required with --replay")
        result = control(args.replay, args.receipt, tokenizer)
    else:
        if not all(
            value is not None
            for value in (
                args.ocnli_sha256,
                args.k,
                args.old_score_tokens,
                args.old_total_tokens,
            )
        ):
            parser.error("OCNLI mode requires source hash and control token values")
        result = candidate(
            args.ocnli,
            args.ocnli_sha256,
            tokenizer,
            args.k,
            args.old_score_tokens,
            args.old_total_tokens,
            args.tolerance,
        )
    result["audit_script_sha256"] = sha256(Path(__file__))
    print(canonical(result))


if __name__ == "__main__":
    main()
