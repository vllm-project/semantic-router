"""CPU-only ANLI dev feasibility screen for native three-level Score.

The source labels are counted in aggregate but are never placed in prompts.
This tool runs no model, reads no protected answer key, and writes no source
text, source IDs, individual labels, or individual predictions.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from training.data.audit_27b_full_input_overlap import full_input_overlap_rows
from training.data.plan_goldfree_inventory import (
    CORE_ROLES,
    NATIVE_ROLE_COUNTS,
    PARTITION_ROLE_COUNTS,
    validate_core_rows,
)
from training.model.data import file_sha256
from training.model.decision_model import PROMPT_VERSION, encode
from training.model.infer import question_to_row

SOURCE_ID = "facebook/anli"
SOURCE_REVISION = "8e4813d81f46d313dac7892e1c28076917cfcdf9"
SOURCE_LICENSE = "CC BY-NC 4.0"
ROUND_FILES = {
    1: (
        "dev_r1-00000-of-00001.parquet",
        1000,
        "72e27463177b4363be80f1fc6ccdaab44ddaeb65db58c2280f94690e15468334",
    ),
    2: (
        "dev_r2-00000-of-00001.parquet",
        1000,
        "43e4673665decf0b0e8487e55f98285423cb356b985e206fe5998defae2e38fa",
    ),
    3: (
        "dev_r3-00000-of-00001.parquet",
        1200,
        "61775ec09351f6011ce4dc9ea313f457bba6e11d7665d34d95c111665023a83e",
    ),
}
TOKENIZER_FILES = {
    "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    "tokenizer_config.json": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
    "config.json": "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
}
SOURCE_LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}
SCORE_LEVELS = {"contradiction": 0, "neutral": 1, "entailment": 2}
SCORE_INSTRUCTIONS = (
    "Using only the provided evidence, rate how strongly it supports the claim. "
    "Do not fill gaps with outside knowledge."
)
SCORE_CRITERIA = [
    "The evidence contradicts the claim.",
    "The evidence is insufficient to determine the claim.",
    "The evidence supports the claim.",
]
INPUT_COLUMNS = ["uid", "premise", "hypothesis", "label", "reason"]
PROTECTED_SCHEMA = "decision2-projected-core-inputs-v1"


@dataclass(frozen=True)
class Pair:
    round: int
    premise: str
    hypothesis: str
    label: int
    reason_present: bool


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def percentile(values: list[int], numerator: int, denominator: int = 100) -> int:
    if not values:
        raise ValueError("Cannot summarize empty values")
    ordered = sorted(values)
    return ordered[(numerator * (len(ordered) - 1)) // denominator]


def read_dev(directory: Path) -> tuple[list[Pair], dict[str, str]]:
    pairs = []
    seen_uids = set()
    hashes = {}
    for round_id, (filename, expected_rows, expected_sha) in ROUND_FILES.items():
        path = directory / filename
        if not path.is_file() or file_sha256(path) != expected_sha:
            raise ValueError("Pinned ANLI development bytes are missing or changed")
        table = pq.read_table(path)
        if table.column_names != INPUT_COLUMNS or table.num_rows != expected_rows:
            raise ValueError("ANLI development schema or row count changed")
        hashes[f"r{round_id}"] = expected_sha
        for uid, premise, hypothesis, label, reason in zip(
            *(table[name].to_pylist() for name in INPUT_COLUMNS), strict=True
        ):
            if (
                not isinstance(uid, str)
                or not uid
                or uid in seen_uids
                or not isinstance(premise, str)
                or not premise.strip()
                or not isinstance(hypothesis, str)
                or not hypothesis.strip()
                or type(label) is not int
                or label not in SOURCE_LABELS
                or not isinstance(reason, str)
            ):
                raise ValueError("ANLI development row is malformed or repeated")
            seen_uids.add(uid)
            pairs.append(
                Pair(round_id, premise, hypothesis, label, bool(reason.strip()))
            )
    return pairs, hashes


def score_row(pair: Pair, ordinal: int) -> dict[str, Any]:
    """Build the complete native Score input without using label or reason."""
    prompt = {
        "id": f"anli-dev-ordinal-{ordinal}",
        "state": {"evidence": pair.premise, "claim": pair.hypothesis},
    }
    question = {
        "type": "score",
        "instructions": SCORE_INSTRUCTIONS,
        "criteria": SCORE_CRITERIA,
    }
    row = question_to_row(prompt, "support", question)
    row["family"] = "anli-dev-diagnostic"
    return row


def source_profile(pairs: list[Pair]) -> dict[str, Any]:
    if not pairs:
        raise ValueError("ANLI dev is empty")
    across_groups = collections.Counter(normalize(row.premise) for row in pairs)
    group_rounds: dict[str, set[int]] = collections.defaultdict(set)
    pair_labels: dict[tuple[str, str], set[int]] = collections.defaultdict(set)
    by_round = {}
    for round_id in ROUND_FILES:
        subset = [row for row in pairs if row.round == round_id]
        groups = collections.Counter(normalize(row.premise) for row in subset)
        labels = collections.Counter(SOURCE_LABELS[row.label] for row in subset)
        lengths = [len(row.premise) + len(row.hypothesis) for row in subset]
        by_round[f"r{round_id}"] = {
            "rows": len(subset),
            "labels": {name: labels[name] for name in SOURCE_LABELS.values()},
            "independent_normalized_premises": len(groups),
            "premise_group_size": {
                "median": percentile(list(groups.values()), 50),
                "p90": percentile(list(groups.values()), 90),
                "max": max(groups.values()),
            },
            "reason_present": sum(row.reason_present for row in subset),
            "raw_pair_characters": {
                "median": percentile(lengths, 50),
                "p90": percentile(lengths, 90),
                "p99": percentile(lengths, 99),
                "max": max(lengths),
            },
        }
    for row in pairs:
        group_rounds[normalize(row.premise)].add(row.round)
        pair_labels[(normalize(row.premise), normalize(row.hypothesis))].add(row.label)
    return {
        "total_rows": len(pairs),
        "rounds": by_round,
        "independent_normalized_premises_all_rounds": len(across_groups),
        "premise_groups_crossing_rounds": sum(
            len(rounds) > 1 for rounds in group_rounds.values()
        ),
        "repeated_pair_rows": len(pairs) - len(pair_labels),
        "pair_texts_with_conflicting_labels": sum(
            len(labels) > 1 for labels in pair_labels.values()
        ),
    }


def prompt_profile(pairs: list[Pair], tokenizer: Any) -> dict[str, Any]:
    results = {}
    for round_id in ROUND_FILES:
        lengths = [
            len(encode(score_row(row, index), tokenizer, max_length=1_000_000)["ids"])
            for index, row in enumerate(pairs)
            if row.round == round_id
        ]
        results[f"r{round_id}"] = {
            "requests": len(lengths),
            "tokens": {
                "median": percentile(lengths, 50),
                "p90": percentile(lengths, 90),
                "p99": percentile(lengths, 99),
                "max": max(lengths),
            },
            "over_4096": sum(length > 4096 for length in lengths),
            "over_8192": sum(length > 8192 for length in lengths),
        }
    return {
        "native_prompt_version": PROMPT_VERSION,
        "api_type": "score",
        "ordered_levels": 3,
        "source_to_score_level": {
            SOURCE_LABELS[label]: SCORE_LEVELS[SOURCE_LABELS[label]]
            for label in SOURCE_LABELS
        },
        "reason_in_prompt": False,
        "source_label_in_prompt": False,
        "truncation": "none",
        "rounds": results,
    }


def overlap_profile(
    pairs: list[Pair], roles: dict[str, list[dict[str, Any]]] | None
) -> dict[str, Any]:
    if roles is None:
        return {
            "status": "HOLD_MISSING_PROTECTED_INVENTORY",
            "missing_roles": sorted(CORE_ROLES),
        }
    missing = sorted(CORE_ROLES - set(roles))
    if missing:
        return {"status": "HOLD_MISSING_PROTECTED_ROLES", "missing_roles": missing}
    extras = sorted(set(roles) - CORE_ROLES)
    if extras:
        return {"status": "HOLD_UNATTESTED_OPTIONAL_ROLES", "roles": extras}
    try:
        role_counts = validate_core_rows(roles)
    except (KeyError, TypeError, ValueError):
        return {"status": "HOLD_PROTECTED_ROLE_SCHEMA"}
    source_rows = [
        {
            "id": f"anli-dev-ordinal-{index}",
            "state": row.premise,
            "instructions": row.hypothesis,
        }
        for index, row in enumerate(pairs)
    ]
    by_role = {
        name: full_input_overlap_rows(source_rows, reference)
        for name, reference in sorted(roles.items())
    }
    blocked = any(any(value["counts"].values()) for value in by_role.values())
    return {
        "status": (
            "HOLD_OBSERVABLE_OVERLAP" if blocked else "PASS_BOUNDED_LEXICAL_SCREEN"
        ),
        "reference_role_counts": role_counts,
        "by_role": by_role,
        "method": "Only ANLI premise/hypothesis source text compared with all projected input fields; exact and bounded near row-pair counts, no IDs or matches emitted.",
        "limitation": "Lexical screens cannot establish semantic or original-corpus disjointness.",
    }


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Projected inventory has a duplicate JSON key")
        result[key] = value
    return result


def _sha256_string(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def load_projected_roles(
    manifest: Path, *, expected_sha256: str
) -> dict[str, list[dict[str, Any]]]:
    """Load only a complete pinned eight-role input projection.

    An incomplete candidate is returned as role names with empty rows so the
    overlap gate reports precisely which roles are missing before any prompt
    file is opened. A complete manifest must pass strict projection checks.
    """
    if (
        manifest.is_symlink()
        or not manifest.is_file()
        or not _sha256_string(expected_sha256)
        or file_sha256(manifest) != expected_sha256
    ):
        raise ValueError("Pinned protected inventory manifest is missing or changed")
    document = json.loads(
        manifest.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys
    )
    if (
        not isinstance(document, dict)
        or set(document)
        != {"schema", "roles", "excluded_optional_role_count", "source_manifest_sha256"}
        or document["schema"] != PROTECTED_SCHEMA
        or not isinstance(document["roles"], dict)
        or not document["roles"]
        or type(document["excluded_optional_role_count"]) is not int
        or document["excluded_optional_role_count"] < 0
        or not _sha256_string(document["source_manifest_sha256"])
    ):
        raise ValueError("Protected inventory manifest schema changed")
    indexed = document["roles"]
    if set(indexed) != CORE_ROLES:
        return {role: [] for role in indexed}
    roles = {}
    for role, entry in sorted(indexed.items()):
        required = {"path", "rows", "sha256", "source_sha256"}
        if role in NATIVE_ROLE_COUNTS:
            required.add("native_input_digest_list_sha256")
        expected_rows = (
            NATIVE_ROLE_COUNTS[role]
            if role in NATIVE_ROLE_COUNTS
            else PARTITION_ROLE_COUNTS[role][1]
        )
        if (
            not isinstance(entry, dict)
            or set(entry) != required
            or type(entry["rows"]) is not int
            or entry["rows"] != expected_rows
            or not isinstance(entry["path"], str)
            or not _sha256_string(entry["sha256"])
            or not _sha256_string(entry["source_sha256"])
            or (
                role in NATIVE_ROLE_COUNTS
                and not _sha256_string(entry["native_input_digest_list_sha256"])
            )
        ):
            raise ValueError("Protected inventory role schema changed")
        relative = Path(entry["path"])
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("Protected inventory path escapes its manifest")
        path = manifest.parent / relative
        if not path.resolve().is_relative_to(manifest.parent.resolve()):
            raise ValueError("Protected inventory path escapes its manifest")
        if (
            path.is_symlink()
            or not path.is_file()
            or file_sha256(path) != entry["sha256"]
        ):
            raise ValueError("Pinned projected role changed")
        rows = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    raise ValueError("Projected role contains a blank line")
                rows.append(json.loads(line, object_pairs_hook=_unique_keys))
        roles[role] = rows
    validate_core_rows(roles)
    return roles


def audit(
    source_directory: Path,
    tokenizer_directory: Path,
    *,
    protected_roles: dict[str, list[dict[str, Any]]] | None = None,
    protected_inventory_sha256: str | None = None,
) -> dict[str, Any]:
    pairs, source_hashes = read_dev(source_directory)
    for name, expected in TOKENIZER_FILES.items():
        path = tokenizer_directory / name
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError("Pinned native tokenizer changed")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_directory, local_files_only=True, trust_remote_code=False
    )
    profile = source_profile(pairs)
    overlap = overlap_profile(pairs, protected_roles)
    if protected_inventory_sha256 is not None:
        if not _sha256_string(protected_inventory_sha256):
            raise ValueError("Invalid protected inventory SHA-256")
        overlap["protected_inventory_sha256"] = protected_inventory_sha256
    return {
        "schema": "decision2-anli-score-dev-feasibility/1",
        "source": SOURCE_ID,
        "source_revision": SOURCE_REVISION,
        "source_license": SOURCE_LICENSE,
        "source_sha256": source_hashes,
        "tokenizer_files_sha256": TOKENIZER_FILES,
        "profile": profile,
        "native_score_prompt": prompt_profile(pairs, tokenizer),
        "overlap": overlap,
        "publication_decision": (
            "ELIGIBLE_AS_OPEN_DEV_DIAGNOSTIC_ONLY"
            if overlap["status"] == "PASS_BOUNDED_LEXICAL_SCREEN"
            else "HOLD_PUBLIC_DEV_DIAGNOSTIC_PENDING_COMPLETE_PROTECTION"
        ),
        "evaluation_role": "open development diagnostic only; never untouched release test",
        "gpu_hours": 0,
    }


def write_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("Refusing to replace an existing audit receipt")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-directory", type=Path, required=True)
    parser.add_argument("--tokenizer-directory", type=Path, required=True)
    parser.add_argument("--protected-inventory", type=Path)
    parser.add_argument("--protected-inventory-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.protected_inventory) != bool(args.protected_inventory_sha256):
        parser.error("Protected inventory path and SHA-256 must be supplied together")
    roles = (
        load_projected_roles(
            args.protected_inventory, expected_sha256=args.protected_inventory_sha256
        )
        if args.protected_inventory
        else None
    )
    result = audit(
        args.dev_directory,
        args.tokenizer_directory,
        protected_roles=roles,
        protected_inventory_sha256=args.protected_inventory_sha256,
    )
    write_once(args.output, result)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "rows": result["profile"]["total_rows"],
                "overlap_status": result["overlap"]["status"],
                "publication_decision": result["publication_decision"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
