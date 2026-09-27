"""Freeze separate private source, prompt, answer and proof snapshots for v2.

This prospective DEV feasibility pilot creates no blinded reviewer packet and
does not touch protected FINAL or run models. Raw inputs and outputs remain in
private remote storage; only aggregate commitments enter public notes.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jev_arena.authored_release_scale_v1 import (
    FORM_FAMILIES,
    OPERATIONS,
    file_sha,
    inspect,
    native_row,
    write_private,
)

VERSION = "jevarena-authored-release-scale-v2/dev-feasibility-2"


def _domain_witness_issues(case: dict[str, Any]) -> list[str]:
    """Reject physically impossible complete-source alternatives.

    The typed oracle establishes answer sensitivity, but cannot by itself
    establish that a hypothetical source is plausible for the decision
    domain. Keep these rules narrow and explicit; editorial review remains
    necessary for every surviving source.
    """
    if case["operation"] != "net_range":
        return []
    left, right = (item["data"] for item in case["sources"])
    variant = case["variant"]
    pairs = [("original", left, right)]
    pair_left = variant["data"] if variant["side"] == "left" else left
    pair_right = variant["data"] if variant["side"] == "right" else right
    pairs.append(("variant", pair_left, pair_right))
    for phase, base_left, base_right in list(pairs):
        witnesses = (
            case["witnesses"] if phase == "original" else case["variant_witnesses"]
        )
        pairs.extend(
            (
                (f"{phase}/{side}/{index}", value, base_right)
                if side == "left"
                else (f"{phase}/{side}/{index}", base_left, value)
            )
            for side in ("left", "right")
            for index, value in enumerate(witnesses[side])
        )
    issues = []
    for phase, source_left, source_right in pairs:
        gross = source_left.get("gross")
        tare = source_right.get("tare")
        if (
            type(gross) not in (int, float)
            or type(tare) not in (int, float)
            or tare < 0
            or gross < tare
        ):
            issues.append(f"{phase}: impossible gross/tare relation")
    return issues


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    path.chmod(0o600)


def _verify_serialized_prompts(
    cases: list[dict[str, Any]], expected: list[dict[str, Any]], path: Path
) -> None:
    actual = [json.loads(line) for line in path.read_text().splitlines()]
    if len(actual) != len(cases):
        raise ValueError("Serialized authored prompt count differs")
    for case, before, after in zip(cases, expected, actual, strict=True):
        for key in ("id", "state", "questions"):
            if json.dumps(before[key], ensure_ascii=False) != json.dumps(
                after[key], ensure_ascii=False
            ):
                raise ValueError("Serialized authored native prompt changed")
        question = after["questions"]["decision"]
        if (
            OPERATIONS[case["operation"]] == "choice"
            and list(question["criteria"]) != case["option_order"]
        ):
            raise ValueError("Serialized Choice option order differs from casebook")


def _distribution(
    cases: list[dict[str, Any]], proofs: list[dict[str, Any]]
) -> dict[str, Any]:
    noul = Counter(row["original"] for row in proofs if row["type"] == "noul")
    score = Counter(row["original"] for row in proofs if row["type"] == "score")
    choice_positions = Counter()
    hold_answers = 0
    for case, proof in zip(cases, proofs):
        if proof["type"] == "choice":
            choice_positions[case["option_order"].index(proof["original"]) + 1] += 1
            hold_answers += proof["original"] == "HOLD"
    if min(noul.get(True, 0), noul.get(False, 0)) < 2:
        raise ValueError("Noul original targets are not balanced")
    if (
        min(score.get(level, 0) for level in (0, 1, 2)) < 1
        or max(score.values()) > sum(score.values()) / 2
    ):
        raise ValueError("Score original targets omit or overconcentrate a band")
    if any(choice_positions[position] < 1 for position in (1, 2, 3, 4)):
        raise ValueError("Choice original target positions are incomplete")
    if hold_answers < 1:
        raise ValueError("No correct joint-evidence HOLD choice")
    return {
        "noul_false": noul[False],
        "noul_true": noul[True],
        "score_levels": {str(level): score[level] for level in (0, 1, 2)},
        "choice_positions": {
            str(position): choice_positions[position] for position in (1, 2, 3, 4)
        },
        "choice_hold_answers": hold_answers,
    }


def prepare(
    casebook: Path, output: Path, prereg: Path, source_commit: str
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("V2 private candidate cannot be overwritten")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("Source commit is not a full SHA-1")
    source = json.loads(casebook.read_text())
    cases = source["cases"]
    if not 12 <= len(cases) <= 18 or len({case["slug"] for case in cases}) != len(
        cases
    ):
        raise ValueError("V2 feasibility requires 12–18 unique cases")
    kinds = Counter(OPERATIONS[case["operation"]] for case in cases)
    if any(kinds[kind] < 4 for kind in ("choice", "noul", "score")):
        raise ValueError("Native family floor not met")
    if len({case["operation"] for case in cases}) < 12:
        raise ValueError("V2 needs at least twelve semantic mechanisms")
    if len({case["domain"] for case in cases}) < 6:
        raise ValueError("Domain coverage below v2 floor")
    forms = [item["form_family"] for case in cases for item in case["sources"]]
    if not set(forms) <= FORM_FAMILIES or len(set(forms)) < 9:
        raise ValueError("Canonical document-family coverage below v2 floor")
    if any(
        "\n\n" not in item["document"] for case in cases for item in case["sources"]
    ):
        raise ValueError("Every source needs at least two causal paragraphs")
    sources: list[dict[str, Any]] = []
    substitutions: list[dict[str, Any]] = []
    originals: list[dict[str, Any]] = []
    variants: list[dict[str, Any]] = []
    answers: list[dict[str, Any]] = []
    proofs: list[dict[str, Any]] = []
    for case in cases:
        if _domain_witness_issues(case):
            raise ValueError("Domain-invalid source necessity witness")
        proof = inspect(case)
        left, right = (item["data"] for item in case["sources"])
        sources.append(
            {
                key: value
                for key, value in case.items()
                if key not in {"witnesses", "variant", "variant_witnesses"}
            }
        )
        substitutions.append({"slug": case["slug"], **case["variant"]})
        originals.append(native_row(case, left, right, case["slug"]))
        pair_left = (
            case["variant"]["data"] if case["variant"]["side"] == "left" else left
        )
        pair_right = (
            case["variant"]["data"] if case["variant"]["side"] == "right" else right
        )
        variants.append(native_row(case, pair_left, pair_right, case["slug"] + ":pair"))
        kind = OPERATIONS[case["operation"]]
        answers.append(
            {
                "slug": case["slug"],
                "type": kind,
                "original": proof["original"],
                "variant": proof["variant"],
            }
        )
        proofs.append(
            {
                "slug": case["slug"],
                "type": kind,
                **proof,
                "witness_source_completions": case["witnesses"],
                "variant_witness_source_completions": case["variant_witnesses"],
            }
        )
    balance = _distribution(cases, proofs)
    output.mkdir(mode=0o700, parents=True)
    components = {
        "sources": sources,
        "substitutions": substitutions,
        "originals": originals,
        "variants": variants,
        "answers": answers,
        "proofs": proofs,
    }
    hashes = {}
    for name, rows in components.items():
        path = output / f"{name}.private.jsonl"
        _write_rows(path, rows)
        hashes[name] = file_sha(path)
    _verify_serialized_prompts(cases, originals, output / "originals.private.jsonl")
    _verify_serialized_prompts(cases, variants, output / "variants.private.jsonl")
    receipt = {
        "version": VERSION,
        "status": "PRIVATE_V2_CANDIDATE_PREFLIGHT_PENDING_NO_BLIND_PACKET",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "prereg_sha256": file_sha(prereg),
        "builder_sha256": file_sha(Path(__file__)),
        "input_casebook_sha256": file_sha(casebook),
        "components_sha256": hashes,
        "independent_original_candidates": len(cases),
        "paired_variants_not_independent": len(variants),
        "by_type": dict(kinds),
        "semantic_mechanisms": len({case["operation"] for case in cases}),
        "document_families": len(set(forms)),
        "target_distribution": balance,
        "model_inference": False,
        "reviewer_packet_created": False,
        "release_qualified": False,
    }
    write_private(output / "receipt.private.json", receipt)
    return {
        key: receipt[key]
        for key in (
            "status",
            "independent_original_candidates",
            "paired_variants_not_independent",
            "by_type",
            "semantic_mechanisms",
            "document_families",
            "target_distribution",
            "source_commit",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--casebook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.casebook, args.output, args.prereg, args.source_commit),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
