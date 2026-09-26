"""Verify two independent v12 seals before a private aggregate-only key join.

The candidate, reviews, keys and raw row judgments stay in the authorized
private workspace. This program writes no individual answer or target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GOLD_FREE = {
    "reviewer-a/originals.jsonl": "124677770eeb3f36fae9914bf08cb863b62fa7b838b95f5378297c8406a24d25",
    "reviewer-a/manifest.json": "55a07f3d7c0c35de50a4d72f2c35f24569e9e4a5e44f2fd960cefeb3015b91d7",
    "reviewer-b/variants.jsonl": "df3c5013153f6e9d47e7fae81b27e11179691a05e3603e9f0768e62e034bcb42",
    "reviewer-b/manifest.json": "cd388d7bcbb8b61f48e421bf81993364f2d83a66ab184963943723655b0784fb",
    "freeze.gold-free.json": "17b054a3807ba7683e899a09f0b1bac636f1ec61df5e22accd6b1de96b13118f",
}
REVIEW_A = {
    "row_judgments.jsonl": "19cc97c12279b40e494062270a167f28e4e093f024fa086108461c3affcb2c8d",
    "summary.json": "1fa1a08f314e0610235962181e382d695e38772bd6b84dcd1884d5ca92e52089",
    "seal.json": "0c10c0e7b8a50d84d09f420c658bc453bcb12691c8e97193dab32432050379cb",
}
REVIEW_B = {
    "row_judgments.jsonl": "dcfa2abebe87d297cdebecebd655fad2818927fbffbbab73ee243035f880c498",
    "aggregate.json": "7a131b2a9d55d93279c8154ca757f35fdb6ca49c0c689bd37aab5612cb663ef6",
    "seal.json": "56883117ae024e51d497b8853c6d1a077b9e1bfb766fdc813fff06cd58e91883",
}
PRIVATE = {
    "private/targets.jsonl": "3eb33539fc5426d2d665a0c784130935aec4044951736ae889801684b741ae36",
    "private/variant_targets.jsonl": "89f50e2be94a5e29782a6d85ed9245a493e340c033f7d37feeb40589c3b37618",
    "private/all_source_withdrawals.jsonl": "73d00d46adf810243a2e9f18c9003d2853363240c9795861fc47b123d5cc5bc7",
    "private/withdrawal_targets.jsonl": "bdc2decea9fe3cfcf6800737dbb90a195ed6064eb77ed509a0e8dc7751e09a2e",
    "private/proofs.jsonl": "53a04bec1c6db6f78e38876fb618e9a7b8d8ea9689e9fe48291677fbb8e99f8d",
    "private/join.jsonl": "cae399d6d5f919e71261ec5eeda8f8d9cdf0a9d02045548729521c150854f49c",
}
SPEC_SHA = "0d5f33dff897d3ee2d30599f864ccd31f107989358dd1a83e864cdf8dc046c1d"
BUILDER_SHA = "10ee20d3cd3d960a837717a0286cded9dfaf67d7406c543ac422bcdf78773a43"
SALT_A_SHA = "e435647ea55388db5865ab78264c518383622b9ec6174fdba1d70c424d416829"
SALT_B_SHA = "b2cfdb5e3762e6aa4dce534d2c0679a156f3cb9fa0f0706542886001e836dc8b"
MISSING = "insufficient evidence"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def check_hashes(directory: Path, expected: dict[str, str]) -> None:
    for name, wanted in expected.items():
        if digest(directory / name) != wanted:
            raise ValueError(f"Frozen hash mismatch: {name}")


def aware_time(value: str) -> datetime:
    if value.endswith(" UTC"):
        parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S UTC").replace(
            tzinfo=timezone.utc
        )
    else:
        parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("Seal time has no time zone")
    return parsed.astimezone(timezone.utc)


def distinct_ids(rows: list[dict[str, Any]], expected: int) -> set[str]:
    ids = [row["id"] for row in rows]
    if len(ids) != expected or len(set(ids)) != expected:
        raise ValueError("Review row count or IDs differ")
    return set(ids)


def preflight(candidate: Path, review_a: Path, review_b: Path) -> dict[str, Any]:
    """Read only gold-free packets, review files and their seal receipts."""
    check_hashes(candidate, GOLD_FREE)
    check_hashes(review_a, REVIEW_A)
    check_hashes(review_b, REVIEW_B)
    freeze = read_json(candidate / "freeze.gold-free.json")
    manifest_a = read_json(candidate / "reviewer-a/manifest.json")
    manifest_b = read_json(candidate / "reviewer-b/manifest.json")
    a_summary = read_json(review_a / "summary.json")
    a_seal = read_json(review_a / "seal.json")
    b_aggregate = read_json(review_b / "aggregate.json")
    b_seal = read_json(review_b / "seal.json")
    if (
        freeze["originals_sha256"] != GOLD_FREE["reviewer-a/originals.jsonl"]
        or freeze["variants_sha256"] != GOLD_FREE["reviewer-b/variants.jsonl"]
        or freeze["manifest_a_sha256"] != GOLD_FREE["reviewer-a/manifest.json"]
        or freeze["manifest_b_sha256"] != GOLD_FREE["reviewer-b/manifest.json"]
        or freeze["builder_sha256"] != BUILDER_SHA
    ):
        raise ValueError("Gold-free freeze receipt differs")
    for manifest, role, packet in (
        (manifest_a, "independent-original-reviewer", "reviewer-a/originals.jsonl"),
        (manifest_b, "independent-variant-reviewer", "reviewer-b/variants.jsonl"),
    ):
        if (
            manifest["role"] != role
            or manifest["count"] != 8
            or manifest["sha256"] != GOLD_FREE[packet]
        ):
            raise ValueError("Reviewer manifest differs")
    if (
        a_summary["role"] != "independent-original-reviewer"
        or a_summary["manifest_sha256"] != GOLD_FREE["reviewer-a/manifest.json"]
        or a_summary["packet_sha256"] != GOLD_FREE["reviewer-a/originals.jsonl"]
        or a_summary["review_sha256"] != REVIEW_A["row_judgments.jsonl"]
        or a_summary["rows_reviewed"] != 8
        or a_seal["review_sha256"] != REVIEW_A["row_judgments.jsonl"]
        or a_seal["summary_sha256"] != REVIEW_A["summary.json"]
        or a_seal["sealed_at_utc"] != a_summary["sealed_at_utc"]
    ):
        raise ValueError("Reviewer A summary or seal differs")
    if (
        b_aggregate["manifest_sha256"] != GOLD_FREE["reviewer-b/manifest.json"]
        or b_aggregate["packet_sha256"] != GOLD_FREE["reviewer-b/variants.jsonl"]
        or b_aggregate["reviewed_rows"] != 8
        or b_aggregate["gold_or_original_access"] is not False
        or b_aggregate["models_run"] is not False
        or b_seal["manifest_sha256"] != GOLD_FREE["reviewer-b/manifest.json"]
        or b_seal["packet_sha256"] != GOLD_FREE["reviewer-b/variants.jsonl"]
        or b_seal["row_judgments_sha256"] != REVIEW_B["row_judgments.jsonl"]
        or b_seal["aggregate_sha256"] != REVIEW_B["aggregate.json"]
        or b_seal["reviewed_rows"] != 8
        or b_seal["verdict"] != b_aggregate["verdict"]
    ):
        raise ValueError("Reviewer B aggregate or seal differs")
    originals = read_jsonl(candidate / "reviewer-a/originals.jsonl")
    variants = read_jsonl(candidate / "reviewer-b/variants.jsonl")
    a_review = read_jsonl(review_a / "row_judgments.jsonl")
    b_review = read_jsonl(review_b / "row_judgments.jsonl")
    original_ids = distinct_ids(originals, 8)
    variant_ids = distinct_ids(variants, 8)
    if (
        original_ids & variant_ids
        or distinct_ids(a_review, 8) != original_ids
        or distinct_ids(b_review, 8) != variant_ids
        or any(set(row) != {"id", "state", "questions"} for row in originals + variants)
        or any(
            forbidden in json.dumps(row)
            for row in variants
            for forbidden in ("parent_id", "omitted_source", "perturbation")
        )
    ):
        raise ValueError("Review roster or blinding differs")
    if (
        a_summary["ambiguous"] != sum(row["ambiguous"] for row in a_review)
        or a_summary["directly_solvable"]
        != sum(not row["ambiguous"] for row in a_review)
        or set(b_aggregate["answer_not_in_native_criteria_ids"])
        != {row["id"] for row in b_review if not row["answer_listed_in_criteria"]}
        or sum(b_aggregate["direct_answer_counts"].values()) != 8
        or set(b_aggregate["superseded_record_redundancy_ids"]) - variant_ids
    ):
        raise ValueError("Sealed review aggregates differ from rows")
    frozen_at = aware_time(freeze["utc_sealed_at"])
    a_time = aware_time(a_seal["sealed_at_utc"])
    b_time = aware_time(b_seal["sealed_at_utc"])
    if not frozen_at < a_time < b_time:
        raise ValueError("Freeze and review chronology differs")
    if not (
        (candidate / "freeze.gold-free.json").stat().st_mtime
        < (review_a / "row_judgments.jsonl").stat().st_mtime
        < (review_b / "row_judgments.jsonl").stat().st_mtime
    ):
        raise ValueError("Preserved file times conflict with review chronology")
    if (
        abs(a_time.timestamp() - (review_a / "seal.json").stat().st_mtime) > 2
        or abs(b_time.timestamp() - (review_b / "seal.json").stat().st_mtime) > 2
    ):
        raise ValueError("Declared seal time conflicts with file time")
    return {
        "status": "SEALED_GOLD_FREE_REVIEW_VERIFIED",
        "freeze_utc": frozen_at.isoformat(),
        "review_a_utc": a_time.isoformat(),
        "review_b_utc": b_time.isoformat(),
        "packet_hashes": GOLD_FREE,
        "review_a_hashes": REVIEW_A,
        "review_b_hashes": REVIEW_B,
        "rows": {"originals": 8, "variants": 8},
    }


def normalized(kind: str, value: Any) -> str | bool | int:
    if type(value) is bool and kind == "noul":
        return value
    if type(value) is int and kind == "score" and 0 <= value <= 4:
        return value
    if type(value) is not str:
        raise ValueError("Blind answer type differs")
    value = value.strip()
    if value.casefold() == MISSING:
        return MISSING
    if kind == "noul":
        if value.casefold() in {"true", "certified"}:
            return True
        if value.casefold() in {"false", "not certified"}:
            return False
    if kind == "score":
        match = re.fullmatch(r"(?:grade\s*)?([0-4])", value, re.I)
        if match:
            return int(match.group(1))
    if kind == "choice":
        return value
    raise ValueError("Blind answer has no typed interpretation")


def postkey(
    candidate: Path,
    review_a: Path,
    review_b: Path,
    preflight_receipt: Path,
) -> dict[str, Any]:
    sealed = preflight(candidate, review_a, review_b)
    if read_json(preflight_receipt) != sealed:
        raise ValueError("Prior gold-free preflight receipt differs")
    check_hashes(candidate, PRIVATE)
    source_dir = candidate.parent
    for path, wanted in (
        (source_dir / "specs-r4.json", SPEC_SHA),
        (source_dir / "builder-r4.py", BUILDER_SHA),
        (candidate / "private/salt-a.bin", SALT_A_SHA),
        (candidate / "private/salt-b.bin", SALT_B_SHA),
    ):
        if digest(path) != wanted:
            raise ValueError("Private source or salt commitment differs")
    private_receipt = read_json(candidate / "private/receipt.json")
    if (
        any(
            private_receipt[name + "_sha256"] != wanted
            for name, wanted in (
                ("targets", PRIVATE["private/targets.jsonl"]),
                ("variant_targets", PRIVATE["private/variant_targets.jsonl"]),
                ("withdrawals", PRIVATE["private/all_source_withdrawals.jsonl"]),
                ("withdrawal_targets", PRIVATE["private/withdrawal_targets.jsonl"]),
                ("proofs", PRIVATE["private/proofs.jsonl"]),
                ("join", PRIVATE["private/join.jsonl"]),
            )
        )
        or private_receipt["spec_sha256"] != SPEC_SHA
        or private_receipt["builder_sha256"] != BUILDER_SHA
    ):
        raise ValueError("Private freeze receipt differs")
    # The gold-free preflight is complete. Private key access starts here.
    targets = read_jsonl(candidate / "private/targets.jsonl")
    variant_targets = read_jsonl(candidate / "private/variant_targets.jsonl")
    withdrawals = read_jsonl(candidate / "private/all_source_withdrawals.jsonl")
    withdrawal_targets = read_jsonl(candidate / "private/withdrawal_targets.jsonl")
    proofs = read_jsonl(candidate / "private/proofs.jsonl")
    joins = read_jsonl(candidate / "private/join.jsonl")
    originals = read_jsonl(candidate / "reviewer-a/originals.jsonl")
    variants = read_jsonl(candidate / "reviewer-b/variants.jsonl")
    a_review = read_jsonl(review_a / "row_judgments.jsonl")
    b_review = read_jsonl(review_b / "row_judgments.jsonl")
    b_aggregate = read_json(review_b / "aggregate.json")
    if not (
        len(targets) == len(variant_targets) == len(proofs) == len(joins) == 8
        and len(withdrawals) == len(withdrawal_targets) == 16
        and distinct_ids(targets, 8) == distinct_ids(originals, 8)
        and distinct_ids(variant_targets, 8) == distinct_ids(variants, 8)
        and distinct_ids(withdrawals, 16) == distinct_ids(withdrawal_targets, 16)
    ):
        raise ValueError("Private target, proof or join roster differs")
    original_target = {row["id"]: row for row in targets}
    variant_target = {row["id"]: row for row in variant_targets}
    proof = {row["original_id"]: row for row in proofs}
    original_prompt = {row["id"]: row for row in originals}
    variant_prompt = {row["id"]: row for row in variants}
    if (
        {row["original_id"] for row in joins} != set(original_target)
        or {row["variant_id"] for row in joins} != set(variant_target)
        or set(proof) != set(original_target)
        or any(
            row["reported_status"] != MISSING
            or row["completion_answers"][0] == row["completion_answers"][1]
            for row in withdrawal_targets
        )
        or any(
            len(row["sources"]) != 2
            or any(
                source["withdrawal_status"] != MISSING
                or source["answer_after_substitution"] == row["original_answer"]
                for source in row["sources"]
            )
            for row in proofs
        )
    ):
        raise ValueError("Private source-sensitivity proof differs")
    for link in joins:
        original_id, variant_id = link["original_id"], link["variant_id"]
        target = original_target[original_id]
        p = proof[original_id]
        if (
            p["original_answer"] != target["answer"][target["kind"]]
            or p["variant_answer"] != variant_target[variant_id]["answer"]
            or original_prompt[original_id]["questions"]
            != variant_prompt[variant_id]["questions"]
        ):
            raise ValueError("Original, variant and proof join differs")
    by_a = {row["id"]: row for row in a_review}
    by_b = {row["id"]: row for row in b_review}
    a_matches: Counter[str] = Counter()
    a_counts: Counter[str] = Counter()
    b_matches: Counter[str] = Counter()
    b_counts: Counter[str] = Counter()
    native_missing = 0
    for ident, target in original_target.items():
        kind = target["kind"]
        gold = target["answer"][kind]
        if type(gold) is not {"choice": str, "noul": bool, "score": int}[kind]:
            raise ValueError("Original target type differs")
        a_counts[kind] += 1
        a_matches[kind] += normalized(kind, by_a[ident]["direct_answer"]) == gold
    for ident, target in variant_target.items():
        kind = target["kind"]
        gold = target["answer"]
        if by_b[ident]["task_type"] != kind:
            raise ValueError("Variant review type differs")
        answer = normalized(kind, by_b[ident]["answer"])
        b_counts[kind] += 1
        b_matches[kind] += type(answer) is type(gold) and answer == gold
        criteria = variant_prompt[ident]["questions"]["decision"]["criteria"]
        listed = (
            (answer in criteria)
            if kind == "choice"
            else (
                answer in {True, False}
                if kind == "noul"
                else type(answer) is int and 0 <= answer <= 4
            )
        )
        if listed != by_b[ident]["answer_listed_in_criteria"]:
            raise ValueError("Variant native-criteria judgment differs")
        native_missing += not listed
    if a_counts != {"choice": 4, "noul": 2, "score": 2} or b_counts != a_counts:
        raise ValueError("Typed roster counts differ")
    if native_missing != len(b_aggregate["answer_not_in_native_criteria_ids"]):
        raise ValueError("Native criteria aggregate differs")
    correction_ids = {
        row["variant_id"] for row in joins if row["perturbation"] == "correction"
    }
    if correction_ids != set(b_aggregate["superseded_record_redundancy_ids"]):
        raise ValueError("Correction redundancy aggregate differs")
    status = "HOLD_EDITORIAL_QA"
    return {
        "status": status,
        "release_qualified": False,
        "training_admitted": False,
        "model_inference": False,
        "scope": "sealed private DEV editorial audit only",
        "prekey_verified": True,
        "preflight_receipt_sha256": digest(preflight_receipt),
        "verifier_sha256": digest(Path(__file__)),
        "frozen_hashes": {
            "gold_free": GOLD_FREE,
            "review_a": REVIEW_A,
            "review_b": REVIEW_B,
        },
        "chronology_utc": {
            "candidate": sealed["freeze_utc"],
            "review_a": sealed["review_a_utc"],
            "review_b": sealed["review_b_utc"],
        },
        "originals": {
            "count": 8,
            "type_counts": dict(a_counts),
            "blind_matches": dict(a_matches),
            "total_matches": sum(a_matches.values()),
            "blind_ambiguity_count": sum(row["ambiguous"] for row in a_review),
            "suspected_shortcut_count": sum(
                row["suspected_shortcut"] for row in a_review
            ),
        },
        "variants": {
            "count": 8,
            "type_counts": dict(b_counts),
            "blind_matches": dict(b_matches),
            "total_matches": sum(b_matches.values()),
            "not_in_native_criteria_count": native_missing,
            "superseded_record_redundancy_count": len(correction_ids),
            "missing_or_redacted_fact_count": 6,
        },
        "source_withdrawals": {
            "count": 16,
            "answer_changing_completion_count": len(withdrawal_targets),
        },
        "editorial_blockers": [
            "Three blind uncertainty answers are absent from their Noul or Score native criteria.",
            "Both correction variants make a superseded full record unnecessary for the current answer.",
            "Six of eight variants reduce to detecting a missing or redacted fact; the original reviewer also noted repeated rule framing and possible residual evidence cues.",
        ],
    }


def write_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError("A sealed audit receipt cannot be overwritten")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    path.chmod(0o600)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--review-a-dir", type=Path, required=True)
    parser.add_argument("--review-b-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.preflight_only:
        result = preflight(args.candidate_dir, args.review_a_dir, args.review_b_dir)
    else:
        if args.preflight_receipt is None:
            parser.error("--preflight-receipt is required for post-key verification")
        result = postkey(
            args.candidate_dir,
            args.review_a_dir,
            args.review_b_dir,
            args.preflight_receipt,
        )
    write_once(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "receipt_sha256": digest(args.output),
                "original_matches": result.get("originals", {}).get("total_matches"),
                "variant_matches": result.get("variants", {}).get("total_matches"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
