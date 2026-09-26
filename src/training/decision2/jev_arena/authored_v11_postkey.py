"""Verify the two sealed v11 blind reviews before an aggregate-only key join.

The private candidate and review files must stay outside the public source
tree. This program never writes individual targets or reviewer answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

EXPECTED = {
    "prompts.jsonl": "94b9ff5db3735ca59e0197700a7a4a2e16b9fd8d045cbfc87ffeb0a8ef6187d5",
    "deletions.gold-free.jsonl": "7a47e0ce13e005e9ac1ad1e7c9271ce1e72026840250ff74f8b9fae9a8dde31d",
    "reviewer_manifest.gold-free.json": "9a663ac620795c96fbfc66cea18b1b306d1adc8f40124e23b2c78e0fa6793a93",
    "reviewer-freeze.gold-free.json": "267f17f13006911b8a258b217a465ef76dface378bee8d9294b856eda12e049b",
    "private/targets.jsonl": "7e576f2f7c3826d72bb0cbbeac28523adba280a7c94927025af079d14f8aeb33",
    "private/proofs.jsonl": "7eb0bafa4811843363d7aef89f83405d50999a6ba199d1682d439aa8c2271a70",
}
REVIEW_HASHES = {
    "originals.review.jsonl": "cbd8a6d905a5a6461eb5dd2c56e75a7ffa59ae39a337bddcd83b9ba8381a8233",
    "originals.summary.json": "d69a53ea532648764a8c506c521b56beeb4b30353b897083268523724c996eaf",
    "deletions.review.jsonl": "42b26ad39c2f975644429199d7e2020130592c014fea0c6f4eabda132933fc7b",
    "deletions.summary.json": "0d31154df4f1b17debe018c74a7864835147c2ea29787d9b05ad1eb76a933e2d",
}
SPEC_SHA = "f18420bb74852aa3d807e1513ee73e075018979ea50a75e4ae4ea55b62f5032c"
SALT_COMMITMENT = "8ce975488f1a6aafa8c50ed226677238099754dc71b910df1db9d0d6bb923850"
BUILDER_SHA = "ae538f1024c7859207d83b8d267a7c6306dc123be882c8bbe85cf9fa91d35d6a"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _expect_hashes(directory: Path, expected: dict[str, str]) -> None:
    for name, wanted in expected.items():
        if digest(directory / name) != wanted:
            raise ValueError(f"Frozen SHA differs: {name}")


def verify_seals(
    candidate: Path, review: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check every gold-free artifact and both seal times before key access."""
    _expect_hashes(candidate, EXPECTED)
    _expect_hashes(review, REVIEW_HASHES)
    freeze = json.loads((candidate / "reviewer-freeze.gold-free.json").read_text())
    manifest = json.loads((candidate / "reviewer_manifest.gold-free.json").read_text())
    originals = json.loads((review / "originals.summary.json").read_text())
    deletions = json.loads((review / "deletions.summary.json").read_text())
    if (
        freeze["prompts_sha256"] != EXPECTED["prompts.jsonl"]
        or freeze["deletions_sha256"] != EXPECTED["deletions.gold-free.jsonl"]
    ):
        raise ValueError("Freeze receipt differs from packet")
    if freeze["manifest_sha256"] != EXPECTED["reviewer_manifest.gold-free.json"]:
        raise ValueError("Freeze receipt differs from reviewer manifest")
    if (
        manifest["originals"]["sha256"] != EXPECTED["prompts.jsonl"]
        or manifest["deletions"]["sha256"] != EXPECTED["deletions.gold-free.jsonl"]
        or manifest["originals"]["count"] != 12
        or manifest["deletions"]["count"] != 39
        or manifest["spec_sha256"] != SPEC_SHA
        or manifest["salt_commitment_sha256"] != SALT_COMMITMENT
        or manifest["builder_sha256"] != BUILDER_SHA
    ):
        raise ValueError("Manifest differs from preregistered candidate")
    if (
        originals["stage"] != "originals_gold_blind"
        or originals["per_item_sha256"] != REVIEW_HASHES["originals.review.jsonl"]
        or originals["source_files_sha256"]
        != {
            "prompts.jsonl": EXPECTED["prompts.jsonl"],
            "reviewer-freeze.gold-free.json": EXPECTED[
                "reviewer-freeze.gold-free.json"
            ],
            "reviewer_manifest.gold-free.json": EXPECTED[
                "reviewer_manifest.gold-free.json"
            ],
        }
        or originals["deletions_opened_before_seal"] is not False
        or deletions["stage"] != "single_exhibit_deletions_gold_blind"
        or deletions["per_item_sha256"] != REVIEW_HASHES["deletions.review.jsonl"]
        or deletions["original_stage_summary_sha256"]
        != REVIEW_HASHES["originals.summary.json"]
        or deletions["source_sha256"] != EXPECTED["deletions.gold-free.jsonl"]
        or deletions["private_gold_or_proof_access"] is not False
    ):
        raise ValueError("Blind stage or cross-stage receipt differs")
    original_time = datetime.fromisoformat(originals["sealed_at_utc"])
    deletion_time = datetime.fromisoformat(deletions["sealed_at_utc"])
    if original_time.tzinfo is None or deletion_time.tzinfo is None:
        raise ValueError("Seal times need time zones")
    files = [
        candidate / "reviewer-freeze.gold-free.json",
        review / "originals.review.jsonl",
        review / "originals.summary.json",
        review / "deletions.review.jsonl",
        review / "deletions.summary.json",
    ]
    if not all(a.stat().st_mtime < b.stat().st_mtime for a, b in zip(files, files[1:])):
        raise ValueError("Review artifacts are not in frozen stage order")
    if not original_time < deletion_time or any(
        abs(sealed.timestamp() - (review / filename).stat().st_mtime) > 2
        for sealed, filename in (
            (original_time, "originals.summary.json"),
            (deletion_time, "deletions.summary.json"),
        )
    ):
        raise ValueError("Seal declarations conflict with file times")
    return originals, deletions


def answer_text(kind: str, value: Any) -> str:
    if kind == "noul" and type(value) is bool:
        return "Certified" if value else "Not certified"
    if kind == "score" and type(value) is int:
        return f"Grade {value}"
    if kind == "choice" and type(value) is str:
        return value
    raise ValueError("Invalid typed target")


def verify(
    candidate: Path, review: Path, spec: Path, salt: Path, builder: Path
) -> dict[str, Any]:
    original_summary, deletion_summary = verify_seals(candidate, review)
    if (
        digest(spec) != SPEC_SHA
        or digest(salt) != SALT_COMMITMENT
        or digest(builder) != BUILDER_SHA
    ):
        raise ValueError("Private specification, salt or signed builder differs")
    # All seals and source commitments have been checked; key access starts here.
    prompts = jsonl(candidate / "prompts.jsonl")
    deletions = jsonl(candidate / "deletions.gold-free.jsonl")
    originals_review = jsonl(review / "originals.review.jsonl")
    deletions_review = jsonl(review / "deletions.review.jsonl")
    targets = jsonl(candidate / "private/targets.jsonl")
    proofs = jsonl(candidate / "private/proofs.jsonl")
    if not (len(prompts) == len(targets) == len(proofs) == len(originals_review) == 12):
        raise ValueError("Original row counts differ")
    if not (len(deletions) == len(deletions_review) == 39):
        raise ValueError("Deletion row counts differ")
    ids = [row["id"] for row in prompts]
    if len(set(ids)) != 12 or any(
        {row["id"] for row in table} != set(ids)
        for table in (targets, proofs, originals_review)
    ):
        raise ValueError("Original ID sets differ")
    by_target = {row["id"]: row for row in targets}
    by_proof = {row["id"]: row for row in proofs}
    matches: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    by_blind = {}
    for row in originals_review:
        target = by_target[row["id"]]
        kind = target["kind"]
        gold = target["answer"][kind]
        if row["type"] != kind or by_proof[row["id"]]["answer"] != gold:
            raise ValueError("Review type, target or proof differs")
        if original_summary["answers_by_id"][row["id"]] != row["blind_answer"]:
            raise ValueError("Original review differs from sealed summary")
        type_counts[kind] += 1
        matches[kind] += row["blind_answer"] == answer_text(kind, gold)
        by_blind[row["id"]] = row["blind_answer"]
    if set(type_counts) != {"choice", "noul", "score"} or any(
        type_counts[kind] != 4 for kind in type_counts
    ):
        raise ValueError("Type quota differs")
    expected_pairs = [(r["parent_id"], r["omitted_source"]) for r in deletions]
    observed_pairs = [(r["parent_id"], r["omitted_source"]) for r in deletions_review]
    if (
        expected_pairs != observed_pairs
        or len(set(expected_pairs)) != 39
        or any(row["index"] != i for i, row in enumerate(deletions_review, start=1))
    ):
        raise ValueError("Deletion rows, linkage or order differ")
    if any(
        row["original_blind_answer"] != by_blind[row["parent_id"]]
        for row in deletions_review
    ):
        raise ValueError("Deletion-stage original answer differs")
    ambiguity_indices = [
        r["index"] for r in deletions_review if r["material_ambiguity"]
    ]
    if (
        original_summary["items_reviewed"] != 12
        or deletion_summary["removals_reviewed"] != 39
        or deletion_summary["material_ambiguity_indices"] != ambiguity_indices
        or deletion_summary["aggregate_verdict"] != "BLOCK_ADMISSION"
    ):
        raise ValueError("Review aggregate differs from sealed rows")
    return {
        "status": "BLOCK_FOR_RELEASE_BENCH",
        "release_qualified": False,
        "verifier_sha256": digest(Path(__file__)),
        "scope": "private DEV editorial audit only; no model score or FINAL access",
        "seal_order_verified": True,
        "frozen_hashes": {"candidate": EXPECTED, "reviews": REVIEW_HASHES},
        "originals": {
            "count": 12,
            "type_counts": dict(sorted(type_counts.items())),
            "blind_matches_gold": dict(sorted(matches.items())),
            "total_matches": sum(matches.values()),
            "blind_material_ambiguity_count": sum(
                r["material_ambiguity"] for r in originals_review
            ),
        },
        "deletions": {
            "count": 39,
            "original_answer_entailed_count": sum(
                r["original_answer_entailed_by_remaining_state"]
                for r in deletions_review
            ),
            "material_ambiguity_count": len(ambiguity_indices),
            "material_ambiguity_indices": ambiguity_indices,
        },
        "editorial_blockers": [
            "Four ordered-milestone deletions leave the stage universe and ordering underspecified; three remaining completions can support a different determinate grade.",
            "One rate-rubric deletion leaves a conventional-band hint, even though exact original grade is not entailed.",
            "Uniform source-removal instruction and long-case qualifiers lack editorial variation and evidence density.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--salt", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Post-key audit cannot overwrite a prior receipt")
    report = verify(
        args.candidate_dir, args.review_dir, args.spec, args.salt, args.builder
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output.chmod(0o600)
    print(
        json.dumps(
            {
                "status": report["status"],
                "seal_order_verified": report["seal_order_verified"],
                "originals": report["originals"],
                "deletions": report["deletions"],
                "report_sha256": digest(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
