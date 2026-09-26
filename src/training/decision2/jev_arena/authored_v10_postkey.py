"""Verify sealed authored-v10 blind reviews and write aggregate post-key audit.

This script reads private targets only after the two independently sealed
review stages exist. Its output is aggregate-only and never prints case gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .authored_v5_dossier import opaque

EXPECTED_VERSION = "jevarena-authored-v10-dev12-editorial-pilot/1"
EXPECTED_SOURCE_COMMIT = "dbc95ca94ae392e4ab5f796012d95bbc74b29950"
EXPECTED_PREREG_COMMIT = "e66576f2c1a4787b7b5ee6f514366ca8b407e3ea"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def sealed_review(review: Path, stem: str) -> tuple[str, str, datetime]:
    seal = review / f"{stem}-seal.sha256"
    stamp = review / f"{stem}-seal.utc"
    expected = {
        f"{stem}-review.jsonl",
        f"{stem}-summary.md",
    }
    entries = seal.read_text().splitlines()
    if len(entries) != 2:
        raise ValueError(f"{stem} seal must contain two hashes")
    seen = set()
    for entry in entries:
        digest, separator, filename = entry.partition("  ")
        if (
            not separator
            or filename not in expected
            or filename in seen
            or len(digest) != 64
            or sha(review / filename) != digest
        ):
            raise ValueError(f"{stem} seal is invalid")
        seen.add(filename)
    if seen != expected:
        raise ValueError(f"{stem} seal omits a review artifact")
    sealed_at = datetime.fromisoformat(stamp.read_text().strip().replace("Z", "+00:00"))
    if sealed_at.tzinfo is None:
        raise ValueError(f"{stem} seal has no timezone")
    if max((review / name).stat().st_mtime for name in expected) > seal.stat().st_mtime:
        raise ValueError(f"{stem} review changed after sealing")
    if abs(sealed_at.timestamp() - stamp.stat().st_mtime) > 30:
        raise ValueError(f"{stem} seal timestamp differs from file time")
    return (
        sha(review / f"{stem}-review.jsonl"),
        sha(review / f"{stem}-summary.md"),
        sealed_at,
    )


def verify(candidate: Path, specs: Path, salt: Path, builder: Path) -> dict[str, Any]:
    private = candidate / "private"
    review = candidate / "blind-review-independent"
    audit = json.loads((private / "audit.json").read_text())
    manifest = json.loads((candidate / "reviewer_manifest.gold-free.json").read_text())
    if audit["version"] != EXPECTED_VERSION or manifest["version"] != EXPECTED_VERSION:
        raise ValueError("Version differs from signed source")
    if (
        manifest["source_commit"] != EXPECTED_SOURCE_COMMIT
        or manifest["preregistered_commit"] != EXPECTED_PREREG_COMMIT
    ):
        raise ValueError("Source or preregistered commit differs")
    digest_paths = {
        "spec_sha256": specs,
        "salt_commitment_sha256": salt,
        "builder_sha256": builder,
        "prompts_sha256": candidate / "prompts.jsonl",
        "ablations_sha256": candidate / "ablations.gold-free.jsonl",
        "targets_sha256": private / "targets.jsonl",
        "proof_sha256": private / "proof_traces.jsonl",
    }
    if any(sha(path) != audit[key] for key, path in digest_paths.items()):
        raise ValueError("Frozen input digest differs")
    if (
        manifest["source_spec_sha256"] != audit["spec_sha256"]
        or manifest["salt_commitment_sha256"] != audit["salt_commitment_sha256"]
        or manifest["builder_sha256"] != audit["builder_sha256"]
        or manifest["originals"]["sha256"] != audit["prompts_sha256"]
        or manifest["source_deletions"]["sha256"] != audit["ablations_sha256"]
        or manifest["originals"]["count"] != 12
        or manifest["source_deletions"]["count"] != 42
    ):
        raise ValueError("Reviewer manifest differs from author freeze")

    orig_hash, orig_summary_hash, orig_seal_time = sealed_review(review, "originals")
    ab_hash, ab_summary_hash, ab_seal_time = sealed_review(review, "ablations")
    if not (
        (candidate / "prompts.jsonl").stat().st_mtime
        <= (candidate / "reviewer_manifest.gold-free.json").stat().st_mtime
        < (review / "originals-review.jsonl").stat().st_mtime
        <= (review / "originals-seal.sha256").stat().st_mtime
        < (review / "ablations-review.jsonl").stat().st_mtime
        <= (review / "ablations-seal.sha256").stat().st_mtime
        < datetime.now(timezone.utc).timestamp()
        and orig_seal_time < ab_seal_time
        and (review / "originals-seal.sha256").stat().st_mtime
        < (review / "ablations-review.jsonl").stat().st_mtime
    ):
        raise ValueError("Review stages are not in the frozen order")
    original_summary = (review / "originals-summary.md").read_text()
    ablation_summary = (review / "ablations-summary.md").read_text()
    if "BLOCK_FOR_RELEASE_BENCH" not in ablation_summary:
        raise ValueError("Blind editorial gate verdict missing")
    if "without opening ablations" not in original_summary:
        raise ValueError("Original-stage blind declaration missing")
    if "Only then" not in ablation_summary:
        raise ValueError("Ablation-stage blind declaration missing")

    prompts = rows(candidate / "prompts.jsonl")
    ablations = rows(candidate / "ablations.gold-free.jsonl")
    targets = rows(private / "targets.jsonl")
    proofs = rows(private / "proof_traces.jsonl")
    originals = rows(review / "originals-review.jsonl")
    deletions = rows(review / "ablations-review.jsonl")
    if not (len(prompts) == len(targets) == len(proofs) == len(originals) == 12):
        raise ValueError("Original row count differs")
    if not (len(ablations) == len(deletions) == 42):
        raise ValueError("Ablation row count differs")
    prompt_ids = {row["id"] for row in prompts}
    if len(prompt_ids) != 12 or any(
        {row["id"] for row in table} != prompt_ids
        for table in (targets, proofs, originals)
    ):
        raise ValueError("Original IDs differ")
    expected_ablation_pairs = [
        (row["parent_id"], row["omitted_source"]) for row in ablations
    ]
    observed_ablation_pairs = [
        (row["parent_id"], row["omitted_source"]) for row in deletions
    ]
    if (
        expected_ablation_pairs != observed_ablation_pairs
        or len(set(expected_ablation_pairs)) != 42
        or {row["index"] for row in deletions} != set(range(42))
        or any(row["index"] != index for index, row in enumerate(deletions))
    ):
        raise ValueError("Deletion join or order differs")

    by_target = {row["id"]: row for row in targets}
    by_proof = {row["id"]: row for row in proofs}
    secret = salt.read_bytes()
    essential_pairs = {
        (
            proof["id"],
            opaque(
                secret,
                f"v10:{proof['slug']}:source:{doc_slug}",
                12,
            ),
        ): field
        for proof in proofs
        for doc_slug, field in proof["essential_sources"].items()
    }
    if set(essential_pairs) != set(expected_ablation_pairs):
        raise ValueError("Deletion packet differs from frozen essential-source map")
    if any(
        row["omitted_field"]
        != essential_pairs[(row["parent_id"], row["omitted_source"])]
        for row in deletions
    ):
        raise ValueError("Reviewer omission field differs from frozen proof")
    if any(
        witness["distinct_outputs"] < 2
        for proof in proofs
        for witness in proof["sensitivity"].values()
    ):
        raise ValueError("Frozen proof lacks two distinct completion outputs")
    agreements = Counter()
    determinate = Counter()
    for row in originals:
        target = by_target[row["id"]]
        proof = by_proof[row["id"]]
        kind = target["kind"]
        gold = target["answer"][kind]
        if proof["answer"] != gold or type(proof["answer"]) is not type(gold):
            raise ValueError("Target and proof disagree")
        answer = row["blind_answer"]
        agreements[kind] += int(type(answer) is type(gold) and answer == gold)
        determinate[kind] += int(row["determinate"] is True)
    possible = Counter(by_target[row["id"]]["kind"] for row in originals)
    not_provable = sum(row["original_answer_provable"] is False for row in deletions)
    manifest_hash = sha(candidate / "reviewer_manifest.gold-free.json")
    return {
        "version": EXPECTED_VERSION,
        "status": "BLOCK_FOR_RELEASE_BENCH",
        "release_qualified": False,
        "training_approved": False,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "preregistered_commit": EXPECTED_PREREG_COMMIT,
        "frozen_hashes": {
            "manifest": manifest_hash,
            "prompts": audit["prompts_sha256"],
            "ablations": audit["ablations_sha256"],
            "targets": audit["targets_sha256"],
            "proof": audit["proof_sha256"],
            "spec": audit["spec_sha256"],
            "salt_commitment": audit["salt_commitment_sha256"],
            "builder": audit["builder_sha256"],
            "originals_blind": orig_hash,
            "originals_summary": orig_summary_hash,
            "ablations_blind": ab_hash,
            "ablations_summary": ab_summary_hash,
        },
        "seal_order_verified": True,
        "originals": {
            "count": 12,
            "type_counts": dict(possible),
            "determinate": dict(determinate),
            "blind_matches_gold": dict(agreements),
        },
        "source_deletions": {
            "count": 42,
            "blind_not_provable": not_provable,
        },
        "editorial_blockers": [
            "Surviving introductions cue missing status or the aggregate choice.",
            "Deletion leaves references to absent records in unchanged introductions.",
            "A valid ballot-seal record and a wrong-scope capacity source are not answer-discriminating.",
            "Long case prose repeats field-provenance and DATA-attestation boilerplate.",
        ],
        "scope": "private DEV editorial pilot only; no FINAL, model score or publication",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--salt", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Post-key audit cannot overwrite an existing report")
    report = verify(args.candidate_dir, args.specs, args.salt, args.builder)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output.chmod(0o600)
    print(
        json.dumps(
            {
                "status": report["status"],
                "originals": report["originals"],
                "source_deletions": report["source_deletions"],
                "seal_order_verified": report["seal_order_verified"],
                "report_sha256": sha(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
