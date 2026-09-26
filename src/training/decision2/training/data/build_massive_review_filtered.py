"""Filter a frozen MASSIVE candidate through complete independent semantic verdicts.

This produces another *unapproved* candidate. It cannot turn review decisions
into training authorization, and never refills rejected source IDs with
unreviewed rows.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from training.data import build_massive_expanded_review as expanded
from training.data import build_massive_multilingual as massive
from training.model.data import check_partition_isolation

MANDATORY_EXCLUSIONS = frozenset(
    {
        "1986",
        "7248",
        "7148",
        "14736",
        "49",
        "4181",
        "6653",
        "4879",
        "11042",
        "12431",
        "2580",
        "11439",
        "11828",
        "11772",
        "15335",
        "13289",
        "12199",
    }
)
INITIAL_BLIND_SHA = "62f2aad82fa61c0de432de09d09ae56e48d4d10ec860cdaf9824616a849b30b8"
INITIAL_POST_KEY_SHA = (
    "78f5d2529587b4ae6a39e692bb8fafc81f21f7356f4213c22a008efcd4b55056"
)
EXPANDED_BLIND_SHA = "57ad5324fdac94a5427fd48d74f42a87d6630557ffb3394395838ecdab6c5535"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def verified(path: Path, digest: str) -> None:
    if massive.sha(path) != digest:
        raise ValueError(f"Frozen input digest changed: {path.name}")


def load_verdicts(path: Path, digest: str, expected_ids: set[str]) -> dict[str, dict]:
    verified(path, digest)
    rows = load_jsonl(path)
    if len(rows) != len(expected_ids):
        raise ValueError("Independent English verdict must cover every source group")
    verdicts = {}
    for row in rows:
        identifier = row.get("review_id")
        if (
            not isinstance(identifier, str)
            or identifier in verdicts
            or not isinstance(row.get("verdict"), str)
            or not row["verdict"]
            or not isinstance(row.get("reason"), str)
            or (row["verdict"] != "pass" and not row["reason"].strip())
        ):
            raise ValueError("Malformed or duplicate independent English verdict")
        verdicts[identifier] = row
    if set(verdicts) != expected_ids:
        raise ValueError("Independent English verdict IDs do not match candidate")
    return verdicts


def build(
    candidate: Path,
    manifest_sha: str,
    expanded_review: Path,
    expanded_receipt_sha: str,
    english_verdicts: Path,
    english_verdict_sha: str,
    source_exclusion: Path,
    source_exclusion_sha: str,
    review_manifest: Path,
    review_manifest_sha: str,
    source_directory: Path,
    output: Path,
) -> dict:
    stage = output.with_name(output.name + ".pending")
    if output.exists() or output.is_symlink() or stage.exists() or stage.is_symlink():
        raise FileExistsError(output)
    verified(candidate / "manifest.json", manifest_sha)
    manifest = json.loads((candidate / "manifest.json").read_text())
    if manifest.get("training_approved") is not False or manifest.get(
        "selected_source_groups"
    ) != {"train": 600, "dev": 200}:
        raise ValueError("Expected the exact unapproved MASSIVE v1 candidate")
    for name in ("train.private.jsonl", "dev.private.jsonl"):
        verified(candidate / name, manifest["outputs"][name]["sha256"])
    verified(expanded_review / "receipt.json", expanded_receipt_sha)
    receipt = json.loads((expanded_review / "receipt.json").read_text())
    if (
        receipt.get("candidate_manifest_sha256") != manifest_sha
        or receipt.get("candidate_train_sha256")
        != manifest["outputs"]["train.private.jsonl"]["sha256"]
        or receipt.get("blind_review_rows") != 600
    ):
        raise ValueError("Expanded blind review is not bound to this candidate")
    for name, digest in receipt["files"].items():
        verified(expanded_review / name, digest)
    key = load_jsonl(expanded_review / "english-all600.key.private.jsonl")
    source_by_review = {row["review_id"]: row["source_id"] for row in key}
    if (
        len(key) != 600
        or len(source_by_review) != 600
        or any(
            expanded.review_id(source_id) != review_id
            for review_id, source_id in source_by_review.items()
        )
    ):
        raise ValueError("Expanded English answer key changed")
    verdicts = load_verdicts(
        english_verdicts, english_verdict_sha, set(source_by_review)
    )
    verified(source_exclusion, source_exclusion_sha)
    verified(review_manifest, review_manifest_sha)
    exclusion_report = json.loads(source_exclusion.read_text())
    review_report = json.loads(review_manifest.read_text())
    if (
        review_report.get("training_approved") is not False
        or review_report.get("blind_review_sha256") != EXPANDED_BLIND_SHA
        or review_report.get("packet_sha256")
        != receipt["files"]["english-all600.blind.private.jsonl"]
        or review_report.get("key_sha256")
        != receipt["files"]["english-all600.key.private.jsonl"]
        or review_report.get("verdict_sha256") != english_verdict_sha
        or review_report.get("exclusion_sha256") != source_exclusion_sha
        or exclusion_report.get("blind_review_sha256") != EXPANDED_BLIND_SHA
        or exclusion_report.get("key_sha256")
        != receipt["files"]["english-all600.key.private.jsonl"]
        or exclusion_report.get("force_exclude_prior_17_count")
        != len(MANDATORY_EXCLUSIONS)
    ):
        raise ValueError("Independent English review receipts are inconsistent")
    groups = defaultdict(list)
    for row in load_jsonl(candidate / "train.private.jsonl"):
        groups[row["audit_metadata"]["source_id"]].append(row)
    if (
        set(groups) != set(source_by_review.values())
        or any(len(rows) != 7 for rows in groups.values())
        or sum(map(len, groups.values())) != 4200
    ):
        raise ValueError("Candidate source group lineage changed")
    if not set(groups) >= MANDATORY_EXCLUSIONS:
        raise ValueError("Mandatory independent-review exclusions are missing")
    excluded = {}
    for review_id, source_id in source_by_review.items():
        verdict = verdicts[review_id]["verdict"]
        if source_id in MANDATORY_EXCLUSIONS:
            excluded[source_id] = "initial_independent_review"
        elif verdict != "pass":
            excluded[source_id] = verdict
    if (
        set(exclusion_report.get("excluded_source_ids", [])) != set(excluded)
        or set(exclusion_report.get("reason_by_source_id", {})) != set(excluded)
        or exclusion_report.get("union_excluded_count") != len(excluded)
        or review_report.get("excluded_source_groups") != len(excluded)
        or review_report.get("counts")
        != dict(Counter(row["verdict"] for row in verdicts.values()))
    ):
        raise ValueError("Independent exclusion roster disagrees with verdicts")
    kept_ids = set(groups) - set(excluded)
    train = [
        row
        for row in load_jsonl(candidate / "train.private.jsonl")
        if row["audit_metadata"]["source_id"] in kept_ids
    ]
    dev = load_jsonl(candidate / "dev.private.jsonl")
    if len(dev) != 1400:
        raise ValueError("Candidate DEV row inventory changed")
    check_partition_isolation({"train": train, "select": dev})
    if len(train) != 7 * len(kept_ids):
        raise ValueError("Whole seven-locale groups were not retained")
    for name, expected in (
        ("LICENSE", massive.LICENSE_SHA),
        ("NOTICE.md", massive.NOTICE_SHA),
    ):
        verified(source_directory / name, expected)
    stage.mkdir(parents=True, mode=0o700)
    massive.write_jsonl(stage / "train.private.jsonl", train)
    shutil.copy2(candidate / "dev.private.jsonl", stage / "dev.private.jsonl")
    shutil.copy2(english_verdicts, stage / "english-verdicts.private.jsonl")
    shutil.copy2(source_exclusion, stage / "source-exclusion.private.json")
    shutil.copy2(review_manifest, stage / "independent-review.private.json")
    for name in ("LICENSE", "NOTICE.md"):
        shutil.copy2(source_directory / name, stage / name)
    source_intents = {
        source_id: rows[0]["audit_metadata"]["intent"]
        for source_id, rows in groups.items()
    }
    kept_by_intent = dict(
        sorted(Counter(source_intents[source_id] for source_id in kept_ids).items())
    )
    if (
        exclusion_report.get("retained_source_groups") != len(kept_ids)
        or review_report.get("retained_source_groups") != len(kept_ids)
        or exclusion_report.get("retained_per_intent") != kept_by_intent
        or exclusion_report.get("retained_intent_coverage") != len(kept_by_intent)
        or review_report.get("retained_intent_coverage") != len(kept_by_intent)
    ):
        raise ValueError("Independent retained-group counts disagree with candidate")
    result = {
        "schema_version": "decision2-massive-review-filtered/1",
        "training_approved": False,
        "research_only": True,
        "no_public_raw_text": True,
        "source_candidate_manifest_sha256": manifest_sha,
        "source_candidate_train_sha256": manifest["outputs"]["train.private.jsonl"][
            "sha256"
        ],
        "source_candidate_dev_sha256": manifest["outputs"]["dev.private.jsonl"][
            "sha256"
        ],
        "expanded_review_receipt_sha256": expanded_receipt_sha,
        "independent_english_verdict_sha256": english_verdict_sha,
        "independent_source_exclusion_sha256": source_exclusion_sha,
        "independent_review_manifest_sha256": review_manifest_sha,
        "expanded_blind_review_sha256": EXPANDED_BLIND_SHA,
        "initial_review": {
            "blind_sha256": INITIAL_BLIND_SHA,
            "post_key_sha256": INITIAL_POST_KEY_SHA,
            "mandatory_excluded_source_groups": len(MANDATORY_EXCLUSIONS),
        },
        "verdict_counts": dict(
            sorted(Counter(row["verdict"] for row in verdicts.values()).items())
        ),
        "source_groups": {
            "initial_train": 600,
            "kept_train": len(kept_ids),
            "excluded_train": len(excluded),
            "dev": 200,
        },
        "rows": {"train": len(train), "dev": len(dev)},
        "kept_intent_source_groups": kept_by_intent,
        "train_intent_coverage": len(kept_by_intent),
        "dev_intent_coverage": manifest["intent_coverage"]["dev"],
        "excluded_source_ids": dict(
            sorted(excluded.items(), key=lambda item: int(item[0]))
        ),
        "cross_split": {
            "exact_context_rows": 0,
            "near_context_rows": 0,
            "basis": "Inherited v1 zero-overlap audit; removing TRAIN groups cannot introduce an overlap",
        },
        "rights": {
            "license": "CC-BY-4.0",
            "attribution": manifest["source"]["attribution"],
            "original_license_sha256": massive.LICENSE_SHA,
            "original_notice_sha256": massive.NOTICE_SHA,
        },
        "cross_locale_semantic_review_pending": True,
        "independent_v2_review_pending": True,
        "outputs": {
            name: massive.sha(stage / name)
            for name in (
                "train.private.jsonl",
                "dev.private.jsonl",
                "english-verdicts.private.jsonl",
                "source-exclusion.private.json",
                "independent-review.private.json",
                "LICENSE",
                "NOTICE.md",
            )
        },
        "limitations": [
            "Never refilled rejected groups with unreviewed examples.",
            "English review is insufficient to certify all six translations.",
            "The original quality-filtered DEV cannot cover all 60 intents.",
            "No model training is approved by this manifest.",
        ],
    }
    (stage / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    for path in stage.iterdir():
        os.chmod(path, 0o600)
    os.replace(stage, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--expanded-review", type=Path, required=True)
    parser.add_argument("--expanded-receipt-sha", required=True)
    parser.add_argument("--english-verdicts", type=Path, required=True)
    parser.add_argument("--english-verdict-sha", required=True)
    parser.add_argument("--source-exclusion", type=Path, required=True)
    parser.add_argument("--source-exclusion-sha", required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--review-manifest-sha", required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        args.candidate,
        args.manifest_sha,
        args.expanded_review,
        args.expanded_receipt_sha,
        args.english_verdicts,
        args.english_verdict_sha,
        args.source_exclusion,
        args.source_exclusion_sha,
        args.review_manifest,
        args.review_manifest_sha,
        args.source_directory,
        args.output,
    )
    print(
        json.dumps(
            {
                "source_groups": result["source_groups"],
                "training_approved": result["training_approved"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
