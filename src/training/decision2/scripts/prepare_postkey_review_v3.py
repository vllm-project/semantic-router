"""Prepare exact v3 review bindings without fabricating reviewer decisions.

The context is a private, unsigned work item. Six independent reviewers must
produce separate receipts before `--finalize` can create an amended gate.
The final package verifier rechecks the receipts and their original evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from scripts import package_postkey_aggregate_v3 as postkey
from scripts.create_strict_hold_receipt_v3 import receipt as strict_receipt

CONTEXT_VERSION = "decision2-v3-postkey-review-context/1"


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(payload)


def prepare(
    config_path: Path,
    diagnostic_path: Path,
    rank_failure_path: Path,
    strict_hold_path: Path,
) -> dict[str, Any]:
    common, bundle = postkey.bundle.common, postkey.bundle
    config = common._object(config_path)
    paths = postkey._config_paths(config_path, config)
    record = common._object(paths["package_record"])
    model_id, revision = record["model_id"], record["model_revision"]
    sidecar = common._object(diagnostic_path)
    strict = common._object(strict_hold_path)
    if (
        config.get("postkey_policy") != "aggregate_priority_delta_3"
        or sidecar.get("schema_version")
        != "jevarena-v3-postkey-answer-count-diagnostic/1"
        or sidecar.get("ranked_result") != common._object(paths["arena_rank"])
        or sidecar.get("prekey_freeze_sha256")
        != common.sha_file(paths["freeze_manifest"])
        or sidecar.get("original_failed_rank_log_sha256")
        != common.sha_file(rank_failure_path)
        or strict.get("schema_version") != "decision2-v3-strict-release-hold/1"
        or strict.get("status") != "HOLD"
        or strict.get("candidate_model_id") != model_id
        or strict.get("prekey_freeze_sha256")
        != common.sha_file(paths["freeze_manifest"])
        or strict
        != strict_receipt(
            paths["freeze_manifest"],
            paths["score_inputs"]["typed"]["score"],
            paths["old_typed_report"],
        )
    ):
        raise ValueError("Post-key evidence does not bind the chosen frozen run")
    scores = {}
    for family in bundle.SCORE_FAMILIES:
        inputs = paths["score_inputs"][family]
        native = common._object(inputs["native_manifest"])
        scores[family] = {
            "score_sha256": common.sha_file(inputs["score"]),
            "predictions_sha256": common.sha_file(inputs["predictions"]),
            "native_manifest_sha256": common.sha_file(inputs["native_manifest"]),
            "adapter_sha256": bundle._native_adapter_sha(native, family),
        }
    comparisons = {
        "aggregate_pair_report_sha256": common.sha_file(paths["aggregate_pair_report"]),
        "old_typed_report_sha256": common.sha_file(paths["old_typed_report"]),
        "old_css_report_sha256": common.sha_file(paths["old_css_report"]),
    }
    record_files = record["model_files_sha256"]
    files_sha = hashlib.sha256(
        json.dumps(record_files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    bindings = {
        "package_record_sha256": common.sha_file(paths["package_record"]),
        "parity_receipt_sha256": common.sha_file(paths["parity_receipt"]),
        "artifact_manifest_sha256": common.sha_file(
            paths["artifacts"] / "manifest.json"
        ),
        "arena_rank_sha256": common.sha_file(paths["arena_rank"]),
        "jevbench_public_rank_sha256": common.sha_file(paths["public_rank"]),
        "pretest_freeze_sha256": common.sha_file(paths["freeze_manifest"]),
        "native_model_sha256": record["native_identity"]["sha256"],
        "model_files_sha256": files_sha,
    }
    context_sha = bundle._context_digest(
        bindings["package_record_sha256"],
        bindings["parity_receipt_sha256"],
        bindings["artifact_manifest_sha256"],
        bindings["arena_rank_sha256"],
        bindings["jevbench_public_rank_sha256"],
        bindings["pretest_freeze_sha256"],
        scores,
        comparisons,
    )
    freeze = common._object(paths["freeze_manifest"])
    return {
        "schema_version": CONTEXT_VERSION,
        "status": "awaiting_independent_reviews",
        "model_id": model_id,
        "model_revision": revision,
        "release_context_sha256": context_sha,
        "gate_bindings": bindings,
        "score_inputs_sha256": scores,
        "comparison_inputs_sha256": comparisons,
        "original_protocol_sha256": freeze["protocol_sha256"],
        "amendment_sha256": common.sha_file(postkey.AMENDMENT),
        "rank_diagnostic_sha256": common.sha_file(diagnostic_path),
        "original_rank_failure_log_sha256": common.sha_file(rank_failure_path),
        "original_strict_hold_sha256": common.sha_file(strict_hold_path),
        "required_independent_checks": list(bundle.GATE_CHECKS),
    }


def finalize(context_path: Path, reviews: dict[str, Path]) -> dict[str, Any]:
    common, bundle = postkey.bundle.common, postkey.bundle
    context = common._object(context_path)
    if (
        context.get("schema_version") != CONTEXT_VERSION
        or context.get("status") != "awaiting_independent_reviews"
        or context.get("amendment_sha256") != common.sha_file(postkey.AMENDMENT)
        or context.get("required_independent_checks") != list(bundle.GATE_CHECKS)
        or set(reviews) != set(bundle.GATE_CHECKS)
    ):
        raise ValueError("Review context or independent check roster differs")
    checks = {}
    for name in bundle.GATE_CHECKS:
        path = reviews[name]
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing independent reviewer receipt: {name}")
        receipt = common._object(path)
        required = {
            "schema_version": (
                bundle.FREEZE_AUDIT_VERSION
                if name == "candidate_freeze"
                else bundle.CHECK_VERSION
            ),
            "status": "passed",
            "check": name,
            "model_id": context["model_id"],
            "model_revision": context["model_revision"],
            "release_context_sha256": context["release_context_sha256"],
        }
        if any(receipt.get(field) != value for field, value in required.items()):
            raise ValueError(f"Independent review is incomplete: {name}")
        common._sha(receipt.get("reviewer_identity_sha256"), f"{name} reviewer")
        common._sha(receipt.get("source_evidence_sha256"), f"{name} source")
        bundle._utc(receipt.get("reviewed_at_utc"), f"{name} review time")
        if name == "release_thresholds" and any(
            receipt.get(field) != value
            for field, value in (
                ("policy_phase", "postkey_user_directed"),
                ("amendment_sha256", context["amendment_sha256"]),
                ("rank_diagnostic_sha256", context["rank_diagnostic_sha256"]),
                (
                    "original_rank_failure_log_sha256",
                    context["original_rank_failure_log_sha256"],
                ),
                ("original_strict_hold_sha256", context["original_strict_hold_sha256"]),
                ("predeclared_policy_sha256", context["original_protocol_sha256"]),
                ("comparison_inputs_sha256", context["comparison_inputs_sha256"]),
            )
        ):
            raise ValueError("Threshold review lacks the explicit post-key decision")
        checks[name] = {"status": "passed", "evidence_sha256": common.sha_file(path)}
    return {
        "schema_version": postkey.GATE_VERSION,
        "status": postkey.GATE_STATUS,
        "model_id": context["model_id"],
        "model_revision": context["model_revision"],
        **context["gate_bindings"],
        "amendment_sha256": context["amendment_sha256"],
        "minimum_aggregate_delta": 3.0,
        "typed_answer_denominator": 2000,
        "original_strict_gate": "HOLD",
        "rank_diagnostic_sha256": context["rank_diagnostic_sha256"],
        "original_rank_failure_log_sha256": context["original_rank_failure_log_sha256"],
        "original_strict_hold_sha256": context["original_strict_hold_sha256"],
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--postkey-rank-diagnostic", type=Path)
    parser.add_argument("--original-rank-failure-log", type=Path)
    parser.add_argument("--original-strict-hold", type=Path)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--review-paths", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.context is not None:
        if args.review_paths is None:
            parser.error("--context requires --review-paths mapping")
        mapping = postkey.bundle.common._object(args.review_paths)
        result = finalize(
            args.context,
            {
                name: postkey._resolve(args.review_paths, value)
                for name, value in mapping.items()
            },
        )
    else:
        if any(
            value is None
            for value in (
                args.config,
                args.postkey_rank_diagnostic,
                args.original_rank_failure_log,
                args.original_strict_hold,
            )
        ):
            parser.error("prepare requires config, diagnostic, original log and HOLD")
        result = prepare(
            args.config,
            args.postkey_rank_diagnostic,
            args.original_rank_failure_log,
            args.original_strict_hold,
        )
    _write_exclusive(args.output, result)
    print(
        json.dumps(
            {"status": result["status"], "schema_version": result["schema_version"]}
        )
    )


if __name__ == "__main__":
    main()
