"""Post-key gate creation must wait for six independent reviewed receipts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import prepare_postkey_review_v3 as review


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class PostkeyReviewTest(unittest.TestCase):
    def test_finalizer_rejects_missing_or_blocked_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = {
                "schema_version": review.CONTEXT_VERSION,
                "status": "awaiting_independent_reviews",
                "model_id": "llm-semantic-router/DEV2.0-4B",
                "model_revision": "r1",
                "release_context_sha256": "a" * 64,
                "amendment_sha256": review.postkey.bundle.common.sha_file(
                    review.postkey.AMENDMENT
                ),
                "required_independent_checks": list(review.postkey.bundle.GATE_CHECKS),
                "gate_bindings": {"package_record_sha256": "b" * 64},
                "rank_diagnostic_sha256": "c" * 64,
                "original_rank_failure_log_sha256": "d" * 64,
                "original_strict_hold_sha256": "e" * 64,
                "original_protocol_sha256": "f" * 64,
                "comparison_inputs_sha256": {"aggregate_pair_report_sha256": "1" * 64},
            }
            context_path = root / "context.json"
            write(context_path, context)
            with self.assertRaisesRegex(ValueError, "Review context"):
                review.finalize(context_path, {})
            paths = {}
            for name in review.postkey.bundle.GATE_CHECKS:
                path = root / f"{name}.json"
                receipt = {
                    "schema_version": (
                        review.postkey.bundle.FREEZE_AUDIT_VERSION
                        if name == "candidate_freeze"
                        else review.postkey.bundle.CHECK_VERSION
                    ),
                    "status": "passed",
                    "check": name,
                    "model_id": context["model_id"],
                    "model_revision": context["model_revision"],
                    "release_context_sha256": context["release_context_sha256"],
                    "reviewer_identity_sha256": "2" * 64,
                    "source_evidence_sha256": "3" * 64,
                    "reviewed_at_utc": "2026-09-27T00:00:00+00:00",
                }
                if name == "release_thresholds":
                    receipt.update(
                        {
                            "policy_phase": "postkey_user_directed",
                            "amendment_sha256": context["amendment_sha256"],
                            "rank_diagnostic_sha256": context["rank_diagnostic_sha256"],
                            "original_rank_failure_log_sha256": context[
                                "original_rank_failure_log_sha256"
                            ],
                            "original_strict_hold_sha256": context[
                                "original_strict_hold_sha256"
                            ],
                            "predeclared_policy_sha256": context[
                                "original_protocol_sha256"
                            ],
                            "comparison_inputs_sha256": context[
                                "comparison_inputs_sha256"
                            ],
                        }
                    )
                write(path, receipt)
                paths[name] = path
            result = review.finalize(context_path, paths)
            self.assertEqual(result["status"], review.postkey.GATE_STATUS)
            self.assertEqual(set(result["checks"]), set(paths))
            receipt = json.loads(paths["native_parity"].read_text())
            receipt["status"] = "blocked"
            write(paths["native_parity"], receipt)
            with self.assertRaisesRegex(ValueError, "incomplete: native_parity"):
                review.finalize(context_path, paths)


if __name__ == "__main__":
    unittest.main()
