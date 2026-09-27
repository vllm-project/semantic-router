"""The amended package requires distinct old HOLD and new review evidence."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import package_postkey_aggregate_v3 as postkey


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class PostkeyPackageTest(unittest.TestCase):
    def test_card_discloses_exact_original_score_tradeoff(self) -> None:
        frozen = SimpleNamespace(
            __file__=str(postkey.AMENDMENT),
            _freeze=lambda *_args: None,
            SCORER_SOURCE_PATHS={},
        )
        strict = {
            "candidate_score_accuracy_all": 0.4175,
            "comparator_score_accuracy_all": 0.445,
        }
        with patch.object(postkey.bundle, "_card", return_value="# model\nbody"):
            with postkey._amended_bundle(
                frozen, {"amendment_sha256": "a" * 64}, strict
            ):
                card = postkey.bundle._card("model")
        self.assertIn(postkey.CARD_MARKER, card)
        self.assertIn("44.50% to 41.75% (-2.75 percentage points)", card)

    def test_loopback_exception_is_limited_to_native_serve_file(self) -> None:
        original = postkey.bundle.common._public_text
        frozen = SimpleNamespace(
            __file__=str(postkey.AMENDMENT),
            _freeze=lambda *_args: None,
            SCORER_SOURCE_PATHS={},
        )
        with postkey._amended_bundle(frozen, {"amendment_sha256": "a" * 64}):
            postkey.bundle.common._public_text(
                'DEFAULT_URL = "http://127.0.0.1:8000"', "serve.py"
            )
            for content, label in (
                ('DEFAULT_URL = "http://10.1.2.3:8000"', "serve.py"),
                ('DEFAULT_URL = "http://127.0.0.1:8000"', "README.md"),
                ('API_KEY = "hf_' + "a" * 30 + '"', "serve.py"),
            ):
                with self.assertRaisesRegex(ValueError, "private infrastructure"):
                    postkey.bundle.common._public_text(content, label)
        self.assertIs(postkey.bundle.common._public_text, original)

    def test_shell_inventory_exception_keeps_full_screening(self) -> None:
        frozen = SimpleNamespace(
            __file__=str(postkey.AMENDMENT),
            _freeze=lambda *_args: None,
            SCORER_SOURCE_PATHS={},
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.safetensors").write_bytes(b"test-weight")
            (root / "serve_vllm.sh").write_text(
                "#!/bin/sh\ncurl http://127.0.0.1:8000\n"
            )
            with postkey._amended_bundle(frozen, {"amendment_sha256": "a" * 64}):
                files = postkey.bundle.common._inventory(root)
                self.assertIn("serve_vllm.sh", files)
                (root / "unexpected.sh").write_text("#!/bin/sh\nexit 0\n")
                with self.assertRaisesRegex(ValueError, "reviewed native"):
                    postkey.bundle.common._inventory(root)
                (root / "unexpected.sh").unlink()
                (root / "serve_vllm.sh").write_text(
                    "#!/bin/sh\nexport API_KEY=hf_" + "a" * 30 + "\n"
                )
                with self.assertRaisesRegex(ValueError, "private infrastructure"):
                    postkey.bundle.common._inventory(root)

    def test_gate_is_distinct_from_strict_prekey_pass(self) -> None:
        required = {
            name: name[0] * 64
            for name in (
                "package_record_sha256",
                "parity_receipt_sha256",
                "artifact_manifest_sha256",
                "arena_rank_sha256",
                "jevbench_public_rank_sha256",
                "pretest_freeze_sha256",
                "native_model_sha256",
                "model_files_sha256",
            )
        }
        gate = {
            "schema_version": postkey.GATE_VERSION,
            "status": postkey.GATE_STATUS,
            "model_id": "org/DEV2.0-4B",
            "model_revision": "fixed",
            "amendment_sha256": postkey.bundle.common.sha_file(postkey.AMENDMENT),
            "minimum_aggregate_delta": 3.0,
            "typed_answer_denominator": 2000,
            "original_strict_gate": "HOLD",
            "original_rank_failure_log_sha256": "a" * 64,
            "original_strict_hold_sha256": "b" * 64,
            "rank_diagnostic_sha256": "c" * 64,
            "checks": {
                name: {"status": "passed", "evidence_sha256": "d" * 64}
                for name in postkey.bundle.GATE_CHECKS
            },
            **required,
        }
        arguments = {
            "record_sha": required["package_record_sha256"],
            "parity_sha": required["parity_receipt_sha256"],
            "artifact_sha": required["artifact_manifest_sha256"],
            "arena_sha": required["arena_rank_sha256"],
            "public_sha": required["jevbench_public_rank_sha256"],
            "freeze_sha": required["pretest_freeze_sha256"],
            "native_sha": required["native_model_sha256"],
            "model_files_sha": required["model_files_sha256"],
            "model_id": gate["model_id"],
            "revision": gate["model_revision"],
        }
        postkey._gate(gate, **arguments)
        gate["status"] = "passed"
        with self.assertRaisesRegex(ValueError, "Post-key gate"):
            postkey._gate(gate, **arguments)

    def test_original_strict_hold_receipt_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rank = {"models": [{"key": "new"}]}
            write(root / "rank.json", rank)
            write(root / "freeze.json", {"status": "prekey_frozen"})
            (root / "failed.log").write_text("count defect\n", encoding="utf-8")
            write(
                root / "new.json",
                {
                    "model": {"id": "org/new"},
                    "by_type": {"score": {"accuracy_all": 0.4175}},
                },
            )
            write(
                root / "old.json",
                {
                    "model": {"id": "org/old"},
                    "by_type": {"score": {"accuracy_all": 0.445}},
                },
            )
            diagnostic = {
                "schema_version": "jevarena-v3-postkey-answer-count-diagnostic/1",
                "status": "postkey_diagnostic_not_preregistered_release_gate",
                "ranked_result": rank,
                "prekey_freeze_sha256": postkey.bundle.common.sha_file(
                    root / "freeze.json"
                ),
                "original_failed_rank_log_sha256": postkey.bundle.common.sha_file(
                    root / "failed.log"
                ),
            }
            write(root / "diagnostic.json", diagnostic)
            strict = {
                "schema_version": "decision2-v3-strict-release-hold/1",
                "status": "HOLD",
                "reason": "typed_score_slice_regression",
                "prekey_freeze_sha256": diagnostic["prekey_freeze_sha256"],
                "candidate_model_id": "org/new",
                "comparator_model_id": "org/old",
                "candidate_typed_report_sha256": postkey.bundle.common.sha_file(
                    root / "new.json"
                ),
                "comparator_typed_report_sha256": postkey.bundle.common.sha_file(
                    root / "old.json"
                ),
                "candidate_score_accuracy_all": 0.4175,
                "comparator_score_accuracy_all": 0.445,
                "predeclared_max_absolute_regression": 0.02,
            }
            write(root / "strict.json", strict)
            gate = {
                "rank_diagnostic_sha256": postkey.bundle.common.sha_file(
                    root / "diagnostic.json"
                ),
                "original_rank_failure_log_sha256": postkey.bundle.common.sha_file(
                    root / "failed.log"
                ),
                "original_strict_hold_sha256": postkey.bundle.common.sha_file(
                    root / "strict.json"
                ),
                "amendment_sha256": postkey.bundle.common.sha_file(postkey.AMENDMENT),
            }
            write(root / "gate.json", gate)
            write(
                root / "threshold.json",
                {
                    "amendment_sha256": gate["amendment_sha256"],
                    "rank_diagnostic_sha256": gate["rank_diagnostic_sha256"],
                    "original_rank_failure_log_sha256": gate[
                        "original_rank_failure_log_sha256"
                    ],
                    "original_strict_hold_sha256": gate["original_strict_hold_sha256"],
                    "policy_phase": "postkey_user_directed",
                },
            )
            paths = {
                "release_gate": root / "gate.json",
                "arena_rank": root / "rank.json",
                "freeze_manifest": root / "freeze.json",
                "old_typed_report": root / "old.json",
                "score_inputs": {
                    "typed": {"score": root / "new.json"},
                    "css": {},
                    "public": {},
                },
                "gate_evidence": dict.fromkeys(
                    postkey.bundle.GATE_CHECKS, root / "threshold.json"
                ),
            }
            self.assertEqual(
                postkey._validate_addenda(
                    {"postkey_policy": "aggregate_priority_delta_3"},
                    paths,
                    root / "diagnostic.json",
                    root / "failed.log",
                    root / "strict.json",
                ),
                gate,
            )
            strict["status"] = "passed"
            write(root / "strict.json", strict)
            with self.assertRaisesRegex(ValueError, "Original strict Score HOLD"):
                postkey._validate_addenda(
                    {"postkey_policy": "aggregate_priority_delta_3"},
                    paths,
                    root / "diagnostic.json",
                    root / "failed.log",
                    root / "strict.json",
                )


if __name__ == "__main__":
    unittest.main()
