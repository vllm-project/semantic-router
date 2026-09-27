"""Gold-free contract tests for the prospective 27B release lock."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import preflight_27b_v3 as preflight


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FirstRelease27BTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.private = self.root / "private"
        self.private.mkdir(mode=0o700)
        os.chmod(self.private, 0o700)
        self.package = self.root / "package"
        self.package.mkdir()
        self.source = self.root / "official-source"
        self.source.mkdir()
        self.manifest = {
            "model_id": preflight.MODEL_ID,
            "base": {"repo_id": preflight.BASE_ID, "revision": preflight.BASE_REVISION},
            "model_sha256": preflight.BEST368_SHA256,
            "scored_prediction_manifest_sha256": preflight.SCORED_MANIFEST_SHA256,
            "calibration_sha256": preflight.CAL_SHA256,
            "parameter_count": preflight.LOADED_PARAMETERS,
            "publication_status": "candidate-parity-pending",
            "loader_files_sha256": {"infer.py": "a" * 64},
        }
        write_json(self.package / "MODEL_MANIFEST.json", self.manifest)
        self.parity = self.private / "native-parity.json"
        write_json(self.parity, {"status": "passed"})
        self.roster = self.private / "roster.json"
        self.prompts = {
            name: self.private / f"{name}.prompts.jsonl" for name in preflight.PANELS
        }
        for name, path in self.prompts.items():
            row = {"id": name, "state": "case", "questions": {"q": {"type": "noul"}}}
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        write_json(
            self.roster,
            {
                "panel_prompt_sha256": {
                    name: digest(path) for name, path in self.prompts.items()
                }
            },
        )
        self.prediction_dir = self.private / "predictions"
        self.lock = self.private / "candidate-lock.json"
        self.identity = {
            "model_sha256": preflight.BEST368_SHA256,
            "files_sha256": {"checkpoint/a": "a" * 64},
        }

    def lock_candidate(self) -> dict:
        with (
            patch.object(preflight, "verify_bundle", return_value=self.manifest),
            patch.object(preflight, "_verify_staged_runtime"),
            patch.object(
                preflight, "checkpoint_fingerprint", return_value=self.identity
            ),
            patch.object(preflight, "_parity"),
            patch.object(
                preflight,
                "_comparators",
                return_value={"baseline_keys": ["lux", "jevk5-9b"]},
            ),
            patch.object(
                preflight,
                "_prompts",
                return_value={
                    name: {"path": str(path), "sha256": digest(path), "items": count}
                    for name, (path, count) in zip(
                        self.prompts,
                        zip(self.prompts.values(), preflight.PANELS.values()),
                    )
                },
            ),
        ):
            return preflight.candidate_lock(
                package=self.package,
                source=self.source,
                expected_package_sha256=digest(self.package / "MODEL_MANIFEST.json"),
                parity_receipt=self.parity,
                parity_sha256=digest(self.parity),
                roster_path=self.roster,
                prompt_paths=self.prompts,
                prediction_dir=self.prediction_dir,
                source_root=Path(preflight.__file__).resolve().parents[1],
                model_root=self.root,
                external_root=self.root,
                output=self.lock,
            )

    def test_lock_binds_uppercase_package_and_exposed_panel(self) -> None:
        record = self.lock_candidate()
        self.assertEqual(record["candidate"]["model_id"], preflight.MODEL_ID)
        self.assertEqual(record["scoring"]["formula"], "100*sqrt(T*H)")
        self.assertIn("not_virgin_blind", record["label_exposure"]["evidence_status"])
        self.assertEqual(self.lock.stat().st_mode & 0o777, 0o600)

    def test_old_model_id_and_changed_prompt_fail_before_lock(self) -> None:
        self.manifest["model_id"] = "llm-semantic-router/dev-2.0-27b"
        write_json(self.package / "MODEL_MANIFEST.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "lineage"):
            self.lock_candidate()
        self.assertFalse(self.lock.exists())

        self.manifest["model_id"] = preflight.MODEL_ID
        write_json(self.package / "MODEL_MANIFEST.json", self.manifest)
        declared = json.loads(self.roster.read_text())
        declared["panel_prompt_sha256"]["css"] = "b" * 64
        write_json(self.roster, declared)
        with self.assertRaisesRegex(ValueError, "prompt bytes"):
            self.lock_candidate()
        self.assertFalse(self.lock.exists())

    def test_seal_rejects_mutated_roster_before_predictions(self) -> None:
        self.lock_candidate()
        self.roster.write_text(self.roster.read_text() + "\n", encoding="utf-8")
        with patch.object(preflight, "_verify_staged_runtime"):
            with self.assertRaisesRegex(ValueError, "roster changed"):
                preflight.prediction_seal(
                    lock_path=self.lock,
                    lock_sha256=digest(self.lock),
                    output=self.private / "prediction-seal.json",
                    source_root=Path(preflight.__file__).resolve().parents[1],
                )

    def test_comparator_roster_cannot_be_swapped_after_policy(self) -> None:
        roster = {"baseline_keys": ["lux", "kev"]}
        with self.assertRaisesRegex(ValueError, "roster must pin"):
            preflight._comparators(
                roster,
                Path(preflight.__file__).resolve().parents[1],
                self.root,
                self.root,
                preflight.LOADED_PARAMETERS / 1_000_000_000,
            )

    def test_same_size_control_must_qualify_before_candidate_lock(self) -> None:
        roster = {
            "baseline_keys": list(preflight.BASELINE_KEYS),
            "jebadiah27b": {
                "model_id": "frontier-infra/jebadiah-27b",
                "status": "hold_unqualified",
                "reason": "Exact revision and native three-type runtime are not yet attested",
            },
            "decision1_pair": {
                "comparator": "lux",
                "size_relation": "nearest",
                "rationale": "Lux is our largest Decision 1.0 model and has fewer parameters",
            },
            "open_control": {
                "key": "jevk5-9b",
                "size_relation": "nearest",
                "rationale": "JevK5 9B has fewer loaded parameters than this 27B candidate",
            },
            "same_size_control": {
                "key": "autojev27b",
                "size_relation": "same",
                "rationale": "Native 27B opponent",
            },
            "baseline_repeatability": [],
        }
        autojev = {
            "native_model_sha256": "a" * 64,
            "runtime_source_sha256": "b" * 64,
            "model_config_sha256": "c" * 64,
            "loaded_parameters": 25_000_000_000,
        }
        attestations = {
            "lux": {
                "size_b": 9.6,
                "receipt_sha256": "d" * 64,
                "native_model_sha256": "d" * 64,
                "adapter_sha256": "d" * 64,
                "calibration_sha256": None,
            },
            "jevk5-9b": {
                "size_b": 9.2,
                "receipt_sha256": "e" * 64,
                "native_model_sha256": "e" * 64,
                "adapter_sha256": "e" * 64,
                "calibration_sha256": None,
            },
            "autojev27b": {
                "size_b": 25.0,
                "receipt_sha256": "f" * 64,
                "native_model_sha256": "a" * 64,
                "adapter_sha256": "f" * 64,
                "calibration_sha256": None,
            },
        }
        repeats = {key: {"receipt_sha256": "0" * 64} for key in attestations}
        with (
            patch.object(preflight, "verify_autojev_release", return_value=autojev),
            patch.object(
                preflight, "_baseline_attestations", return_value=attestations
            ),
            patch.object(preflight, "_baseline_repeatability", return_value=repeats),
        ):
            result = preflight._comparators(
                roster,
                Path(preflight.__file__).resolve().parents[1],
                self.root,
                self.root,
                preflight.LOADED_PARAMETERS / 1_000_000_000,
            )
            self.assertEqual(result["same_size_control"]["key"], "autojev27b")
            roster["jebadiah27b"]["status"] = "qualified"
            with self.assertRaisesRegex(ValueError, "qualification decision"):
                preflight._comparators(
                    roster,
                    Path(preflight.__file__).resolve().parents[1],
                    self.root,
                    self.root,
                    preflight.LOADED_PARAMETERS / 1_000_000_000,
                )


if __name__ == "__main__":
    unittest.main()
