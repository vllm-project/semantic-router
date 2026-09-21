"""Reject incomplete or relabeled evidence from the real image calibration tool."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import image_calibration
from ci_results import collection_errors

SOURCE = "a" * 40
REVISION = "b" * 40


class ImageCalibrationEvidenceTests(unittest.TestCase):
    def test_file_digest_matches_go_report_wire_format(self):
        path = self.root / "known.txt"
        path.write_bytes(b"abc")
        self.assertEqual(
            image_calibration.file_sha(path),
            "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "reports"
        self.output.mkdir()
        self.model = self.root / "models"
        self.model.mkdir()
        (self.model / "image.onnx").write_text("pinned graph")
        artifact = {
            "format_version": 1,
            "adapter": "vela_omni",
            "source": {"repo_id": "fixture/model", "revision": REVISION},
            "files": {
                "image.onnx": image_calibration.file_sha(
                    self.model / "image.onnx"
                ).removeprefix("sha256:")
            },
        }
        (self.model / image_calibration.OMNI_MANIFEST).write_text(json.dumps(artifact))
        manifest = {
            "provider": "ort",
            "models": [
                {
                    "name": "Multimodal",
                    "env": "MULTIMODAL_MODEL_PATH",
                    "path": str(self.model),
                    "repo_id": "fixture/model",
                    "revision": REVISION,
                }
            ],
        }
        self.write("models.json", manifest)
        _, hashes = image_calibration.model_identity(manifest)
        for filename in ("positive.png", "negative.png", "excluded.png"):
            (self.root / filename).write_text(filename)
        cases = self.root / image_calibration.CASES
        cases.parent.mkdir(parents=True)
        cases.write_text(
            json.dumps(
                {
                    "positives": [
                        {"image_file": "positive.png", "signal_name": "office"}
                    ],
                    "negatives": ["negative.png"],
                    "excluded": [{"image_file": "excluded.png", "reason": "ambiguous"}],
                }
            )
        )
        rules = self.root / image_calibration.RULES
        rules.parent.mkdir(parents=True)
        rules.write_text(
            yaml.safe_dump(
                {"routing": {"signals": {"embeddings": [{"name": "office"}]}}}
            )
        )
        self.report = {
            "model": {
                "repository": "fixture/model",
                "artifact_revision": REVISION,
                "artifact_files": hashes,
            },
            "source": {
                "repo_commit": SOURCE,
                "repo_dirty": False,
                "excluded_fixtures": [
                    {
                        "path": "excluded.png",
                        "sha256": image_calibration.file_sha(
                            self.root / "excluded.png"
                        ),
                        "reason": "ambiguous",
                    }
                ],
            },
            "fixtures": [
                {
                    "path": name,
                    "sha256": image_calibration.file_sha(self.root / name),
                    "positive_for": ["office"] if name == "positive.png" else [],
                    "scores": {"office": 0.4},
                }
                for name in ("positive.png", "negative.png")
            ],
            "rules": [{"name": "office"}],
            "checks": [{"id": "threshold/office", "passed": True}],
        }
        self.write(
            "execution.json",
            {
                "source_sha": SOURCE,
                "commands": {
                    "profile-discovery": 0,
                    "profile-tests": 0,
                    "owned-discovery": 0,
                    "owned-tests": 0,
                    "calibration": 0,
                },
            },
        )
        self.write_profile()
        self.write_owned()
        self.source_patch = patch.object(
            image_calibration.subprocess, "check_output", return_value=SOURCE
        )
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)

    def write(self, name, value):
        (self.output / name).write_text(json.dumps(value))

    def write_profile(self, *, action="pass", omit=False):
        discovered = [
            {"Action": "output", "Package": "profile", "Output": "TestMirror\n"}
        ]
        executed = (
            []
            if omit
            else [{"Action": action, "Package": "profile", "Test": "TestMirror"}]
        )
        for filename, rows in (
            ("profile-discovery.jsonl", discovered),
            ("profile-tests.jsonl", executed),
        ):
            (self.output / filename).write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )

    def evaluate(self):
        self.write("report.json", self.report)
        return image_calibration.evidence(self.output, root=self.root)

    def write_owned(self, *, action="pass", omit=None):
        cases = sorted(image_calibration.owned_omni_case_ids())
        discovery, executed = [], []
        for identity in cases:
            package, name = identity.rsplit("/", 1)
            discovery.append(
                {"Action": "output", "Package": package, "Output": name + "\n"}
            )
            if identity != omit:
                executed.append({"Action": action, "Package": package, "Test": name})
        for filename, rows in (
            ("owned-discovery.jsonl", discovery),
            ("owned-tests.jsonl", executed),
        ):
            (self.output / filename).write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )

    def test_source_inventory_includes_scoring_threshold_and_profile_but_not_exclusions(
        self,
    ):
        result = self.evaluate()
        self.assertEqual(
            set(result["expected_cases"]),
            {
                "score/positive.png",
                "score/negative.png",
                "threshold/office",
                "profile/profile/TestMirror",
            }
            | {"owned/" + name for name in image_calibration.owned_omni_case_ids()},
        )
        self.assertEqual(collection_errors(result, "test"), [])
        self.assertEqual(result["excluded_fixtures"][0]["reason"], "ambiguous")
        # Identical positive and negative scores remain valid observations. The
        # executable Go threshold assertion, not invented perfect accuracy, gates.
        self.assertEqual(result["cases"][0]["scores"], result["cases"][1]["scores"])

    def test_prototype_quality_requires_frozen_inputs_and_zero_holdout_errors(self):
        rules = self.root / image_calibration.RULES
        config = yaml.safe_load(rules.read_text())
        config["routing"]["signals"]["embeddings"][0]["image_candidates"] = [
            "./positive.png"
        ]
        rules.write_text(yaml.safe_dump(config))
        for field, relative in (
            ("prototype_manifest_sha256", "config/assets/image-routing/manifest.json"),
            (
                "prototype_protocol_sha256",
                "tools/calibration/image-routing/testdata/prototype-protocol.json",
            ),
        ):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
            self.report["source"][field] = image_calibration.file_sha(path)
        self.report["rules"][0]["validation"] = {
            "true_positive": 1,
            "false_positive": 0,
            "false_negative": 0,
            "true_negative": 1,
        }
        self.report["checks"].append({"id": "validation/office", "passed": True})
        self.assertIn("validation/office", self.evaluate()["expected_cases"])
        original = copy.deepcopy(self.report)
        for mutation in (
            lambda r: r["rules"][0]["validation"].update(false_positive=1),
            lambda r: r["rules"][0]["validation"].update(false_negative=1),
            lambda r: r["rules"][0]["validation"].update(true_positive=0),
            lambda r: r["rules"][0].pop("validation"),
            lambda r: r["source"].update(prototype_protocol_sha256="changed"),
            lambda r: r["source"].update(prototype_manifest_sha256="changed"),
            lambda r: r["checks"].pop(),
        ):
            self.report = copy.deepcopy(original)
            mutation(self.report)
            with self.assertRaises(ValueError):
                self.evaluate()

    def test_threshold_failure_is_preserved_for_required_gate(self):
        self.report["checks"][0]["passed"] = False
        self.assertTrue(collection_errors(self.evaluate(), "test"))

    def test_missing_duplicate_or_nonfinite_observations_are_rejected(self):
        original = copy.deepcopy(self.report)
        for mutation in (
            lambda r: r["fixtures"].pop(),
            lambda r: r["fixtures"].append(r["fixtures"][0]),
            lambda r: r["checks"].clear(),
            lambda r: r["fixtures"][0]["scores"].clear(),
            lambda r: r["fixtures"][0]["scores"].update(office=float("nan")),
            lambda r: r["fixtures"][0].update(positive_for=[]),
        ):
            self.report = copy.deepcopy(original)
            mutation(self.report)
            with self.assertRaises(ValueError):
                self.evaluate()

    def test_source_pin_model_bytes_and_excluded_provenance_are_required(self):
        original = copy.deepcopy(self.report)
        for mutation in (
            lambda r: r["source"].update(repo_dirty=True),
            lambda r: r["source"].update(repo_commit="c" * 40),
            lambda r: r["model"].update(artifact_revision="c" * 40),
            lambda r: r["model"]["artifact_files"].update(config="wrong"),
            lambda r: r["source"]["excluded_fixtures"][0].update(reason="unreviewed"),
            lambda r: r["source"].update(excluded_fixtures=[]),
        ):
            self.report = copy.deepcopy(original)
            mutation(self.report)
            with self.assertRaises(ValueError):
                self.evaluate()

    def test_every_model_file_requires_manifest_checksum(self):
        (self.model / "image.onnx").write_text("replaced graph")
        with self.assertRaisesRegex(ValueError, "checksum"):
            image_calibration.model_identity(
                image_calibration.read(self.output / "models.json")
            )

    def test_profile_missing_or_skipped_tests_and_crashed_process_fail(self):
        self.write_profile(action="skip")
        with self.assertRaises(ValueError):
            self.evaluate()
        self.write_profile(omit=True)
        with self.assertRaises(ValueError):
            self.evaluate()
        self.write_profile()
        execution = image_calibration.read(self.output / "execution.json")
        execution["commands"]["calibration"] = 2
        self.write("execution.json", execution)
        with self.assertRaisesRegex(ValueError, "complete"):
            self.evaluate()

    def test_owned_integrations_cannot_be_missing_skipped_or_failed(self):
        for name in image_calibration.owned_omni_case_ids():
            with self.subTest(missing=name):
                self.write_owned(omit=name)
                with self.assertRaises(ValueError):
                    self.evaluate()
        for action in ("skip", "fail"):
            with self.subTest(action=action):
                self.write_owned(action=action)
                with self.assertRaises(ValueError):
                    self.evaluate()
        self.write_owned()
        (self.output / "owned-discovery.jsonl").write_text("")
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            self.evaluate()

    def test_owned_commands_use_the_attested_artifact_and_exact_inventory(self):
        normalized = {
            "cases": [{"id": "case", "status": "passed"}],
            "expected_cases": ["case"],
        }
        with (
            patch.object(image_calibration, "ROOT", self.root),
            patch.object(image_calibration, "evidence", return_value=normalized),
            patch.object(
                image_calibration.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=0),
            ) as run,
        ):
            image_calibration.run(self.output / "models.json", self.root / "execution")
        owned = run.call_args_list[:2]
        self.assertEqual(len(run.call_args_list), 5)
        for call in owned:
            self.assertEqual(call.kwargs["cwd"], self.root / "src/semantic-router")
            self.assertEqual(call.kwargs["env"]["VELA_OMNI_ARTIFACT"], str(self.model))
            self.assertEqual(call.kwargs["env"]["REQUIRE_OMNI_TESTS"], "1")
            for package, tests in image_calibration.OWNED_OMNI_TESTS.items():
                self.assertIn("./pkg/" + package, call.args[0])
                self.assertTrue(all(name in " ".join(call.args[0]) for name in tests))
        self.assertIn("-list", owned[0].args[0])
        self.assertIn("-run", owned[1].args[0])


if __name__ == "__main__":
    unittest.main()
