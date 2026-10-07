"""The published-model contract runs every listed managed test and its receipt cannot shrink."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_model_tests
import runtime_evidence

SHA = "a" * 40


def report() -> dict:
    suites = []
    for package, names, _ in run_model_tests.SELECTIONS:
        suites.append(
            {
                "package": package,
                "expected": list(names),
                "passed": list(names),
                "skipped": [],
                "failed": [],
                "missing": [],
                "exit_code": 0,
            }
        )
    return {
        "source_sha": SHA,
        "runtime": "model-runtime",
        "device": "cpu",
        "models": [{"env": "VLLM_SR_DOMAIN_MODEL", "path": "/models/domain"}],
        "suites": suites,
        "success": True,
    }


class ModelContractTests(unittest.TestCase):
    def test_skip_failure_or_missing_test_fails_the_suite(self):
        expected = {"TestA", "TestB"}
        passing = [{"Action": "pass", "Test": name} for name in sorted(expected)]
        self.assertTrue(run_model_tests.validate_results(passing, expected)["success"])
        for events in (
            passing[:1],
            [*passing[:1], {"Action": "skip", "Test": "TestB"}],
            [*passing, {"Action": "fail", "Test": "TestB"}],
            [],
        ):
            with self.subTest(events=events):
                self.assertFalse(
                    run_model_tests.validate_results(events, expected)["success"]
                )
        self.assertFalse(run_model_tests.validate_results([], set())["success"])

    def test_every_package_variable_names_a_runtime_built_in(self):
        self.assertEqual(
            set(run_model_tests.PACKAGES),
            {
                "VLLM_SR_DOMAIN_MODEL",
                "VLLM_SR_PII_MODEL",
                "VLLM_SR_JAILBREAK_MODEL",
                "VLLM_SR_FACTCHECK_MODEL",
                "VLLM_SR_FEEDBACK_MODEL",
                "VLLM_SR_MMBERT_TEST_MODEL",
            },
        )
        self.assertTrue(
            all(
                repo.startswith("vllm-sr/Vela-1.0-")
                for repo in run_model_tests.PACKAGES.values()
            )
        )

    def test_provisioned_packages_are_plain_directories_at_the_pinned_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            blobs = root / "blobs"
            snapshot.mkdir()
            blobs.mkdir()
            (blobs / "weights").write_text("weights")
            (snapshot / "model.safetensors").symlink_to(blobs / "weights")
            ref = types.SimpleNamespace(root=snapshot, revision="b" * 40)
            resolver = types.ModuleType("vllm_srun.registry.resolve")
            resolver.resolve = mock.Mock(return_value=ref)
            resolver.fetch = mock.Mock(side_effect=lambda ref, patterns, **_: ref)
            modules = {
                "vllm_srun": types.ModuleType("vllm_srun"),
                "vllm_srun.registry": types.ModuleType("vllm_srun.registry"),
                "vllm_srun.registry.resolve": resolver,
            }
            with mock.patch.dict(sys.modules, modules):
                models = run_model_tests.provision(root / "models")
                # A later fetch adds the router's label mappings to the snapshot.
                (snapshot / "category_mapping.json").write_text("{}")
                run_model_tests.provision(root / "models")
            self.assertEqual(set(models), set(run_model_tests.PACKAGES))
            for call in resolver.fetch.call_args_list:
                self.assertEqual(call.args[1], run_model_tests.ROUTER_FILES)
            for model in models.values():
                weights = Path(model["path"]) / "model.safetensors"
                self.assertFalse(weights.is_symlink())
                self.assertEqual(weights.read_text(), "weights")
                self.assertTrue(
                    (Path(model["path"]) / "category_mapping.json").is_file()
                )
                self.assertEqual(model["revision"], "b" * 40)

    def test_receipt_requires_the_complete_inventory_source_and_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)

            def evidence(value):
                (path / "results.json").write_text(json.dumps(value))
                with mock.patch.object(
                    runtime_evidence.subprocess, "check_output", return_value=SHA
                ):
                    return runtime_evidence.models_evidence(path)

            result = evidence(report())
            self.assertEqual(result["runtime"], "model-runtime")
            self.assertEqual(
                len(result["expected_cases"]), len(run_model_tests.required_inventory())
            )
            self.assertTrue(all(case["status"] == "passed" for case in result["cases"]))
            skipped = report()
            skipped["suites"][0]["skipped"] = [skipped["suites"][0]["passed"].pop()]
            self.assertTrue(
                any(case["status"] == "skipped" for case in evidence(skipped)["cases"])
            )
            crashed = report()
            crashed["suites"][0]["exit_code"] = 2
            self.assertIn(
                "failed", {case["status"] for case in evidence(crashed)["cases"]}
            )
            for mutate in (
                lambda value: value.update(source_sha="c" * 40),
                lambda value: value.update(device="cuda:0"),
                lambda value: value.update(runtime="candle"),
                lambda value: value["suites"][0]["expected"].pop(),
                lambda value: value["suites"].pop(),
            ):
                candidate = copy.deepcopy(report())
                mutate(candidate)
                with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                    evidence(candidate)


if __name__ == "__main__":
    unittest.main()
