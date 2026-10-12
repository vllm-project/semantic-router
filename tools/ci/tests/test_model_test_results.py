"""The published-model contract runs every listed managed test and its receipt cannot shrink."""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import types
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_model_tests
import runtime_evidence

SHA = "a" * 40
# A runtime that records its arguments and answers /health until it is killed.
FAKE_RUNTIME = """
import http.server, json, sys
args = sys.argv[2:]
open(sys.argv[1], "w").write(json.dumps(args))
port = int(args[args.index("--port") + 1])
class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")
    def log_message(self, *_):
        pass
http.server.HTTPServer(("127.0.0.1", port), Health).serve_forever()
"""


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

    def test_each_vela2_size_has_its_own_runtime_and_endpoint(self):
        suites = run_model_tests.VELA2_SUITES.values()
        self.assertEqual(
            [(repo, profile) for repo, profile, _, _ in suites],
            [
                ("vllm-sr/Vela-2.0-0.3B", "max_speed"),
                ("vllm-sr/Vela-2.0-0.8B", "exact"),
            ],
        )
        self.assertEqual(len({env for _, _, env, _ in suites}), len(suites))
        logs = {filename for _, _, filename in run_model_tests.SELECTIONS}
        self.assertTrue(set(run_model_tests.VELA2_SUITES) <= logs)

    def test_core_excludes_exactly_the_published_inventory(self):
        profiles = json.loads(
            (run_model_tests.ROOT / "tools/ci/core_test_profiles.json").read_text()
        )
        excluded = {
            (record["package"], record["test"])
            for record in profiles["excluded"]
            if record["profile"] == "model-runtime"
        }
        self.assertEqual(excluded, run_model_tests.required_inventory())

    def test_vela2_sizes_download_their_pinned_files_into_the_runtime_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weights").write_text("weights")
            ref = types.SimpleNamespace(root=root, revision="b" * 40)
            resolver = types.ModuleType("vllm_srun.registry.resolve")
            resolver.resolve = mock.Mock(return_value=ref)
            modules = {
                "vllm_srun": types.ModuleType("vllm_srun"),
                "vllm_srun.registry": types.ModuleType("vllm_srun.registry"),
                "vllm_srun.registry.resolve": resolver,
            }
            with mock.patch.dict(sys.modules, modules):
                models = run_model_tests.provision_vela2(root / "cache")
        self.assertEqual(set(models), set(run_model_tests.VELA2_SUITES))
        for call in resolver.resolve.call_args_list:
            self.assertEqual(call.kwargs, {"cache_dir": root / "cache"})
        for log, model in models.items():
            repo, profile, env, _ = run_model_tests.VELA2_SUITES[log]
            self.assertEqual(
                (model["repo_id"], model["profile"], model["env"], model["revision"]),
                (repo, profile, env, "b" * 40),
            )
            self.assertEqual(model["bytes"], len("weights"))

    def test_a_vela2_runtime_serves_its_pinned_size_offline_until_its_tests_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "srun.py").write_text(FAKE_RUNTIME)
            arguments = root / "args.json"
            model = {
                "env": "VLLM_SRUN_VELA2_ENDPOINT",
                "repo_id": "vllm-sr/Vela-2.0-0.3B",
                "revision": "b" * 40,
                "profile": "max_speed",
            }
            command = f"{sys.executable} {root / 'srun.py'} {arguments}"
            with (
                mock.patch.dict(os.environ, {"VLLM_SRUN_COMMAND": command}),
                run_model_tests.serve_vela2(
                    model, root / "cache", root / "runtime.log"
                ) as endpoint,
            ):
                with urllib.request.urlopen(endpoint + "/health") as response:
                    self.assertEqual(response.status, 200)
                argv = json.loads(arguments.read_text())
            with self.assertRaises(OSError):
                urllib.request.urlopen(endpoint + "/health", timeout=2)
        self.assertEqual(argv[:2], ["serve", "vllm-sr/Vela-2.0-0.3B"])
        for flag, value in (
            ("--revision", "b" * 40),
            ("--device", "cpu"),
            ("--profile", "max_speed"),
            ("--cache-dir", str(root / "cache")),
            ("--result-cache-entries", "0"),
        ):
            self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertIn("--offline", argv)
        self.assertIn("ready_seconds", model)

    def test_a_vela2_runtime_that_exits_fails_every_test_it_serves(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = {
                "env": "VLLM_SRUN_VELA2_ENDPOINT",
                "repo_id": "vllm-sr/Vela-2.0-0.3B",
                "revision": "b" * 40,
                "profile": "max_speed",
            }
            command = f"{sys.executable} -c 'raise SystemExit(3)'"
            with mock.patch.dict(os.environ, {"VLLM_SRUN_COMMAND": command}):
                result = run_model_tests.run_vela2_suite(
                    "./pkg/classification",
                    ("TestA", "TestB"),
                    {},
                    root / "vela2-0.3b.jsonl",
                    model,
                    root / "cache",
                )
        self.assertFalse(result["success"])
        self.assertEqual(result["missing"], ["TestA", "TestB"])
        self.assertIn("exited 3", result["error"])

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
