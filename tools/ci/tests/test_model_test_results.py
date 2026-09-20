import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location(
    "run_model_tests", Path(__file__).parents[1] / "run_model_tests.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

MULTIMODAL_BINDING = (
    "TestMultiModalEmbeddingInit",
    "TestMultiModalEncodeText",
    "TestMultiModalInputValidation",
)
MULTIMODAL_CLASSIFICATION = (
    "TestEmbeddingClassifier_IntegrationImageQueryEndToEnd",
    "TestEmbeddingClassifier_IntegrationTextRulesIgnoredOnImagePath",
)


def invoke_runner(
    suite="multimodal", provider="candle", *, events=None, codes=None, models=None
):
    """Exercise CLI selection, subprocess construction, and receipt validation."""
    if events is None:
        expected = (
            (MULTIMODAL_BINDING, MULTIMODAL_CLASSIFICATION)
            if suite == "multimodal"
            else (
                tuple("TestPublishedVelaModels/" + name for name in runner.FAMILIES),
                runner.CLASSIFIER_TESTS,
            )
        )
        if suite == "runtime" and provider == "candle":
            expected += (runner.CANDLE_CACHE_TESTS,)
        if suite == "runtime" and provider == "ort":
            expected += (("TestOwnedImplicitORTEmbeddingAndExplicitCandleOverride",),)
        events = [
            [{"Action": "pass", "Test": name} for name in names] for names in expected
        ]
    if codes is None:
        codes = (0,) * len(events)
    processes = [
        SimpleNamespace(
            stdout=iter(json.dumps(event) + "\n" for event in batch),
            wait=Mock(return_value=code),
        )
        for batch, code in zip(events, codes, strict=True)
    ]
    if models is None:
        models = (
            [
                {
                    "name": "Multimodal",
                    "env": "MULTIMODAL_MODEL_PATH",
                    "path": "/explicit/multimodal",
                }
            ]
            if suite == "multimodal"
            else [
                {"name": name, "env": f"MODEL_{name}", "path": f"/explicit/{name}"}
                for name in runner.FAMILIES
            ]
        )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = root / "models.json"
        manifest.write_text(json.dumps({"provider": provider, "models": models}))
        output = root / "report"
        argv = [
            "run_model_tests.py",
            "--suite",
            suite,
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ]
        with (
            patch.object(runner.sys, "argv", argv),
            patch.object(runner.subprocess, "Popen", side_effect=processes) as popen,
            patch.object(
                runner.subprocess, "check_output", return_value="source-sha\n"
            ),
        ):
            code = runner.main()
        receipt = json.loads((output / "results.json").read_text())
        logs = {path.name: path.read_text() for path in output.glob("*.jsonl")}
        return code, receipt, popen.call_args_list, logs


class ModelResultTests(unittest.TestCase):
    def test_multimodal_runs_only_the_five_cpu_compatibility_cases(self):
        code, receipt, calls, logs = invoke_runner()
        self.assertEqual(code, 0)
        self.assertTrue(receipt["success"])
        self.assertEqual(receipt["suite"], "multimodal")
        self.assertEqual(receipt["provider"], "candle")
        self.assertEqual(receipt["device"], "cpu")
        self.assertEqual(receipt["source_sha"], "source-sha")
        self.assertEqual(set(logs), {"binding.jsonl", "classification.jsonl"})
        self.assertEqual(len(calls), 2)
        for call, module, package, names in zip(
            calls,
            ("candle-binding", "src/semantic-router"),
            (".", "./pkg/classification"),
            (MULTIMODAL_BINDING, MULTIMODAL_CLASSIFICATION),
            strict=True,
        ):
            args = call.args[0]
            self.assertEqual(call.kwargs["cwd"], runner.ROOT / module)
            self.assertEqual(args[-1], package)
            self.assertEqual(
                args[args.index("-run") + 1], "^(" + "|".join(names) + ")$"
            )
            self.assertIn("-count=1", args)
            self.assertEqual(
                call.kwargs["env"]["MULTIMODAL_MODEL_PATH"], "/explicit/multimodal"
            )
            self.assertEqual(call.kwargs["env"]["VLLM_SR_REQUIRE_MODEL_TESTS"], "1")

    def test_multimodal_rejects_each_missing_skipped_or_failed_case(self):
        for name in (*MULTIMODAL_BINDING, *MULTIMODAL_CLASSIFICATION):
            for action in (None, "skip", "fail"):
                with self.subTest(test=name, action=action):
                    events = [
                        [
                            {"Action": action if test == name else "pass", "Test": test}
                            for test in names
                            if test != name or action is not None
                        ]
                        for names in (MULTIMODAL_BINDING, MULTIMODAL_CLASSIFICATION)
                    ]
                    code, receipt, _, _ = invoke_runner(events=events)
                    self.assertEqual(code, 1)
                    self.assertFalse(receipt["success"])

    def test_multimodal_rejects_empty_execution_and_process_failure(self):
        for arguments in ({"events": [[], []]}, {"codes": (1, 0)}, {"codes": (0, 1)}):
            with self.subTest(arguments=arguments):
                code, receipt, _, _ = invoke_runner(**arguments)
                self.assertEqual(code, 1)
                self.assertFalse(receipt["success"])

    def test_multimodal_requires_candle_and_one_explicit_model(self):
        valid = {"name": "Multimodal", "env": "MULTIMODAL_MODEL_PATH", "path": "/model"}
        for arguments in (
            {"provider": "ort"},
            {"models": []},
            {"models": [valid, valid]},
            {"models": [{**valid, "name": "Embedding"}]},
            {"models": [{**valid, "env": "WRONG_MODEL_PATH"}]},
            {"models": [{**valid, "path": ""}]},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                invoke_runner(**arguments)

    def test_runtime_still_requires_ten_families_and_nine_integrations(self):
        self.assertEqual(len(runner.FAMILIES), 10)
        self.assertEqual(len(runner.CLASSIFIER_TESTS), 9)
        for provider in ("candle", "ort"):
            with self.subTest(provider=provider):
                code, receipt, calls, logs = invoke_runner("runtime", provider)
                self.assertEqual(code, 0)
                self.assertTrue(receipt["success"])
                self.assertEqual(receipt["suite"], "runtime")
                self.assertEqual(
                    set(logs),
                    {"native.jsonl", "classification.jsonl"}
                    | (
                        {"default-execution.jsonl"}
                        if provider == "ort"
                        else {"cache.jsonl"}
                    ),
                )
                self.assertEqual(
                    [len(result["passed"]) for result in receipt["suites"]],
                    [10, 9, 1],
                )
                for call in calls:
                    self.assertEqual(
                        call.kwargs["cwd"], runner.ROOT / "src/semantic-router"
                    )
                    self.assertEqual(
                        call.kwargs["env"]["VLLM_SR_MODEL_TEST_PROVIDER"], provider
                    )
                    self.assertIn("CANDLE_GENERIC_CLASSIFIER_MODEL", call.kwargs["env"])

    def test_cache_checkpoint_has_a_required_candle_owner(self):
        code, receipt, calls, _ = invoke_runner("runtime", "candle")
        self.assertEqual(code, 0)
        cache = calls[-1]
        self.assertEqual(cache.args[0][-1], "./pkg/cache")
        self.assertEqual(
            cache.kwargs["env"]["VLLM_SR_MMBERT_TEST_MODEL"], "/explicit/Embedding"
        )
        self.assertEqual(
            receipt["suites"][-1]["expected"], list(runner.CANDLE_CACHE_TESTS)
        )
        profiles = json.loads(
            (runner.ROOT / "tools/ci/core_test_profiles.json").read_text()
        )
        owners = [
            row
            for row in profiles["excluded"]
            if row["package"] == "./pkg/cache"
            and row["test"] in runner.CANDLE_CACHE_TESTS
        ]
        self.assertEqual(len(owners), len(runner.CANDLE_CACHE_TESTS))
        self.assertTrue(all(row["profile"] == "native" for row in owners))

    def test_candle_cache_checkpoint_cannot_be_missing_skipped_or_failed(self):
        native = [
            {"Action": "pass", "Test": "TestPublishedVelaModels/" + name}
            for name in runner.FAMILIES
        ]
        classifiers = [
            {"Action": "pass", "Test": name} for name in runner.CLASSIFIER_TESTS
        ]
        for action in (None, "skip", "fail"):
            with self.subTest(action=action):
                cache = (
                    [{"Action": action, "Test": runner.CANDLE_CACHE_TESTS[0]}]
                    if action is not None
                    else []
                )
                code, receipt, _, _ = invoke_runner(
                    "runtime", "candle", events=[native, classifiers, cache]
                )
                self.assertEqual(code, 1)
                self.assertFalse(receipt["success"])

    def test_published_batch_must_execute_for_both_providers(self):
        expected = set(runner.CLASSIFIER_TESTS)
        self.assertIn("TestUnifiedClassifierPublishedModels", expected)
        events = [{"Action": "pass", "Test": name} for name in expected]
        self.assertTrue(runner.validate_results(events, expected)["success"])
        missing_batch = [
            event
            for event in events
            if event["Test"] != "TestUnifiedClassifierPublishedModels"
        ]
        self.assertFalse(runner.validate_results(missing_batch, expected)["success"])

    def test_required_execution(self):
        passed = [{"Action": "pass", "Test": "a"}]
        self.assertTrue(runner.validate_results(passed, {"a"})["success"])
        for events, expected in (
            (passed, {"a", "b"}),
            ([], set()),
            ([*passed, {"Action": "skip", "Test": "a/optional"}], {"a"}),
            ([*passed, {"Action": "fail", "Package": "native"}], {"a"}),
        ):
            self.assertFalse(runner.validate_results(events, expected)["success"])
