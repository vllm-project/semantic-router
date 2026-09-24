"""Execute the real model Make recipes without downloading or running models."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_MAKE = REPO_ROOT / "tools/make/models.mk"


class ModelMakeContractsTests(unittest.TestCase):
    def test_image_calibration_is_separate_and_every_stage_failure_propagates(
        self,
    ) -> None:
        for failure, count in (
            ("", 3),
            ("python3:model-artifacts", 1),
            ("python3:image-calibration-manifest", 2),
            ("python3:image-calibration", 3),
        ):
            with self.subTest(failure=failure):
                result, calls = self._run_target(
                    "ort", failure, "verify-image-routing-calibration"
                )
                self.assertEqual(
                    result.returncode == 0, not failure, result.stdout + result.stderr
                )
                self.assertEqual(len(calls), count)
                self.assertTrue(
                    calls[0]["suite"] == "model-artifacts"
                    and all(
                        call["suite"].startswith("image-calibration")
                        for call in calls[1:]
                    )
                )
                if not failure:
                    self.assertEqual(calls[1]["manifest"], calls[2]["manifest"])

    def test_candle_runs_both_suites_with_separate_manifests_and_reports(self) -> None:
        result, calls = self._run_target("candle")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            [(call["command"], call["suite"]) for call in calls],
            [
                ("go", "runtime"),
                ("python3", "runtime"),
                ("go", "multimodal"),
                ("python3", "multimodal"),
            ],
        )
        for provision, evaluation in (calls[:2], calls[2:]):
            self.assertEqual(provision["manifest"], evaluation["manifest"])
            self.assertEqual(provision["provider"], "candle")
            self.assertEqual(evaluation["device"], "cpu")
            self.assertEqual(
                evaluation["manifest"], str(Path(evaluation["output"]) / "models.json")
            )
        self.assertEqual(
            calls[3]["output"], str(Path(calls[1]["output"]) / "multimodal")
        )

    def test_any_selected_suite_failure_fails_the_candle_target(self) -> None:
        for command, suite, expected_calls in (
            ("go", "runtime", 1),
            ("python3", "runtime", 2),
            ("go", "multimodal", 3),
            ("python3", "multimodal", 4),
        ):
            with self.subTest(command=command, suite=suite):
                result, calls = self._run_target("candle", f"{command}:{suite}")
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("injected model contract failure", result.stderr)
                self.assertEqual(len(calls), expected_calls, calls)

    def test_ort_runs_only_its_supported_runtime_suite(self) -> None:
        result, calls = self._run_target("ort", "python3:multimodal")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            [(call["command"], call["suite"]) for call in calls],
            [("python3", "model-artifacts"), ("go", "runtime"), ("python3", "runtime")],
        )
        self.assertEqual(calls[1]["provider"], "ort")
        self.assertEqual(calls[2]["device"], "cpu")

    def test_runtime_and_calibration_use_the_shared_preparation_directory(self):
        for target in ("test-models", "verify-image-routing-calibration"):
            with self.subTest(target=target):
                result, calls = self._run_target("ort", target=target)
                self.assertEqual(result.returncode, 0, result.stderr)
                root = Path(calls[1]["manifest"]).parent.parent
                self.assertEqual(
                    calls[0]["output"], str(root / "models/vela-omni-artifacts")
                )
                self.assertEqual(calls[0]["variants"], "nano")

    def test_ort_preparation_failure_stops_execution(self):
        result, calls = self._run_target("ort", "python3:model-artifacts")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(calls), 1)

    def _run_target(
        self, provider: str, failure: str = "", target: str = "test-models"
    ) -> tuple[subprocess.CompletedProcess[str], list[dict]]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bin").mkdir()
            (root / "src/semantic-router").mkdir(parents=True)
            # Recursive $(MAKE) loads this same fixture, including the actual
            # production targets. Only the native compiler prerequisite is inert.
            (root / "Makefile").write_text(
                f"include {MODELS_MAKE}\n"
                "LOG_TARGET = :\nNATIVE_ENV = MODEL_CONTRACT_NATIVE=1\n"
                ".PHONY: rust-ci\nrust-ci:\n\t@:\n"
            )
            for name in ("go", "python3", "docker"):
                executable = root / "bin" / name
                executable.write_text(
                    f"#!{sys.executable}\n"
                    "import json, os, pathlib, sys\n"
                    "args = sys.argv[1:]\n"
                    "def option(name, default=''):\n"
                    "    return args[args.index(name) + 1] if name in args else default\n"
                    "default_suite = 'image-calibration' if args[0].endswith('image_calibration.py') or pathlib.Path(sys.argv[0]).name == 'docker' else 'runtime'\n"
                    "if args[0].endswith('prepare_model_test_assets.py'): default_suite = 'model-artifacts'\n"
                    "if '--prepare-manifest' in args: default_suite += '-manifest'\n"
                    "call = {'command': pathlib.Path(sys.argv[0]).name,\n"
                    "        'suite': option('--suite', default_suite)}\n"
                    "for name in ('manifest', 'output', 'provider', 'device', 'variants'):\n"
                    "    call[name] = option('--' + name)\n"
                    "with open(os.environ['MODEL_CONTRACT_CALLS'], 'a') as output:\n"
                    "    output.write(json.dumps(call) + '\\n')\n"
                    "if call['command'] + ':' + call['suite'] == os.environ['MODEL_CONTRACT_FAIL']:\n"
                    "    print('injected model contract failure', file=sys.stderr)\n"
                    "    sys.exit(37)\n"
                )
                executable.chmod(0o755)
            calls_path = root / "calls.jsonl"
            environment = {
                key: value
                for key, value in os.environ.items()
                if key not in {"MAKEFLAGS", "MFLAGS", "MAKEOVERRIDES", "MAKEFILES"}
            }
            environment.update(
                PATH=str(root / "bin") + os.pathsep + environment.get("PATH", ""),
                MODEL_CONTRACT_CALLS=str(calls_path),
                MODEL_CONTRACT_FAIL=failure,
            )
            result = subprocess.run(
                [
                    "make",
                    "--no-print-directory",
                    target,
                    "SHELL=/bin/sh",
                    f"MODEL_TEST_PROVIDER={provider}",
                    "MODEL_TEST_DEVICE=cpu",
                    f"MODEL_TEST_REPORT_DIR={root / 'reports'}",
                    f"MODEL_TEST_MODELS_DIR={root / 'models'}",
                ],
                cwd=root,
                env=environment,
                text=True,
                capture_output=True,
                timeout=15,
                check=False,
            )
            calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
            return result, calls


if __name__ == "__main__":
    unittest.main()
