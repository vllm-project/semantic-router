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
                result, calls = self._run_target(failure)
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

    def test_calibration_prepares_the_nano_artifact_in_the_models_directory(self):
        result, calls = self._run_target()
        self.assertEqual(result.returncode, 0, result.stderr)
        root = Path(calls[1]["manifest"]).parent.parent
        self.assertEqual(calls[0]["output"], str(root / "models/vela-omni-artifacts"))
        self.assertEqual(calls[0]["variants"], "nano")

    def _run_target(
        self, failure: str = ""
    ) -> tuple[subprocess.CompletedProcess[str], list[dict]]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bin").mkdir()
            (root / "src/semantic-router").mkdir(parents=True)
            # Recursive $(MAKE) loads this same fixture, including the actual
            # production targets.
            (root / "Makefile").write_text(f"include {MODELS_MAKE}\nLOG_TARGET = :\n")
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
                    "for name in ('manifest', 'output', 'variants'):\n"
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
                    "verify-image-routing-calibration",
                    "SHELL=/bin/sh",
                    "AGENT_PYTHON=python3",
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
