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
    def test_every_published_model_stage_failure_propagates(self) -> None:
        for failure, count in (
            ("", 2),
            ("python3:model-artifacts", 1),
            ("python3:runtime", 2),
        ):
            with self.subTest(failure=failure):
                result, calls = self._run_target(failure)
                self.assertEqual(
                    result.returncode == 0, not failure, result.stdout + result.stderr
                )
                self.assertEqual(
                    [call["suite"] for call in calls],
                    ["model-artifacts", "runtime"][:count],
                )

    def test_the_contract_serves_the_nano_artifact_from_the_models_directory(self):
        result, calls = self._run_target()
        self.assertEqual(result.returncode, 0, result.stderr)
        models = Path(calls[1]["models-dir"])
        self.assertEqual(calls[0]["output"], str(models / "vela-omni-artifacts"))
        self.assertEqual(calls[0]["variants"], "nano")
        self.assertEqual(
            calls[1]["omni"], str(models / "vela-omni-artifacts/vela-1.0-omni-nano")
        )

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
                    "default_suite = 'runtime'\n"
                    "if args[0].endswith('prepare_model_test_assets.py'): default_suite = 'model-artifacts'\n"
                    "call = {'command': pathlib.Path(sys.argv[0]).name,\n"
                    "        'suite': option('--suite', default_suite)}\n"
                    "for name in ('models-dir', 'omni', 'output', 'variants'):\n"
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
                    "test-models",
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
