import glob
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


class LocalIntegrationEvidenceTests(unittest.TestCase):
    def setUp(self):
        workflow = yaml.safe_load(
            (REPO_ROOT / ".github/workflows/test-local.yml").read_text()
        )
        steps = workflow["jobs"]["integration"]["steps"]
        self.execute = next(
            step for step in steps if step.get("name") == "Execute selected suite"
        )
        self.publish = next(
            step
            for step in steps
            if step.get("name") == "Publish local integration evidence"
        )

    def test_evidence_is_published_after_suite_failure(self):
        self.assertEqual(self.publish["if"], "always()")
        self.assertEqual(self.publish["with"]["if-no-files-found"], "error")
        self.assertTrue(self.publish["with"]["include-hidden-files"])

    def test_actual_suite_step_uploads_only_collected_logs(self):
        for suite, exit_code in (("cli", 0), ("memory", 0), ("memory", 37)):
            with self.subTest(
                suite=suite, exit_code=exit_code
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                workspace = root / "workspace"
                workspace.mkdir()
                temporary = root / "runner-temp"
                temporary.mkdir()
                binaries = root / "bin"
                binaries.mkdir()
                make = binaries / "make"
                make.write_text(
                    f"#!{sys.executable}\n"
                    + r"""import os
import sys
from pathlib import Path
if os.environ["SUITE"] == "memory":
    state = Path(os.environ["MEMORY_TEST_DIR"])
    restricted = state / ".vllm-sr/dashboard-data/evaluation"
    restricted.mkdir(parents=True)
    (restricted / "session.db").write_text("runtime state")
    restricted.chmod(0)
    (state / "models").mkdir()
    (state / "models/model.safetensors").write_text("weights")
    evidence = Path(os.environ["MEMORY_TEST_ARTIFACT_DIR"]) / ("memory-" + os.environ["VLLM_SR_STACK_NAME"])
    evidence.mkdir(parents=True)
    for name in ("serve.log", "router-startup.log", "router.predump.log"):
        (evidence / name).write_text("collected log")
print("suite evidence", flush=True)
sys.exit(int(os.environ["FIXTURE_EXIT_CODE"]))
"""
                )
                make.chmod(0o755)
                replacements = {
                    "${{ github.workspace }}": str(workspace),
                    "${{ runner.temp }}": str(temporary),
                    "${{ github.run_id }}": "12345",
                    "${{ github.run_attempt }}": "2",
                    "${{ fromJSON(inputs.verification).suite }}": suite,
                }
                environment = dict(os.environ)
                for key, value in self.execute["env"].items():
                    resolved = value
                    for source, target in replacements.items():
                        resolved = resolved.replace(source, target)
                    environment[key] = resolved
                environment["PATH"] = (
                    str(binaries) + os.pathsep + environment.get("PATH", "")
                )
                environment["FIXTURE_EXIT_CODE"] = str(exit_code)
                state = Path(environment["MEMORY_TEST_DIR"])
                restricted = state / ".vllm-sr/dashboard-data/evaluation"
                try:
                    result = subprocess.run(
                        ["bash", "-e", "-c", self.execute["run"]],
                        cwd=workspace,
                        env=environment,
                        text=True,
                        capture_output=True,
                        check=False,
                        timeout=10,
                    )
                    self.assertEqual(
                        result.returncode, exit_code, result.stdout + result.stderr
                    )
                    self.assertTrue(state.is_relative_to(temporary))
                    selected = []
                    for pattern in self.publish["with"]["path"].splitlines():
                        selected.extend(
                            Path(path)
                            for path in glob.glob(
                                str(workspace / pattern), recursive=True
                            )
                        )
                    self.assertTrue(selected)
                    # Directory roots would let upload-artifact recurse into state.
                    self.assertTrue(
                        all(
                            path.is_file() and path.suffix == ".log"
                            for path in selected
                        ),
                        selected,
                    )
                    self.assertFalse(
                        any(path.is_relative_to(state) for path in selected)
                    )
                    self.assertEqual(
                        {path.name for path in selected},
                        {suite + ".log"}
                        | (
                            {"serve.log", "router-startup.log", "router.predump.log"}
                            if suite == "memory"
                            else set()
                        ),
                    )
                finally:
                    if restricted.exists():
                        restricted.chmod(0o700)


if __name__ == "__main__":
    unittest.main()
