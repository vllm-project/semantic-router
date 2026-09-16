"""Root-level documentation builds provision Python before checking artifacts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class DocsBuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.log = self.root / "calls.jsonl"
        self.bin = self.root / "bin"
        self.bin.mkdir()
        (self.root / "website/scripts").mkdir(parents=True)
        (self.root / "tools/docs").mkdir(parents=True)
        shutil.copyfile(ROOT / "tools/make/docs.mk", self.root / "docs.mk")
        shutil.copyfile(
            ROOT / "website/scripts/check-generated.mjs",
            self.root / "website/scripts/check-generated.mjs",
        )
        (self.root / "tools/docs/requirements.txt").write_text("click==8.5.0\n")
        self.node = shutil.which("node")
        self.assertIsNotNone(self.node, "Documentation build tests require Node.js")
        self.environment = {
            **os.environ,
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "DOCS_TEST_ROOT": str(self.root),
            "DOCS_TEST_LOG": str(self.log),
        }
        self.environment.pop("VLLM_SR_DOCS_PYTHON", None)
        self.bootstrap = self.bin / "bootstrap-python"
        self.write_executable(
            self.bootstrap,
            """import json, os, pathlib, shutil, sys
root = pathlib.Path(os.environ['DOCS_TEST_ROOT'])
args = sys.argv[1:]
with open(os.environ['DOCS_TEST_LOG'], 'a') as stream:
    stream.write(json.dumps(['bootstrap', args]) + '\\n')
assert args[:2] == ['-m', 'venv'], args
python = pathlib.Path(args[2]) / 'bin/python'
python.parent.mkdir(parents=True)
shutil.copyfile(root / 'fake-python', python)
python.chmod(0o755)
""",
        )
        self.write_executable(
            self.root / "fake-python",
            """import json, os, pathlib, sys
root = pathlib.Path(os.environ['DOCS_TEST_ROOT'])
args = sys.argv[1:]
with open(os.environ['DOCS_TEST_LOG'], 'a') as stream:
    stream.write(json.dumps(['python', args, sys.argv[0]]) + '\\n')
if args[:3] == ['-m', 'pip', 'install']:
    assert args[-2:] == ['-r', 'tools/docs/requirements.txt'], args
    (root / 'dependencies-ready').touch()
else:
    assert (root / 'dependencies-ready').is_file(), 'dependencies were not installed'
    assert args[-1] == '--check', 'build must check, never regenerate artifacts'
    if os.environ.get('DOCS_TEST_STALE'):
        sys.exit(9)
""",
        )
        self.write_executable(
            self.bin / "npm",
            f"""import json, os, pathlib, subprocess, sys
root = pathlib.Path(os.environ['DOCS_TEST_ROOT'])
args = sys.argv[1:]
with open(os.environ['DOCS_TEST_LOG'], 'a') as stream:
    stream.write(json.dumps(['npm', args, os.environ.get('VLLM_SR_DOCS_PYTHON')]) + '\\n')
assert (root / 'dependencies-ready').is_file(), 'npm started before Python setup'
if args == ['run', 'build']:
    assert os.environ['VLLM_SR_DOCS_PYTHON'] == str(root / 'website/.venv/bin/python')
    sys.exit(subprocess.call([{self.node!r}, str(root / 'website/scripts/check-generated.mjs')]))
assert args == ['install'], args
""",
        )
        for name in (
            "generate-configuration-catalog.mjs",
            "generate-contributor-rank.mjs",
            "generate-committer-activity.mjs",
        ):
            (self.root / "website/scripts" / name).write_text(
                "import { appendFileSync } from 'node:fs'\n"
                "appendFileSync(process.env.DOCS_TEST_LOG, "
                "JSON.stringify(['node', process.argv.slice(1)]) + '\\n')\n"
            )

    @staticmethod
    def write_executable(path, body):
        path.write_text("#!/usr/bin/env python3\n" + body)
        path.chmod(0o755)

    def run_make(self, target, **environment):
        return subprocess.run(
            [
                "make",
                "-f",
                "docs.mk",
                "-j4",
                target,
                "LOG_TARGET=true",
                f"DOCS_PYTHON={self.bootstrap}",
            ],
            cwd=self.root,
            env={**self.environment, **environment},
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_clean_root_build_installs_python_before_npm_and_checks(self):
        result = self.run_make("docs-build")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls()
        self.assertEqual(
            [call[0] for call in calls[:4]], ["bootstrap", "python", "npm", "npm"]
        )
        self.assertEqual(calls[1][1][:3], ["-m", "pip", "install"])
        self.assertEqual(calls[2][1], ["install"])
        self.assertEqual(calls[3][1], ["run", "build"])
        checks = [call for call in calls[4:] if call[0] == "python"]
        self.assertEqual(len(checks), 3)
        self.assertTrue(
            all(
                call[2] == str(self.root / "website/.venv/bin/python")
                for call in checks
            )
        )
        self.assertTrue(all(call[1][-1] == "--check" for call in checks))

    def test_build_still_fails_when_generated_artifact_is_stale(self):
        result = self.run_make("docs-build", DOCS_TEST_STALE="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Generated reference check failed", result.stderr)
        self.assertEqual(len([call for call in self.calls() if call[0] == "python"]), 2)

    def test_direct_npm_and_later_make_checks_reuse_docs_environment(self):
        installed = self.run_make("docs-python-install")
        self.assertEqual(installed.returncode, 0, installed.stderr)
        self.log.write_text("")
        result = subprocess.run(
            [self.node, "website/scripts/check-generated.mjs"],
            cwd=self.root,
            env=self.environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(
            ["make", "-f", "docs.mk", "docs-cli-check"],
            cwd=self.root,
            env=self.environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        checks = [call for call in self.calls() if call[0] == "python"]
        self.assertEqual(len(checks), 4)
        self.assertTrue(
            all(
                call[2] == str(self.root / "website/.venv/bin/python")
                for call in checks
            )
        )


if __name__ == "__main__":
    unittest.main()
