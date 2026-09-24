"""Stable Decision selection follows packaged source capability."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/ci"))

from decision_release_policy import requires_qualification  # noqa: E402


class DecisionReleasePolicyTests(unittest.TestCase):
    def test_current_source_requires_qualification_even_without_a_diff(self) -> None:
        self.assertTrue(requires_qualification(ROOT))

    def test_unrelated_cli_release_does_not_require_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "src/vllm-sr"
            project.mkdir(parents=True)
            (project / "pyproject.toml").write_text(
                '[project]\nname = "vllm-sr"\nversion = "0.3.0"\n'
            )
            self.assertFalse(requires_qualification(root))
            command = project / "cli/commands/drun.py"
            command.parent.mkdir(parents=True)
            command.touch()
            self.assertTrue(requires_qualification(root))

    def test_publisher_rejects_unqualified_current_source(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/ci/decision_release_policy.py"),
                "--expect",
                "false",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("differs from the release source", result.stderr)


if __name__ == "__main__":
    unittest.main()
