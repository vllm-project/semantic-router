"""Shared GPU lease entries in the node launchers (static checks; no GPU, no docker)."""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "eval/run_same_panel.sh"
RELEASE = ROOT / "release/release.sh"


class SharedLeaseTest(unittest.TestCase):
    def test_scripts_parse(self):
        for script in (RUNNER, RELEASE):
            subprocess.run(["bash", "-n", str(script)], check=True)

    def test_no_launcher_writes_the_owner_entry_directly(self):
        for script in (RUNNER, RELEASE):
            text = script.read_text(encoding="utf-8")
            self.assertIn("--shared-lease", text, script.name)
            self.assertIsNone(re.search(r'>>?\s*"\$lease/owner"', text), script.name)

    def test_runner_shared_lease_is_a_named_entry_without_the_idle_gate(self):
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn('lease_name="owner.$2" shared=1', text)
        self.assertRegex(
            text, r'if \[\[ "\$lease_name" == "owner" && -f "\$lease/owner" \]\]'
        )
        self.assertRegex(text, r'if \[\[ "\$shared" == 0 && \(')
        self.assertRegex(text, r'>\s*"\$lease/\$lease_name"')

    def test_release_shared_lease_writes_only_its_entry(self):
        text = RELEASE.read_text(encoding="utf-8")
        self.assertIn('lease_file="$lease/owner.$shared"', text)
        self.assertRegex(text, r'>\s*"\$lease_file"')


if __name__ == "__main__":
    unittest.main()
