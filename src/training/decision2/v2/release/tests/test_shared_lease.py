"""Shared GPU lease entries in the node launchers (static checks; no GPU, no docker)."""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = (ROOT / "eval/run_same_panel.sh", ROOT / "release/release.sh")


class SharedLeaseTest(unittest.TestCase):
    def test_scripts_parse(self):
        for script in SCRIPTS:
            subprocess.run(["bash", "-n", str(script)], check=True)

    def test_every_lease_write_goes_through_the_selected_entry(self):
        for script in SCRIPTS:
            text = script.read_text(encoding="utf-8")
            self.assertIsNone(re.search(r'>>?\s*"\$lease/owner"', text), script.name)
            self.assertIn('lease_file="$lease/owner.$shared"', text)
            self.assertRegex(text, r'>\s*"\$lease_file"')

    def test_shared_mode_skips_the_owner_checks(self):
        text = SCRIPTS[0].read_text(encoding="utf-8")
        shared, _, rest = text.partition('lease_file="$lease/owner.$shared"')
        owner_check = rest.split("else", 1)[1].split("\nfi\n", 1)[0]
        self.assertIn("leased by another track", owner_check)
        self.assertIn("VRAM", owner_check)


if __name__ == "__main__":
    unittest.main()
