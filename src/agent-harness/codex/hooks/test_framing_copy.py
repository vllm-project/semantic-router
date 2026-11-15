"""framing.py is the Claude Code plugin's copy of its hooks' questions, copied again here, since
a plugin folder imports nothing from beside it. In the repository the two copies must match, so
Codex asks what Claude Code asks."""

from __future__ import annotations

import unittest
from pathlib import Path

HERE = Path(__file__).with_name("framing.py")
SOURCE = (
    Path(__file__).parents[2]
    / "claude-code"
    / "skills"
    / "setup"
    / "scripts"
    / "framing.py"
)


class FramingCopyTest(unittest.TestCase):
    @unittest.skipUnless(SOURCE.is_file(), "outside the repository")
    def test_the_copy_matches_the_claude_code_plugin(self):
        self.assertEqual(
            HERE.read_text(),
            SOURCE.read_text(),
            "copy claude-code/skills/setup/scripts/framing.py here",
        )


if __name__ == "__main__":
    unittest.main()
