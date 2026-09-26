"""Run the self-contained Decision 2.0 contracts from their package root."""

import subprocess
import sys
import unittest
from pathlib import Path


class Decision2ContractsTest(unittest.TestCase):
    def test_decision2_contracts(self) -> None:
        root = Path(__file__).resolve().parents[1] / "decision2"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                ".",
                "-p",
                "test_*.py",
            ],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
