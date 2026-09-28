from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def _build(arm: str, seed: str, hashseed: str, out: Path) -> dict[str, str]:
    env = dict(os.environ, PYTHONHASHSEED=hashseed)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "v2.data.verifiable.build",
            "--arm",
            arm,
            "--seed",
            seed,
            "--out-dir",
            str(out),
            "--groups-per-family",
            "24",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
    )
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.glob("*.jsonl"))
    }


class HashSeedIndependenceTest(unittest.TestCase):
    def test_builds_do_not_depend_on_pythonhashseed(self) -> None:
        for arm, seed in (
            ("a2", "hs-a2"),
            ("a6", "hs-a6"),
            ("a4", "hs-a4"),
            ("a4v2", "hs-a4v2"),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                first = _build(arm, seed, "0", Path(tmp) / "h0")
                second = _build(arm, seed, "12345", Path(tmp) / "h1")
            self.assertEqual(first, second, arm)


if __name__ == "__main__":
    unittest.main()
