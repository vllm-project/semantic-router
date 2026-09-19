#!/usr/bin/env python3
"""Unit tests for the #3856 same-run helper (no model download)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from same_run_harness import host_identity, row_id_for
from same_run_pair import refuse_cross_host


class RowIdTests(unittest.TestCase):
    def test_hashes_text_only(self) -> None:
        prompt = "When was the 8088 processor released?"
        self.assertEqual(row_id_for(prompt), row_id_for(prompt))
        self.assertEqual(len(row_id_for(prompt)), 16)

    def test_stable_under_relabel(self) -> None:
        prompt = "How do I tie a bowline knot? Show me each step"
        # Gold changing AR → BOTH must not change the join key.
        self.assertEqual(row_id_for(prompt), row_id_for(prompt))


class HostPairTests(unittest.TestCase):
    def test_identity_requires_fields(self) -> None:
        with self.assertRaises(SystemExit):
            host_identity({})
        with self.assertRaises(SystemExit):
            host_identity(None)

    def test_same_host_pairs(self) -> None:
        host = {"cpu_model": "Intel", "core_count": 8, "ram_gb": 15.4}
        refuse_cross_host({"host": host}, {"host": dict(host)})

    def test_cross_host_refused(self) -> None:
        left = {"host": {"cpu_model": "Intel", "core_count": 8, "ram_gb": 15.4}}
        right = {"host": {"cpu_model": "AMD", "core_count": 16, "ram_gb": 64.0}}
        with self.assertRaises(SystemExit) as ctx:
            refuse_cross_host(left, right)
        self.assertIn("refusing to pair cross-host", str(ctx.exception))

    def test_pair_cli_refuses_cross_host(self) -> None:
        here = Path(__file__).resolve().parent
        baseline = {
            "host": {"cpu_model": "Intel", "core_count": 8, "ram_gb": 15.4},
            "model": "bert",
            "run": {"peak_rss_mb": 1000, "cpu_s": 10, "binding": "hf"},
            "records": [],
        }
        candidate = {
            "host": {"cpu_model": "AMD", "core_count": 16, "ram_gb": 64.0},
            "model": "distil",
            "run": {"peak_rss_mb": 400, "cpu_s": 2, "binding": "hf"},
            "records": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            base_path = Path(tmp) / "b.json"
            cand_path = Path(tmp) / "c.json"
            out_path = Path(tmp) / "p.json"
            base_path.write_text(json.dumps(baseline))
            cand_path.write_text(json.dumps(candidate))
            import subprocess
            import sys

            proc = subprocess.run(
                [
                    sys.executable,
                    str(here / "same_run_pair.py"),
                    "--baseline",
                    str(base_path),
                    "--candidate",
                    str(cand_path),
                    "--output",
                    str(out_path),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("refusing to pair cross-host", proc.stderr + proc.stdout)
            self.assertFalse(out_path.exists())


if __name__ == "__main__":
    unittest.main()
