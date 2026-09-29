"""CPU tests for ops/m6b/m6b_finalists.py (synthetic rules output and line points)."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "ops" / "m6b" / "m6b_finalists.py"
_spec = importlib.util.spec_from_file_location("m6b_finalists", SCRIPT)
m6b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m6b)


class M6bFinalistsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        lines = self.root / "lines"
        rows = []
        for point, step in (("4b-N6D-b1", "1"), ("4b-N6D-b2_3", "2/3")):
            d = lines / point
            (d / "dev").mkdir(parents=True)
            (d / "css-pilot").mkdir()
            (d / "dev" / "dev.predictions.jsonl").write_text("{}\n")
            (d / "css-pilot" / "css-pilot.predictions.jsonl").write_text("{}\n")
            (d / "weights.json").write_text(
                json.dumps(
                    {
                        "checkpoint": f"/ckpt/{point}",
                        "effective_weights": {"A": step},
                        "files_sha256_list": f"/lists/{point}",
                        "files_sha256_list_sha256": "ab" * 32,
                    }
                )
            )
            rows.append(
                {
                    "alpha": step,
                    "arm": point,
                    "T": 0.72,
                    "G": 0.02,
                    "H3": 0.555,
                    "proxy": 62.5,
                    "eligible": False,
                    "reasons": ["H_mean 0.5550 < reference 0.5625"],
                }
            )
        self.rules = self.root / "4b-rules.json"
        self.rules.write_text(json.dumps({"lines": {"L-N6D": {"rows": rows}}}))
        self.lines = lines

    def tearDown(self):
        self.tmp.cleanup()

    def test_slots_follow_the_given_order(self):
        out = self.root / "select" / "4b-finalists.json"
        m6b.main(
            [
                "--rules",
                str(self.rules),
                "--lines-root",
                str(self.lines),
                "--point",
                "4b-N6D-b1",
                "--point",
                "4b-N6D-b2_3",
                "--output",
                str(out),
            ]
        )
        doc = json.loads(out.read_text())
        self.assertEqual(doc["tier"], "4b")
        self.assertEqual(
            [(f["slot"], f["point"], f["line"], f["step"]) for f in doc["finalists"]],
            [(1, "4b-N6D-b1", "L-N6D", "1"), (2, "4b-N6D-b2_3", "L-N6D", "2/3")],
        )
        f = doc["finalists"][1]
        for key in (
            "checkpoint",
            "weights_json",
            "files_sha256_list",
            "typed_dev_predictions",
        ):
            self.assertIn(key, f)
        self.assertFalse(f["m6_eligible"])
        with self.assertRaises(SystemExit):
            m6b.main(
                [
                    "--rules",
                    str(self.rules),
                    "--lines-root",
                    str(self.lines),
                    "--point",
                    "4b-N6D-b1",
                    "--output",
                    str(out),
                ]
            )

    def test_unknown_point_is_refused(self):
        with self.assertRaises(SystemExit):
            m6b.finalists(
                json.loads(self.rules.read_text()), self.lines, ["4b-N6D-b1_3"]
            )


if __name__ == "__main__":
    unittest.main()
