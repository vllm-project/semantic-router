"""CPU tests for ops/m7/m7_rules.py (synthetic development readouts, the real 9B eligibility module)."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "m7_rules", HERE / "ops" / "m7" / "m7_rules.py"
)
mr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mr)
RULES = HERE.parent / "9b" / "lux9b" / "m4_rules.py"


def arm(c=500, n=270, s=360, fam=0.7, h3=0.56, t=0.70, proxy=61.0):
    return {
        "by_type": {
            "choice": {"correct": c, "n": 800},
            "noul": {"correct": n, "n": 400},
            "score": {"correct": s, "n": 400},
        },
        "by_family": {
            "f1": {"correct": int(fam * 100), "n": 100},
            "f2": {"correct": 70, "n": 100},
        },
        "H_mean": h3,
        "H": h3 - 0.01,
        "T": t,
        "proxy": proxy,
    }


class RulesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "lines"
        (self.root / "readout").mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def line(self, name, points):
        arms = {"4b-I": arm()}
        spec = []
        for step, (point, a) in points.items():
            arms[point] = a
            spec.append(f"{step}={point}")
            d = self.root / point
            (d / "dev").mkdir(parents=True)
            (d / "css-pilot").mkdir()
            (d / "weights.json").write_text(
                json.dumps(
                    {
                        "checkpoint": f"/ckpt/{point}",
                        "effective_weights": {"A": step},
                        "files_sha256_list": f"/l/{point}",
                        "files_sha256_list_sha256": "cd" * 32,
                    }
                )
            )
        (self.root / "readout" / f"{name}.json").write_text(json.dumps({"arms": arms}))
        (self.root / "readout" / f"{name}.line").write_text(
            f"{name}:" + ",".join(spec) + "\n"
        )

    def test_largest_passing_step_and_slots(self):
        self.line(
            "L-N7H",
            {
                "1/3": ("4b-N7H-b1_3", arm(h3=0.57)),
                "1/2": ("4b-N7H-b1_2", arm(h3=0.565)),
                "1": ("4b-N7H-b1", arm(h3=0.55)),
            },
        )
        self.line(
            "L-N7P",
            {
                "1/3": ("4b-N7P-b1_3", arm(n=200)),
                "1/2": ("4b-N7P-b1_2", arm(n=200)),
                "1": ("4b-N7P-b1", arm(n=200)),
            },
        )
        self.line(
            "L-N7C",
            {
                "1/3": ("4b-N7C-b1_3", arm()),
                "1/2": ("4b-N7C-b1_2", arm()),
                "1": ("4b-N7C-b1", arm(fam=0.55)),
            },
        )
        out = Path(self.tmp.name) / "select" / "4b-finalists.json"
        mr.main(
            [
                "--tier",
                "4b",
                "--lines-root",
                str(self.root),
                "--rules-module",
                str(RULES),
                "--output",
                str(out),
            ]
        )
        doc = json.loads(out.read_text())
        self.assertEqual(
            [(f["slot"], f["line"], f["point"]) for f in doc["finalists"]],
            [(1, "L-N7H", "4b-N7H-b1_2"), (2, "L-N7C", "4b-N7C-b1_2")],
        )
        self.assertEqual(doc["not_finalists"][0]["line"], "L-N7P")
        self.assertIn("checkpoint", doc["finalists"][0])

    def test_proxy_drop_and_dropped_line(self):
        self.line("L-N7H", {"1": ("4b-N7H-b1", arm(proxy=70.0))})
        self.line("L-N7C", {"1": ("4b-N7C-b1", arm(proxy=61.0))})
        out = Path(self.tmp.name) / "s" / "4b-finalists.json"
        mr.main(
            [
                "--tier",
                "4b",
                "--lines-root",
                str(self.root),
                "--rules-module",
                str(RULES),
                "--output",
                str(out),
                "--dropped",
                "L-N7P=arm stopped by a stop rule",
            ]
        )
        doc = json.loads(out.read_text())
        self.assertEqual([f["point"] for f in doc["finalists"]], ["4b-N7H-b1"])
        self.assertEqual(doc["proxy_dropped"], {"L-N7C": "4b-N7C-b1"})
        with self.assertRaises(SystemExit):
            mr.main(
                [
                    "--tier",
                    "4b",
                    "--lines-root",
                    str(self.root),
                    "--rules-module",
                    str(RULES),
                    "--output",
                    str(out),
                ]
            )


if __name__ == "__main__":
    unittest.main()
