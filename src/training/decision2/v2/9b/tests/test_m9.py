from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lux9b import m9_data, m9_rules


def pn1(clean_no, lo, hop):
    return {
        "delta": {"clean_no": clean_no, "hop": hop, "pawsx6": 0.0, "all8": 0.0},
        "delta_ci95": {"clean_no": [lo, lo + 0.04], "hop": [hop - 0.01, hop + 0.01]},
        "candidate": {"clean_no": {"yes": 0.3}, "hop": {"yes": 0.98}},
        "reference_summary": {"clean_no": {"yes": 0.26}, "hop": {"yes": 0.98}},
    }


def hs1(point, ref):
    return {
        "families": {
            "hs1_unmet_condition": {
                "diagnostics": {
                    "X": {"f3_false_yes_rate": point},
                    "C0": {"f3_false_yes_rate": ref},
                }
            }
        }
    }


class YesBiasGuardTest(unittest.TestCase):
    def test_level_point_passes(self):
        _, reasons = m9_rules.yes_bias(
            pn1(0.01, -0.01, 0.0), hs1(0.15, 0.15), "X", "C0"
        )
        self.assertEqual(reasons, [])

    def test_margin_and_significance_both_bind(self):
        _, reasons = m9_rules.yes_bias(
            pn1(0.03, -0.005, 0.0), hs1(0.15, 0.15), "X", "C0"
        )
        self.assertEqual(len(reasons), 1)
        self.assertIn("> +0.02", reasons[0])
        _, reasons = m9_rules.yes_bias(
            pn1(0.015, 0.001, 0.0), hs1(0.15, 0.15), "X", "C0"
        )
        self.assertEqual(len(reasons), 1)
        self.assertIn("CI lower", reasons[0])

    def test_margin_edge_is_inclusive(self):
        _, reasons = m9_rules.yes_bias(
            pn1(0.02, -0.001, 0.0), hs1(0.15, 0.15), "X", "C0"
        )
        self.assertEqual(reasons, [])

    def test_hop_and_hs1(self):
        _, reasons = m9_rules.yes_bias(
            pn1(0.0, -0.02, -0.031), hs1(0.21, 0.15), "X", "C0"
        )
        self.assertEqual([r[:2] for r in reasons], ["Y2", "Y3"])
        _, reasons = m9_rules.yes_bias(
            pn1(0.0, -0.02, -0.03), hs1(0.20, 0.15), "X", "C0"
        )
        self.assertEqual(reasons, [])

    def test_missing_readouts_fail(self):
        _, reasons = m9_rules.yes_bias(None, None, "X", "C0")
        self.assertEqual(len(reasons), 2)


class MultilingualTest(unittest.TestCase):
    def test_upper_bounds(self):
        mlx = {
            "metrics": {
                "noul_ml": {"diff": -0.01, "ci95": [-0.02, -0.001]},
                "choice_ml": {"diff": 0.0, "ci95": [-0.01, 0.01]},
                "score_ml": {"diff": -0.05, "ci95": [-0.08, -0.02]},
            }
        }
        _, reasons = m9_rules.multilingual(mlx)
        self.assertEqual(len(reasons), 1)
        self.assertIn("noul_ml", reasons[0])
        self.assertEqual(m9_rules.multilingual(None)[1], ["MLX-DEV-9B readout missing"])


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


class Stage2BuildTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        x60 = self.tmp / "x60"
        x60.mkdir()
        write_jsonl(
            x60 / "train.jsonl",
            [
                {"id": "x1", "group_id": "gx", "input_sha256": "hx1"},
                {"id": "x2", "group_id": "gx", "input_sha256": "hx2"},
            ],
        )
        write_jsonl(x60 / "teacher.jsonl", [{"id": "x1"}, {"id": "x2"}])
        (x60 / "manifest.json").write_text(json.dumps({"recipe": {"tokens": 10}}))
        self.x60 = x60

    def ib(self, name, rows):
        path = self.tmp / f"{name}.train.jsonl"
        write_jsonl(path, rows)
        write_jsonl(
            self.tmp / f"{name}.train.tokens.jsonl",
            [{"id": r["id"], "native": 7} for r in rows],
        )
        return path

    def test_concatenates_excludes_and_keeps_x60_bytes(self):
        ib1 = self.ib(
            "ib1",
            [
                {"id": "a1", "group_id": "g1", "input_sha256": "h1", "family": "args"},
                {"id": "a2", "group_id": "g1", "input_sha256": "h2", "family": "isarc"},
            ],
        )
        ib2 = self.ib(
            "ib2",
            [{"id": "b1", "group_id": "g2", "input_sha256": "h3", "family": "hover"}],
        )
        out = self.tmp / "out"
        m9_data.main(
            [
                "--x60-dir",
                str(self.x60),
                "--ib1",
                str(ib1),
                "--ib2",
                str(ib2),
                "--exclude-family",
                "isarc",
                "--output",
                str(out),
            ]
        )
        lines = (out / "train.jsonl").read_text().splitlines()
        self.assertEqual([json.loads(x)["id"] for x in lines], ["x1", "x2", "a1", "b1"])
        self.assertEqual(
            (out / "teacher.jsonl").read_bytes(),
            (self.x60 / "teacher.jsonl").read_bytes(),
        )
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["ib_rows_dropped"], {"ib1/isarc": 1})
        self.assertEqual(manifest["ib_native_tokens_kept"], 14)

    def test_refuses_x60_group_or_repeated_input(self):
        ib1 = self.ib(
            "ib1",
            [{"id": "a1", "group_id": "gx", "input_sha256": "h1", "family": "args"}],
        )
        ib2 = self.ib(
            "ib2",
            [{"id": "b1", "group_id": "g2", "input_sha256": "h3", "family": "hover"}],
        )
        with self.assertRaises(ValueError):
            m9_data.main(
                [
                    "--x60-dir",
                    str(self.x60),
                    "--ib1",
                    str(ib1),
                    "--ib2",
                    str(ib2),
                    "--output",
                    str(self.tmp / "o1"),
                ]
            )
        ib1 = self.ib(
            "ib1",
            [{"id": "a1", "group_id": "g1", "input_sha256": "h3", "family": "args"}],
        )
        with self.assertRaises(ValueError):
            m9_data.main(
                [
                    "--x60-dir",
                    str(self.x60),
                    "--ib1",
                    str(ib1),
                    "--ib2",
                    str(ib2),
                    "--output",
                    str(self.tmp / "o2"),
                ]
            )


if __name__ == "__main__":
    unittest.main()
