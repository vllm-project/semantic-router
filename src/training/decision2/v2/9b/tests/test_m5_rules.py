import contextlib
import io
import json
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

from lux9b import m5_rules  # noqa: E402

A13, A12, A23, A1 = Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), Fraction(1)
FAMS = ("attribute_gate", "rule_precedence", "set_reconciliation", "transition_table")


def arm(choice, noul, score, fams, h3, h=0.58, proxy=None):
    t = sum(c / 400 for c in fams) / len(fams)
    return {
        "by_type": {
            "choice": {"correct": choice, "n": 800},
            "noul": {"correct": noul, "n": 400},
            "score": {"correct": score, "n": 400},
        },
        "by_family": {f: {"correct": c, "n": 400} for f, c in zip(FAMS, fams)},
        "T": t,
        "H_mean": h3,
        "H": h,
        "proxy": 100 * (t * h) ** 0.5 if proxy is None else proxy,
    }


REF = arm(799, 338, 343, [400, 338, 343, 399], 0.5622, proxy=73.0)


def line(points, arms):
    return m5_rules.alpha_line(points, {"ref": REF, **arms}, "ref")


class EligibilityTest(unittest.TestCase):
    def test_type_floor_equality_is_eligible(self):
        on = arm(775, 350, 343, [400, 350, 343, 399], 0.57)  # 799 - 24 = 775
        below = arm(774, 350, 343, [400, 350, 343, 399], 0.57)
        out = line([(A12, "on"), (A23, "below")], {"on": on, "below": below})
        self.assertEqual([r["eligible"] for r in out["rows"]], [True, False])
        self.assertIn("type choice", out["rows"][1]["reasons"][0])

    def test_score_and_noul_floors(self):
        low_score = arm(799, 350, 330, [400, 350, 330, 399], 0.57)  # floor 331
        low_noul = arm(799, 325, 350, [400, 325, 350, 399], 0.57)  # floor 326
        out = line([(A12, "s"), (A1, "n")], {"s": low_score, "n": low_noul})
        self.assertEqual([r["eligible"] for r in out["rows"]], [False, False])

    def test_h3_floor(self):
        equal = arm(799, 350, 343, [400, 350, 343, 399], 0.5622)
        low = arm(799, 350, 343, [400, 350, 343, 399], 0.5621)
        out = line([(A12, "e"), (A1, "l")], {"e": equal, "l": low})
        self.assertEqual([r["eligible"] for r in out["rows"]], [True, False])
        self.assertIn("H3", out["rows"][1]["reasons"][0])

    def test_family_floor_exact_at_boundary(self):
        on = arm(799, 350, 343, [400, 350, 343, 359], 0.57)  # 399/400 - 1/10 = 359/400
        below = arm(799, 350, 343, [400, 350, 343, 358], 0.57)
        out = line([(A13, "on"), (A12, "below")], {"on": on, "below": below})
        self.assertEqual([r["eligible"] for r in out["rows"]], [True, False])

    def test_family_floor_float_fallback(self):
        ref = {**REF, "by_family": {"f": {"accuracy": 0.9}}}
        ok = {**REF, "by_family": {"f": {"accuracy": 0.8}}}
        low = {**REF, "by_family": {"f": {"accuracy": 0.7999}}}
        self.assertTrue(m5_rules.eligibility(ok, ref)["eligible"])
        self.assertFalse(m5_rules.eligibility(low, ref)["eligible"])


class PickTest(unittest.TestCase):
    def test_no_eligible_alpha(self):
        bad = arm(700, 350, 343, [400, 350, 343, 399], 0.57)
        out = line([(A12, "b")], {"b": bad})
        self.assertIsNone(out["pick"])
        self.assertEqual(out["no_pick_reason"], "no eligible alpha")

    def test_no_pick_when_gain_below_one_point(self):
        tiny = arm(799, 353, 343, [400, 353, 343, 399], 0.57)  # G = 15/1600 < .01
        edge = arm(799, 354, 343, [400, 354, 343, 399], 0.57)  # G = 16/1600 = .01
        out = line([(A12, "t")], {"t": tiny})
        self.assertIsNone(out["pick"])
        self.assertEqual(out["no_pick_reason"], "G* < 0.01")
        self.assertEqual(line([(A12, "e")], {"e": edge})["pick"]["alpha"], "1/2")

    def test_smallest_alpha_keeping_three_quarters(self):
        # G in 1600ths: 1/3 -> 23, 1/2 -> 30, 2/3 -> 40, 1 -> 36 (G* = 40, 3/4 G* = 30)
        pts = {
            "a": arm(799, 361, 343, [400, 361, 343, 399], 0.57),
            "b": arm(799, 368, 343, [400, 368, 343, 399], 0.57),
            "c": arm(799, 378, 343, [400, 378, 343, 399], 0.57),
            "d": arm(799, 374, 343, [400, 374, 343, 399], 0.57),
        }
        out = line([(A13, "a"), (A12, "b"), (A23, "c"), (A1, "d")], pts)
        self.assertAlmostEqual(out["G_star"], 40 / 1600)
        self.assertEqual(out["pick"]["alpha"], "1/2")  # equality with 3/4 G* qualifies
        pts["b"] = arm(799, 367, 343, [400, 367, 343, 399], 0.57)  # 29 < 30
        out = line([(A13, "a"), (A12, "b"), (A23, "c"), (A1, "d")], pts)
        self.assertEqual(out["pick"]["alpha"], "2/3")

    def test_ineligible_best_gain_is_ignored(self):
        pts = {
            "a": arm(799, 361, 343, [400, 361, 343, 399], 0.57),  # G = 23/1600
            "d": arm(760, 390, 343, [400, 390, 343, 399], 0.57),  # choice below floor
        }
        out = line([(A13, "a"), (A1, "d")], pts)
        self.assertAlmostEqual(out["G_star"], 23 / 1600)
        self.assertEqual(out["pick"]["alpha"], "1/3")

    def test_proxy_drop_is_line_local_and_strict(self):
        at = arm(799, 370, 343, [400, 370, 343, 399], 0.57, proxy=65.0)  # 73 - 8
        below = arm(799, 370, 343, [400, 370, 343, 399], 0.57, proxy=64.99)
        out = line([(A12, "x")], {"x": at})
        self.assertFalse(out["dropped"])
        self.assertEqual(out["pick"]["alpha"], "1/2")
        out = line([(A12, "x")], {"x": below})
        self.assertTrue(out["dropped"])
        self.assertIsNone(out["pick"])
        self.assertEqual(out["rule_pick"]["alpha"], "1/2")

    def test_rejects_unknown_alpha(self):
        with self.assertRaises(ValueError):
            line([(Fraction(1, 4), "ref")], {})


class FinalistsTest(unittest.TestCase):
    def test_priority_order_skips_missing_and_dropped(self):
        pick = {
            "alpha": "1/2",
            "arm": "x",
            "T": 0.9,
            "G": 0.02,
            "H3": 0.57,
            "proxy": 74,
        }
        out = m5_rules.finalists(
            [
                ("KD", {"pick": None}),
                ("KG", {"pick": {**pick, "arm": "kg"}}),
                ("UM5", {"pick": {**pick, "arm": "um"}}),
            ]
        )
        self.assertEqual([f["line"] for f in out["finalists"]], ["KG", "UM5"])
        self.assertEqual(out["no_finalist"], ["KD"])
        many = [(f"L{i}", {"pick": pick}) for i in range(5)]
        self.assertEqual(len(m5_rules.finalists(many)["finalists"]), 3)


class SeedRuleTest(unittest.TestCase):
    @staticmethod
    def run_seed(soup, seeds, extra=()):
        arms = {"soup": {"proxy": soup}}
        arms.update({f"s{i}": {"proxy": p} for i, p in enumerate(seeds, 1)})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.json"
            path.write_text(json.dumps({"arms": arms}))
            out = Path(tmp) / "o.json"
            names = ",".join(f"s{i}" for i in range(1, len(seeds) + 1))
            with contextlib.redirect_stdout(io.StringIO()):
                m5_rules.main(
                    [
                        "seed",
                        "--readout",
                        str(path),
                        "--soup",
                        "soup",
                        "--seeds",
                        names,
                        "--output",
                        str(out),
                        *extra,
                    ]
                )
            return json.loads(out.read_text())["artifact"]

    def test_two_seeds(self):
        self.assertEqual(self.run_seed(70.0, [69.0, 71.0]), "soup")  # equal to mean
        self.assertEqual(self.run_seed(69.9, [69.0, 71.0]), "s1")
        self.assertEqual(self.run_seed(60.0, [65.0, 70.0], ["--primary", "s2"]), "s2")

    def test_three_seeds(self):
        self.assertEqual(self.run_seed(70.0, [69.0, 70.0, 71.0]), "soup")
        self.assertEqual(self.run_seed(60.0, [72.0, 65.0, 70.0]), "s3")


class KLineRegressionTest(unittest.TestCase):
    """M4 K line development readout (readout-lines/readout.json), R = the K 1/3 point."""

    ARMS = {
        "ka13": arm(
            799,
            338,
            343,
            [400, 338, 343, 399],
            0.5621901602795478,
            0.5795371486683455,
            73.21692854239514,
        ),
        "ka12": arm(
            793,
            363,
            346,
            [400, 363, 346, 393],
            0.5747967647712683,
            0.5829175152179625,
            73.97390197974299,
        ),
        "ka23": arm(
            776,
            368,
            346,
            [400, 368, 346, 376],
            0.5706896254436632,
            0.5677566527789677,
            72.7133676087426,
        ),
        "ka1": arm(
            704,
            309,
            340,
            [400, 309, 340, 304],
            0.5671055580799272,
            0.5345460288482933,
            67.23284060969297,
        ),
        "lux": arm(
            799,
            272,
            331,
            [400, 272, 331, 399],
            0.5283085874397198,
            0.5723757434179543,
            70.81978856011803,
        ),
    }
    POINTS = [(A13, "ka13"), (A12, "ka12"), (A23, "ka23"), (A1, "ka1")]

    def test_k_line_against_k13(self):
        out = m5_rules.alpha_line(self.POINTS, self.ARMS, "ka13")
        rows = {r["alpha"]: r for r in out["rows"]}
        self.assertEqual(
            [rows[a]["eligible"] for a in ("1/3", "1/2", "2/3", "1")],
            [True, True, True, False],
        )
        self.assertEqual(  # choice and noul floors, transition_table family floor
            [r.split()[1] for r in rows["1"]["reasons"]],
            ["choice", "noul", "transition_table"],
        )
        self.assertAlmostEqual(out["G_star"], 0.01375)
        self.assertEqual(out["pick"]["alpha"], "1/2")
        self.assertFalse(out["dropped"])

    def test_cli_with_lux_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "readout.json"
            path.write_text(json.dumps({"arms": self.ARMS}))
            out = Path(tmp) / "alpha.json"
            argv = [
                "alpha",
                "--readout",
                str(path),
                "--name",
                "K",
                "--ref",
                "ka13",
                "--lux",
                "lux",
                "--output",
                str(out),
            ]
            for a, k in self.POINTS:
                argv += ["--point", f"{a}={k}"]
            with contextlib.redirect_stdout(io.StringIO()):
                m5_rules.main(argv)
            data = json.loads(out.read_text())
        self.assertEqual(data["pick"]["arm"], "ka12")
        self.assertEqual(data["m4_lux_rule_report_only"]["pick"]["alpha"], "1/3")


if __name__ == "__main__":
    unittest.main()
