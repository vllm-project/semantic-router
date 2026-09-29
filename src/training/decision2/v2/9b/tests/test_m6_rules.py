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

from lux9b import m6_rules  # noqa: E402

A13, A12, A23 = Fraction(1, 3), Fraction(1, 2), Fraction(2, 3)
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


REF = arm(799, 338, 343, [400, 338, 343, 399], 0.5622, proxy=73.22)


def line(points, arms):
    return m6_rules.alpha_line(points, {"ref": REF, **arms}, "ref")


class NoulFloorTest(unittest.TestCase):
    def test_rule_precedence_floor_is_stricter_than_the_type_floor(self):
        on = arm(799, 334, 380, [400, 334, 380, 399], 0.57)  # 338 - 4 = 334
        below = arm(
            799, 333, 380, [400, 333, 380, 399], 0.57
        )  # passes the type floor 326
        out = line([(A12, "on"), (A23, "below")], {"on": on, "below": below})
        self.assertEqual([r["eligible"] for r in out["rows"]], [True, False])
        self.assertEqual(out["rows"][1]["reasons"], ["rule_precedence 333 < floor 334"])

    def test_m5_floors_still_apply(self):
        low_h3 = arm(799, 350, 380, [400, 350, 380, 399], 0.5621)
        low_choice = arm(774, 350, 380, [400, 350, 380, 399], 0.57)
        out = line([(A13, "h"), (A12, "c")], {"h": low_h3, "c": low_choice})
        self.assertEqual([r["eligible"] for r in out["rows"]], [False, False])

    def test_pick_smallest_alpha_with_three_quarters_of_best_gain(self):
        # T_ref = (400+338+343+399)/1600 = .925
        a13 = arm(799, 340, 350, [400, 340, 350, 399], 0.57)  # G = 9/1600
        a12 = arm(800, 350, 370, [400, 350, 370, 400], 0.57)  # G = 40/1600
        a23 = arm(800, 360, 380, [400, 360, 380, 400], 0.57)  # G = 60/1600
        out = line([(A13, "x"), (A12, "y"), (A23, "z")], {"x": a13, "y": a12, "z": a23})
        self.assertAlmostEqual(out["G_star"], 60 / 1600)
        self.assertEqual(out["pick"]["alpha"], "2/3")  # 40 < 45 = 3/4 * 60
        self.assertEqual(out["pick"]["rule_precedence"], 360)

    def test_no_pick_below_min_gain(self):
        small = arm(799, 340, 350, [400, 340, 350, 399], 0.57)  # G = 9/1600 < .01
        out = line([(A13, "s")], {"s": small})
        self.assertIsNone(out["pick"])
        self.assertEqual(out["no_pick_reason"], "G* < 0.01")

    def test_proxy_drop(self):
        good = arm(800, 360, 380, [400, 360, 380, 400], 0.57, proxy=65.0)
        out = line([(A12, "g")], {"g": good})
        self.assertIsNotNone(out["rule_pick"])
        self.assertTrue(out["dropped"])
        self.assertIsNone(out["pick"])


class EarlyStopTest(unittest.TestCase):
    CTL = arm(799, 330, 340, [400, 330, 340, 399], 0.560, proxy=72.0)

    def run_early(self, a, protect):
        return m6_rules.early({"a": a, "k": self.CTL}, "a", "k", protect)

    def test_continue_needs_half_a_proxy_point_and_the_screen(self):
        ok = arm(799, 330, 340, [400, 330, 340, 399], 0.560, proxy=72.5)
        self.assertTrue(self.run_early(ok, "H3")["continue"])
        short = arm(799, 330, 340, [400, 330, 340, 399], 0.570, proxy=72.49)
        out = self.run_early(short, "H3")
        self.assertFalse(out["continue"])
        self.assertIn("P gain", out["reasons"][0])

    def test_protected_screens(self):
        low_h3 = arm(799, 340, 340, [400, 340, 340, 399], 0.5599, proxy=73.0)
        self.assertFalse(self.run_early(low_h3, "H3")["continue"])
        self.assertTrue(self.run_early(low_h3, "RP")["continue"])
        low_rp = arm(799, 329, 340, [400, 329, 340, 399], 0.60, proxy=73.0)
        self.assertTrue(self.run_early(low_rp, "H3")["continue"])
        out = self.run_early(low_rp, "RP")
        self.assertFalse(out["continue"])
        self.assertEqual(out["reasons"], ["RP 329 < control 330"])


class CliTest(unittest.TestCase):
    def test_alpha_and_finalists_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            readout = tmp / "readout.json"
            y = arm(800, 350, 370, [400, 350, 370, 400], 0.57)
            readout.write_text(json.dumps({"arms": {"ref": REF, "y": y}}))
            alpha = tmp / "alpha.json"
            with contextlib.redirect_stdout(io.StringIO()):
                m6_rules.main(
                    [
                        "alpha",
                        "--readout",
                        str(readout),
                        "--name",
                        "KA",
                        "--ref",
                        "ref",
                        "--point",
                        "1/2=y",
                        "--output",
                        str(alpha),
                    ]
                )
                m6_rules.main(
                    [
                        "finalists",
                        "--line",
                        f"KA={alpha}",
                        "--line",
                        f"KH={alpha}",
                        "--output",
                        str(tmp / "fin.json"),
                    ]
                )
            fin = json.loads((tmp / "fin.json").read_text())
            self.assertEqual([f["line"] for f in fin["finalists"]], ["KA", "KH"])
            self.assertEqual(json.loads(alpha.read_text())["pick"]["alpha"], "1/2")


if __name__ == "__main__":
    unittest.main()
