import json
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

from lux9b import m7_rules  # noqa: E402

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


def screen(verdict="TIE", hop=0.0, clean_no=-0.05, noul_high=0.01, choice_high=0.01):
    htdev2 = {
        "htdev2": {
            "vs_reference": {"delta": -0.005, "ci95": [-0.02, 0.01], "verdict": verdict}
        }
    }
    pn1 = {
        "delta": {"hop": hop, "clean_no": clean_no, "all8": 0, "pawsx6": 0},
        "delta_ci95": {},
    }
    mlx = {
        "metrics": {
            k: {"diff": 0.0, "ci95": [-0.02, high]}
            for k, high in (
                ("noul_ml", noul_high),
                ("choice_ml", choice_high),
                ("score_ml", 0.01),
                ("noul_pred_yes_rate_macro", 0.01),
            )
        }
    }
    return m7_rules.screens(htdev2, pn1, mlx)


def line(points, arms, screens):
    return m7_rules.alpha_line(points, {"ref": REF, **arms}, "ref", screens)


class ScreenTest(unittest.TestCase):
    def test_each_screen_blocks(self):
        self.assertEqual(screen()["reasons"], [])
        self.assertEqual(screen(verdict="FLAG")["reasons"], ["HT-DEV v2 FLAG"])
        self.assertEqual(len(screen(hop=-0.031)["reasons"]), 1)
        self.assertEqual(screen(hop=-0.03)["reasons"], [])
        self.assertEqual(len(screen(clean_no=0.001)["reasons"]), 1)
        self.assertEqual(len(screen(noul_high=-0.001)["reasons"]), 1)
        self.assertEqual(len(screen(choice_high=-0.001)["reasons"]), 1)
        self.assertEqual(screen(verdict="GAIN", noul_high=0.0)["reasons"], [])


class AlphaTest(unittest.TestCase):
    def test_h3_is_report_only_and_screens_gate(self):
        low_h3 = arm(
            800, 350, 370, [400, 350, 370, 400], 0.50
        )  # H3 far below R: not a floor in M7
        out = line([(A12, "x")], {"x": low_h3}, {"x": screen()})
        self.assertTrue(out["rows"][0]["eligible"])
        self.assertEqual(out["pick"]["alpha"], "1/2")
        flagged = line([(A12, "x")], {"x": low_h3}, {"x": screen(verdict="FLAG")})
        self.assertFalse(flagged["rows"][0]["eligible"])
        self.assertIsNone(flagged["pick"])
        missing = line([(A12, "x")], {"x": low_h3}, {})
        self.assertEqual(missing["rows"][0]["reasons"], ["development screens missing"])

    def test_m6_floors_and_pick_rule_still_apply(self):
        rp_low = arm(799, 333, 380, [400, 333, 380, 399], 0.57)
        a12 = arm(800, 350, 370, [400, 350, 370, 400], 0.57)  # G = 40/1600
        a23 = arm(800, 360, 380, [400, 360, 380, 400], 0.57)  # G = 60/1600
        out = line(
            [(A13, "r"), (A12, "y"), (A23, "z")],
            {"r": rp_low, "y": a12, "z": a23},
            {k: screen() for k in "ryz"},
        )
        self.assertEqual([r["eligible"] for r in out["rows"]], [False, True, True])
        self.assertIn("rule_precedence 333 < floor 334", out["rows"][0]["reasons"])
        self.assertEqual(out["pick"]["alpha"], "2/3")  # 40 < 45 = 3/4 * 60
        blocked = line(
            [(A12, "y"), (A23, "z")],
            {"y": a12, "z": a23},
            {"y": screen(), "z": screen(clean_no=0.01)},
        )
        self.assertEqual(blocked["pick"]["alpha"], "1/2")


def pn1_fixture(tmp: Path):
    gold, a, b = [], [], []
    # group g0: hop (gold yes); g1: near ja (clean no); g2: near ko (noisy no); g3: name ru (noisy)
    specs = [
        ("g0", "pn-hop", "ja", True, True, True),
        ("g0", "pn-hop", "zh", True, True, False),
        ("g1", "pn-near", "ja", False, True, False),
        ("g1", "pn-twin", "de", False, True, True),
        ("g2", "pn-near", "ko", False, True, False),
        ("g3", "pn-name", "ru", False, True, False),
    ]
    for i, (group, fam, lang, value, ya, yb) in enumerate(specs):
        rid = f"pn1dev-{i}"
        gold.append(
            {
                "id": rid,
                "group_id": group,
                "language": lang,
                "task": f"pn1/{fam}",
                "gold": {"decision": {"type": "noul", "value": value}},
            }
        )
        a.append(
            {
                "id": rid,
                "answers": {"decision": {"type": "noul", "noul": 0.9 if ya else 0.1}},
            }
        )
        b.append(
            {
                "id": rid,
                "answers": {"decision": {"type": "noul", "noul": 0.9 if yb else 0.1}},
            }
        )
    paths = {}
    for name, rows in (("gold", gold), ("a", a), ("b", b)):
        paths[name] = tmp / f"{name}.jsonl"
        paths[name].write_text("".join(json.dumps(r) + "\n" for r in rows))
    return paths


class Pn1Test(unittest.TestCase):
    def test_clean_no_excludes_the_dropped_constructions(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = pn1_fixture(Path(tmp))
            rows = m7_rules.pn1_rows(p["gold"])
            ref = m7_rules.pn1_summary(rows, m7_rules.pn1_yes(p["a"]))
            cand = m7_rules.pn1_summary(rows, m7_rules.pn1_yes(p["b"]))
            self.assertEqual(ref["clean_no"], {"n": 2, "yes": 1.0})
            self.assertEqual(cand["clean_no"], {"n": 2, "yes": 0.5})
            self.assertEqual(cand["hop"], {"n": 2, "yes": 0.5})
            ci = m7_rules.pn1_compare(
                rows, m7_rules.pn1_yes(p["a"]), m7_rules.pn1_yes(p["b"]), reps=200
            )
            self.assertLessEqual(ci["clean_no"][0], ci["clean_no"][1])

    def test_early_rule(self):
        def pn(no, hop):
            return {
                "reference": "ref-ka13",
                "candidate": {"clean_no": {"yes": no}, "hop": {"yes": hop}},
            }

        go = m7_rules.early(pn(0.40, 0.97), pn(0.45, 0.99), 0.88, 0.89)
        self.assertTrue(go["continue"])
        weak = m7_rules.early(pn(0.44, 0.99), pn(0.45, 0.99), 0.89, 0.89)
        self.assertFalse(weak["continue"])
        hop = m7_rules.early(pn(0.30, 0.95), pn(0.45, 0.99), 0.89, 0.89)
        self.assertFalse(hop["continue"])
        typed = m7_rules.early(pn(0.30, 0.99), pn(0.45, 0.99), 0.869, 0.89)
        self.assertFalse(typed["continue"])


if __name__ == "__main__":
    unittest.main()
