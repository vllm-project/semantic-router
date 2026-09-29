"""CPU tests for ops/m6/m6_finalists.py with the real 9B rule module (synthetic dev_readout files)."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
SCRIPT = HERE.parents[1] / "ops" / "m6" / "m6_finalists.py"
RULES = HERE.parents[2] / "9b" / "lux9b" / "m4_rules.py"
_spec = importlib.util.spec_from_file_location("m6_finalists", SCRIPT)
fin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fin)

N = 100  # items per family and per type


def arm(fam=(60, 60), typed=(60, 60, 60), h3=0.50, proxy=60.0):
    return {
        "T": sum(fam) / (N * len(fam)),
        "H": h3,
        "H_mean": h3,
        "proxy": proxy,
        "by_family": {f"f{i}": {"correct": c, "n": N} for i, c in enumerate(fam)},
        "by_type": {
            t: {"correct": c, "n": N, "invalid": 0}
            for t, c in zip(("choice", "noul", "score"), typed)
        },
    }


REF = arm()


def gain(g_items, h3=0.50, proxy=60.0, typed=(60, 60, 60)):
    """A point whose family macro is T_I + g_items / N."""
    return arm(fam=(60 + g_items, 60 + g_items), typed=typed, h3=h3, proxy=proxy)


class FinalistsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "lines" / "4b"
        (self.root / "readout").mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def line(self, tier, name, steps, arms):
        """steps: [(step, suffix)], arms: {suffix: arm}."""
        points = {f"{tier}-{name}-{sfx}": arms[sfx] for _s, sfx in steps}
        doc = {"arms": {f"{tier}-I": REF, **points}}
        (self.root / "readout" / f"L-{name}.json").write_text(json.dumps(doc))
        spec = f"L-{name}:" + ",".join(f"{s}={tier}-{name}-{sfx}" for s, sfx in steps)
        (self.root / "readout" / f"L-{name}.line").write_text(spec + "\n")
        for p in points:
            (self.root / p).mkdir(parents=True, exist_ok=True)
            (self.root / p / "weights.json").write_text(
                json.dumps(
                    {
                        "checkpoint": f"/ck/{p}",
                        "effective_weights": {"I": "1/2", "A": "1/2"},
                        "files_sha256_list": f"/ck/{p}.sha256",
                        "files_sha256_list_sha256": "ab" * 32,
                    }
                )
            )

    def arm_line(self, tier, name, g, **kw):
        steps = [("1/3", "b1_3"), ("1/2", "b1_2"), ("2/3", "b2_3"), ("1", "b1")]
        self.line(
            tier, name, steps, {s: gain(x, **kw) for (_st, s), x in zip(steps, g)}
        )

    def run_fin(self, tier, *extra):
        out = self.root.parent.parent / "select" / f"{tier}-finalists.json"
        fin.main(
            [
                "--tier",
                tier,
                "--lines-root",
                str(self.root),
                "--rules",
                str(RULES),
                "--output",
                str(out),
                *extra,
            ]
        )
        return json.loads(out.read_text())

    def test_4b_slots_and_slot3_better_of(self):
        self.arm_line(
            "4b", "N6D", [1, 2, 4, 4]
        )  # G* .04 -> smallest with G >= .03: 2/3
        self.arm_line("4b", "N6A", [0, 0, 0, 0])  # G* 0 -> no pick
        self.line(
            "4b",
            "N5BN",
            [("1/3", "b1_3"), ("1/2", "b1_2")],
            {"b1_3": gain(2), "b1_2": gain(3, proxy=61)},
        )
        self.line(
            "4b",
            "Nox",
            [("1/6", "g1_6"), ("1/3", "g1_3")],
            {"g1_6": gain(3, proxy=62), "g1_3": gain(1)},
        )
        out = self.run_fin("4b")
        got = [(f["slot"], f["line"], f["point"], f["step"]) for f in out["finalists"]]
        # N6A has no pick, so its slot passes on: N6D, then the better of N5BN / Nox (tie on G .03 -> higher P =
        # Nox at 62), then the other.
        self.assertEqual(
            got,
            [
                (1, "L-N6D", "4b-N6D-b2_3", "2/3"),
                (2, "L-Nox", "4b-Nox-g1_6", "1/6"),
                (3, "L-N5BN", "4b-N5BN-b1_2", "1/2"),
            ],
        )
        self.assertEqual(out["slot_note"]["better"], "L-Nox")
        self.assertEqual(out["slot_note"]["by"], "higher G, then higher P")
        self.assertIn(
            {"line": "L-N6A", "reason": "no pick: G* < 0.01"}, out["not_finalists"]
        )
        self.assertEqual(out["finalists"][0]["checkpoint"], "/ck/4b-N6D-b2_3")
        self.assertTrue(Path(out["rules_output"]).is_file())

    def test_eligibility_proxy_drop_and_pn1(self):
        # N6D: the only gaining point loses a type by more than 3% of n -> not eligible -> no pick.
        self.arm_line("4b", "N6D", [0, 0, 0, 5], typed=(60, 60, 56))
        self.arm_line("4b", "N6A", [2, 2, 2, 2], proxy=70.0)
        self.line(
            "4b",
            "N6P",
            [("1/3", "b1_3"), ("1/2", "b1_2"), ("2/3", "b2_3"), ("1", "b1")],
            {s: gain(1, proxy=61.0) for s in ("b1_3", "b1_2", "b2_3", "b1")},
        )
        self.line(
            "4b",
            "N5BN",
            [("1/3", "b1_3"), ("1/2", "b1_2")],
            {"b1_3": gain(3), "b1_2": gain(3)},
        )
        self.line(
            "4b",
            "Nox",
            [("1/6", "g1_6"), ("1/3", "g1_3")],
            {"g1_6": gain(-1), "g1_3": gain(-2)},
        )
        out = self.run_fin("4b")
        # Every point of a line shares one arm() proxy; N5BN (60) and N6P (61) are >= 8 below N6A (70) -> dropped.
        self.assertEqual(sorted(out["proxy_drop"]["dropped"]), ["L-N5BN", "L-N6P"])
        self.assertEqual([f["line"] for f in out["finalists"]], ["L-N6A"])
        self.assertEqual(out["order"][:3], ["L-N6D", "L-N6A", "L-N6P"])
        reasons = {x["line"]: x["reason"] for x in out["not_finalists"]}
        self.assertEqual(reasons["L-N6D"], "no pick: no eligible alpha")
        self.assertEqual(reasons["L-N6P"], "pick removed by the proxy drop rule")
        self.assertEqual(reasons["L-Nox"], "no pick: G* < 0.01")

    def test_h3_floor(self):
        self.arm_line("4b", "N6D", [3, 3, 3, 3], h3=0.49)
        for name in ("N6A",):
            self.arm_line("4b", name, [1, 1, 1, 1])
        out = self.run_fin(
            "4b", "--dropped", "L-N5BN=not built", "--dropped", "L-Nox=test"
        )
        self.assertEqual(out["lines"]["L-N6D"]["no_pick_reason"], "no eligible alpha")
        self.assertEqual([f["point"] for f in out["finalists"]], ["4b-N6A-b1_3"])
        self.assertEqual(out["dropped_lines"], {"L-N5BN": "not built", "L-Nox": "test"})

    def test_missing_line_must_be_declared(self):
        self.arm_line("4b", "N6D", [1, 2, 3, 4])
        with self.assertRaises(SystemExit):
            self.run_fin("4b")

    def test_08b_two_slots_and_2b_order(self):
        self.root = Path(self.tmp.name) / "lines" / "08b"
        (self.root / "readout").mkdir(parents=True)
        self.arm_line("08b", "E6K", [1, 1, 1, 1])
        self.line(
            "08b",
            "Eos",
            [("1/6", "g1_6"), ("1/3", "g1_3")],
            {"g1_6": gain(2), "g1_3": gain(2)},
        )
        out = self.run_fin("08b")
        self.assertEqual(
            [(f["line"], f["point"]) for f in out["finalists"]],
            [("L-E6K", "08b-E6K-b1_3"), ("L-Eos", "08b-Eos-g1_6")],
        )
        self.assertEqual(fin.TIERS["2b"]["lines"], ["L-S6X", "L-S6D", "L-Sol"])

    def test_slots_full(self):
        rules = {
            "lines": {
                l: {
                    "pick": {
                        "arm": l,
                        "alpha": "1",
                        "T": 0.6,
                        "G": 0.02,
                        "H3": 0.5,
                        "proxy": 60.0,
                    },
                    "no_pick_reason": None,
                }
                for l in ("L-S6X", "L-S6D", "L-Sol")
            },
            "proxy_drop": {"dropped": []},
        }
        out = fin.select("2b", rules, {})
        self.assertEqual(
            [f["line"] for f in out["finalists"]], ["L-S6X", "L-S6D", "L-Sol"]
        )
        rules["lines"]["L-E6K"] = rules["lines"].pop("L-S6X")
        rules["lines"]["L-Eos"] = rules["lines"].pop("L-S6D")
        rules["lines"]["L-X"] = rules["lines"].pop("L-Sol")
        out = fin.select("08b", rules, {})
        self.assertEqual(len(out["finalists"]), 2)


if __name__ == "__main__":
    unittest.main()
