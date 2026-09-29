"""CPU tests for ops/m6/m6_successor.py (synthetic run directories)."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "ops" / "m6" / "m6_successor.py"
_spec = importlib.util.spec_from_file_location("m6_successor", SCRIPT)
suc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(suc)


def paired(delta, lo, hi, h_lo=-0.02, h_hi=0.04, right=60.0):
    return {
        "point": {
            "delta": {"score": delta, "T": 0.01, "H": 0.01},
            "left": {"score": right + delta},
            "right": {"score": right},
        },
        "ci95": {"low": lo, "high": hi},
        "axis_ci95": {
            "H": {"delta": {"low": h_lo, "high": h_hi}},
            "T": {"delta": {"low": 0, "high": 0.02}},
        },
    }


def reduced_pair(lo, h_hi=0.03):
    return {
        "v3": {
            "reduced": {
                "ci95": {"low": lo, "high": lo + 5},
                "axis_ci95": {"H": {"delta": {"low": -0.05, "high": h_hi}}},
            }
        }
    }


class SuccessorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def make(
        self,
        name,
        tier="4b",
        v3=64.0,
        bar=(1.0, 0.2, 3.0),
        own=(8.0, 5.0, 11.0),
        peers=None,
        slot=1,
        weights=None,
        collapsed=False,
        mlx_hi=0.01,
        red_bar=0.1,
        red_v3=None,
        exposure_groups=(),
        h_bar_hi=0.02,
    ):
        run = self.root / name
        run.mkdir()
        cfg = suc.TIERS[tier]
        (run / "REPORT.json").write_text(
            json.dumps(
                {
                    "v3": {"score": v3, "T": 0.7, "H": 0.58},
                    "panels": {
                        "public231": {
                            "correct": 150,
                            "tiers": {"easy": {"correct": 60, "items": 77}},
                        }
                    },
                }
            )
        )
        (run / "M6-RECEIPT.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "point": f"{tier}-X-b1_2",
                    "revision": "ab" * 32,
                    "selection": {
                        "slot": slot,
                        "effective_weights": weights or {"I": "1/2", "A": "1/2"},
                    },
                }
            )
        )
        (run / "PAIRED-vs-bar-t1.json").write_text(
            json.dumps(paired(*bar, h_hi=h_bar_hi, right=63.151))
        )
        (run / f"PAIRED-vs-{cfg['gate_own']}.json").write_text(
            json.dumps(paired(*own, right=56.47))
        )
        peers = peers or {
            n: (2.0, -1.0, 5.0, 62.0 - i) for i, n in enumerate(cfg["peers"])
        }
        for n, (d, lo, hi, right) in peers.items():
            (run / f"PAIRED-vs-{n}.json").write_text(
                json.dumps(paired(d, lo, hi, right=right))
            )
        types = {"types": {t: {"verdict": "OK"} for t in ("choice", "noul", "score")}}
        if collapsed:
            types["types"]["noul"]["verdict"] = "COLLAPSED: one answer (True) takes 95%"
        mlx = {
            "reference_name": "ref",
            "card_eligible": {"delta": -0.01, "ci95": {"low": -0.03, "high": mlx_hi}},
            "full": {"delta": 0.0, "ci95": {"low": -0.02, "high": 0.02}},
            "frozen_panel": True,
            "problems": [],
        }
        pairs = {
            f"{name} - bar-t1": reduced_pair(red_bar),
            f"{name} - {cfg['gate_own']}": reduced_pair(4.0),
        }
        for n in cfg["h_peers"]:
            pairs[f"{name} - {n}"] = reduced_pair(-2.0)
        overlap = {
            "pairs": pairs,
            "models": {name: {"reduced": {"v3": red_v3 if red_v3 is not None else v3}}},
            "reproduction": [{"match": True}],
            "problems": [],
        }
        exp = self.root / f"{name}-exposure.json"
        exp.write_text(json.dumps({"groups": list(exposure_groups)}))
        return suc.evaluate(
            tier, run, types, mlx, overlap, [(str(exp), json.loads(exp.read_text()))]
        )

    def test_pass(self):
        r = self.make("m6-4b-a")
        self.assertEqual(r["status"], "PASS", json.dumps(r["items"], indent=1))
        self.assertEqual(r["report_only"]["best_same_size_peer"], "decider4b")
        self.assertIn("none", r["report_only"]["inherited_exposure"])
        self.assertIn("m6-4b-a", suc.render(r))

    def test_each_item_can_fail(self):
        cases = {
            "1_v3_vs_bar": dict(bar=(0.5, -0.1, 2.0)),
            "2_H_vs_bar": dict(h_bar_hi=-0.001),
            "3_types": dict(collapsed=True),
            "4_mlx_card_eligible": dict(mlx_hi=-0.0001),
            "5_tier_gates": dict(v3=55.0),
            "6a_exposure_new_files": dict(exposure_groups=["g1"]),
            "6b_reduced_panels": dict(red_bar=-0.01),
        }
        for i, (key, kw) in enumerate(cases.items()):
            r = self.make(f"m6-4b-f{i}", **kw)
            self.assertEqual(r["status"], "FAIL", key)
            self.assertFalse(r["items"][key]["pass"], key)

    def test_h_vs_peer_gate_and_08b_has_no_floor(self):
        r = self.make(
            "m6-4b-h",
            peers={
                "decider4b": (1.0, -2.0, 4.0, 61.9),
                "jet62": (1.0, -2.0, 4.0, 60.4),
            },
        )
        self.assertTrue(r["items"]["5_tier_gates"]["H_vs_peers"]["jet62"]["pass"])
        run = self.root / "m6-4b-h"
        (run / "PAIRED-vs-jet62.json").write_text(
            json.dumps(paired(1.0, -2.0, 4.0, h_hi=-0.01))
        )
        r = suc.evaluate(
            "4b", run, {"types": {"choice": {"verdict": "OK"}}}, None, None, []
        )
        self.assertFalse(r["items"]["5_tier_gates"]["H_vs_peers"]["jet62"]["pass"])
        self.assertEqual(r["status"], "FAIL")
        r = self.make("m6-08b-a", tier="08b", v3=40.0)
        self.assertEqual(r["status"], "PASS")
        self.assertIn("33", r["report_only"]["inherited_exposure"])
        self.assertEqual(r["report_only"]["best_same_size_peer"], "intern")

    def test_incomplete_without_inputs(self):
        run = self.root / "m6-2b-x"
        run.mkdir()
        r = suc.evaluate("2b", run, None, None, None, [])
        self.assertEqual(r["status"], "INCOMPLETE")
        self.assertIn("bar-t1", r["missing"])

    def test_choose_tie_rule(self):
        a = self.make(
            "m6-4b-a",
            peers={
                "decider4b": (2.0, -1.0, 5.0, 61.9),
                "jet62": (1.0, -3.0, 4.0, 60.4),
            },
            bar=(1.0, 0.3, 3.0),
            slot=1,
        )
        b = self.make(
            "m6-4b-b",
            peers={
                "decider4b": (2.0, -0.5, 5.0, 61.9),
                "jet62": (1.0, -3.0, 4.0, 60.4),
            },
            bar=(1.0, 0.1, 3.0),
            slot=2,
        )
        c = self.make(
            "m6-4b-c",
            peers={
                "decider4b": (2.0, -0.5, 5.0, 61.9),
                "jet62": (1.0, -3.0, 4.0, 60.4),
            },
            bar=(1.0, 0.2, 3.0),
            slot=3,
        )
        d = self.make("m6-4b-d", bar=(1.0, -0.1, 3.0), slot=0)
        out = suc.choose("4b", [a, b, c, d])
        # b and c tie on the lower bound vs the best peer (-0.5); c has the higher lower bound vs the bar.
        self.assertEqual(out["order"], ["m6-4b-c", "m6-4b-b", "m6-4b-a"])
        self.assertEqual(out["successor"], "m6-4b-c")
        e = self.make(
            "m6-4b-e",
            peers={
                "decider4b": (2.0, -0.5, 5.0, 61.9),
                "jet62": (1.0, -3.0, 4.0, 60.4),
            },
            bar=(1.0, 0.2, 3.0),
            slot=1,
        )
        self.assertEqual(suc.choose("4b", [c, e])["successor"], "m6-4b-e")
        self.assertIsNone(suc.choose("4b", [d])["successor"])

    def test_cli(self):
        self.make("m6-2b-a", tier="2b", v3=54.0)
        run = self.root / "m6-2b-a"
        (self.root / "types.json").write_text(
            json.dumps({"types": {"choice": {"verdict": "OK"}}})
        )
        out = self.root / "res"
        self.assertEqual(
            suc.main(
                [
                    "evaluate",
                    "--tier",
                    "2b",
                    "--run",
                    str(run),
                    "--types",
                    str(self.root / "types.json"),
                    "--output",
                    str(out),
                ]
            ),
            0,
        )
        doc = json.loads(Path(f"{out}.json").read_text())
        self.assertEqual(doc["status"], "INCOMPLETE")
        self.assertTrue(Path(f"{out}.md").is_file())
        self.assertEqual(
            suc.main(
                [
                    "choose",
                    "--tier",
                    "2b",
                    "--result",
                    f"{out}.json",
                    "--output",
                    str(self.root / "choice"),
                ]
            ),
            0,
        )
        self.assertEqual(
            json.loads((self.root / "choice.json").read_text())["pending"], ["m6-2b-a"]
        )


if __name__ == "__main__":
    unittest.main()
