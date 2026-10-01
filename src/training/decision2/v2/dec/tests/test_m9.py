"""CPU tests for decoder M9 tooling (ops/m9: compose, early rule, pick, HR2 DEV scoring helpers)."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

OPS = Path(__file__).resolve().parents[1] / "ops"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mc = load("m9_compose", OPS / "m9" / "m9_compose.py")
mr = load("m9_rules", OPS / "m9" / "m9_rules.py")
mh = load("m9_hr2dev", OPS / "m9" / "m9_hr2dev.py")


def row(i, group, source="src", task="noul", lang="en", family="fam", tag=""):
    return {
        "id": f"{tag}r{i}",
        "group_id": f"{tag}g{group}",
        "input_sha256": f"{tag}h{i}",
        "source": source,
        "task_type": task,
        "language": lang,
        "family": family,
        "state": "s",
    }


class ComposeTest(unittest.TestCase):
    def setUp(self):
        self.base = [row(i, i // 2) for i in range(200)]
        self.pool = list(self.base) + [
            row(i, i // 2, source=f"s{i % 3}", task=("choice", "noul")[i % 2])
            for i in range(200, 600)
        ]
        self.hr2 = [
            row(
                i,
                i // 2,
                source="helpsteer3_train",
                family=("hs3_pref", "vitc")[i % 2],
                tag="hr2-",
            )
            for i in range(80)
        ]
        self.length = {
            r["id"]: 10 + (int(r["id"].rsplit("r", 1)[1]) % 7)
            for r in self.pool + self.hr2
        }
        self.pool_of = {
            r["id"]: ("H7" if r["id"] in ("r400", "r401") else "V") for r in self.pool
        }

    def compose(self, hr2=None, quarantine=frozenset({"g0"})):
        return mc.compose(
            self.base,
            self.pool,
            self.hr2 if hr2 is None else hr2,
            self.length,
            quarantine=set(quarantine),
            pool_of=self.pool_of,
            gold_pools={"H7", "H8"},
        )

    def test_arms_share_the_base_and_match_tokens(self):
        arms, report = self.compose()
        base_ids = [r["id"] for r, p in arms["H9"] if p.startswith("base")]
        self.assertEqual(
            base_ids, [r["id"] for r, p in arms["C9"] if p.startswith("base")]
        )
        self.assertNotIn("r0", base_ids)  # quarantined group g0
        self.assertEqual(report["quarantine_rows_dropped"]["base"], 2)
        self.assertEqual({p for _, p in arms["H9"]} - {"base", "base-gold"}, {"hr2"})
        self.assertTrue(
            {p for _, p in arms["C9"]} <= {"base", "base-gold", "fill", "fill-gold"}
        )
        x = sum(self.length[r["id"]] for r in self.hr2)
        self.assertEqual(report["added_tokens_H9"], x)
        self.assertLess(abs(report["filler"]["tokens"] - x), 20)
        self.assertLess(report["match"]["max_relative"], 0.05)
        fill_ids = {r["id"] for r, p in arms["C9"] if p.startswith("fill")}
        self.assertFalse(fill_ids & set(base_ids))

    def test_gold_pool_rows_are_labelled(self):
        arms, _ = self.compose()
        labels = {r["id"]: p for r, p in arms["C9"]}
        for rid in ("r400", "r401"):
            if rid in labels:
                self.assertEqual(labels[rid], "fill-gold")

    def test_filler_grows_monotonically(self):
        _, small = self.compose(hr2=self.hr2[:40])
        _, large = self.compose()
        self.assertTrue(
            set(small["filler"]["group_ids"]) <= set(large["filler"]["group_ids"])
        )

    def test_hr2_shared_with_base_is_an_error(self):
        clash = dict(self.hr2[0], input_sha256="h5")
        with self.assertRaises(ValueError):
            self.compose(hr2=[clash] + self.hr2[1:])


class EarlyRuleTest(unittest.TestCase):
    def run_dir(self, root: Path, name: str, acc: float) -> Path:
        run = root / name
        (run / "ck").mkdir(parents=True)
        (run / "BEST.json").write_text(json.dumps({"checkpoint": "ck"}))
        (run / "ck" / "checkpoint.json").write_text(
            json.dumps({"dev_metrics": {"family_macro_accuracy": acc}})
        )
        return run

    def decide(self, h_acc, c_acc, d_hc, d_hi):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            h, c = self.run_dir(root, "h", h_acc), self.run_dir(root, "c", c_acc)
            pair = lambda d: {
                "delta": d,
                "ci95": [d - 0.01, d + 0.01],
                "verdict": "TIE",
                "H_dev2": {},
            }
            return mr.early(h, c, pair(d_hc), pair(d_hi))

    def test_continue_when_neither_rule_fires(self):
        self.assertEqual(self.decide(0.90, 0.91, -0.01, -0.01)["decision"], "continue")

    def test_typed_stop(self):
        out = self.decide(0.87, 0.905, 0.02, 0.01)
        self.assertTrue(out["E1_typed_stop"])
        self.assertEqual(out["decision"], "stop")

    def test_typed_margin(self):
        self.assertEqual(mr.E1_TYPED, Fraction(3, 100))
        self.assertEqual(self.decide(0.88, 0.905, 0.0, 0.0)["decision"], "continue")

    def test_human_stop_needs_both_conditions(self):
        self.assertEqual(self.decide(0.9, 0.9, -0.04, -0.01)["decision"], "continue")
        self.assertEqual(self.decide(0.9, 0.9, -0.02, -0.03)["decision"], "continue")
        self.assertEqual(self.decide(0.9, 0.9, -0.03, -0.02)["decision"], "stop")


class PickTest(unittest.TestCase):
    def rows(self, *specs):
        return [
            {"step": s, "point": f"p{s}", "eligible": e, "htdev2": {"verdict": v}}
            for s, e, v in specs
        ]

    def test_gain_first_then_larger_alpha(self):
        self.assertEqual(
            mr.pick(self.rows(("1", True, "TIE"), ("1/2", True, "GAIN")))["step"], "1/2"
        )
        self.assertEqual(
            mr.pick(self.rows(("1", True, "TIE"), ("1/2", True, "TIE")))["step"], "1"
        )
        self.assertEqual(
            mr.pick(self.rows(("1", False, "GAIN"), ("1/2", True, "TIE")))["step"],
            "1/2",
        )
        self.assertIsNone(
            mr.pick(self.rows(("1", False, "TIE"), ("1/2", False, "TIE")))
        )


class Hr2DevTest(unittest.TestCase):
    def test_gold_values(self):
        noul = {
            "id": "a",
            "task_type": "noul",
            "label": 1,
            "options": [
                {"key": "false", "description": "No"},
                {"key": "true", "description": "Yes"},
            ],
        }
        self.assertIs(mh.gold_value(noul), True)
        score = {
            "id": "b",
            "task_type": "score",
            "label": 3,
            "options": [{"key": str(k)} for k in range(5)],
        }
        self.assertEqual(mh.gold_value(score), 3)
        choice = {
            "id": "c",
            "task_type": "choice",
            "label": 0,
            "options": [{"key": "A"}, {"key": "B"}],
        }
        self.assertEqual(mh.gold_value(choice), "A")

    def test_summary_and_paired(self):
        gold = [
            {
                "id": f"i{k}",
                "family": ("f1", "f2")[k % 2],
                "task_type": "noul",
                "group_id": f"g{k}",
            }
            for k in range(40)
        ]
        left = {
            g["id"]: {"correct": True, "valid": True, "abs_error": None} for g in gold
        }
        right = {
            g["id"]: {"correct": k % 4 == 0, "valid": True, "abs_error": None}
            for k, g in enumerate(gold)
        }
        s = mh.summarize(gold, left)
        self.assertEqual(s["family_macro"], 1.0)
        p = mh.paired(gold, left, right)
        self.assertGreater(p["ci95"][0], 0)
        self.assertEqual(p["p_le_0"], 0.0)


if __name__ == "__main__":
    sys.exit(unittest.main())
