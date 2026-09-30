"""CPU tests for the decoder M8-small tools (ops/m8s): top-up partition and sampling, the A20r teacher split, the
label parity comparison, and the early / finalist rules."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, HERE / "ops" / "m8s" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mc = _load("m8s_compose")
ml = _load("m8s_label")
mk = _load("m8s_lock")
mr = _load("m8s_rules")


def row(i, source, group=None, task="choice", lang="en", **extra):
    return {
        "id": f"r{i}",
        "group_id": group or f"g{i}",
        "input_sha256": f"{i:064x}",
        "source": source,
        "task_type": task,
        "language": lang,
        "label": 0,
        "options": [{"key": "A"}, {"key": "B"}],
        **extra,
    }


class PartitionTest(unittest.TestCase):
    def test_typed_and_human_sources(self):
        self.assertEqual(mc.part_of(row(1, "dec10:generated_stage4_v2")), "typed")
        self.assertEqual(mc.part_of(row(2, "decision2_verifiable_v2_a6")), "typed")
        self.assertEqual(mc.part_of(row(3, "decision2_programmatic_new_v9")), "typed")
        self.assertEqual(mc.part_of(row(4, "dec10:generated_stage9")), "typed")
        self.assertEqual(
            mc.part_of(row(5, "legacy:stage4-general-composition-v2")), "typed"
        )
        self.assertEqual(
            mc.part_of(row(6, "google_goemotions_official_train")), "human"
        )
        self.assertEqual(mc.part_of(row(7, "dec10:multinli_nonfiction_train")), "human")
        self.assertEqual(mc.part_of(row(8, "legacy:stage3_replay")), "typed")
        self.assertEqual(
            mc.part_of(row(9, "legacy:stage3_replay", upstream_label="x")), "human"
        )


class ComposeTest(unittest.TestCase):
    def setUp(self):
        rows = []
        for i in range(40):
            rows.append(
                row(
                    i,
                    "dec10:generated_stage4_v2" if i % 2 else "squad2_train",
                    task="noul" if i % 3 else "choice",
                )
            )
        rows.append(row(100, "squad2_train", group="mixed"))
        rows.append(row(101, "dec10:generated_stage1_3", group="mixed"))
        rows.append(row(102, "squad2_train", group="quar"))
        self.rows = rows
        self.length = {r["id"]: 10 for r in rows}

    def run_compose(self, seed="s"):
        return mc.compose(
            self.rows,
            self.length,
            quarantine={"quar"},
            budgets={"human": 60, "typed": 60},
            seed=seed,
        )

    def test_budgets_parts_and_order(self):
        out, parts, rep = self.run_compose()
        self.assertEqual(rep["quarantined_rows"], 1)
        self.assertNotIn("r102", {r["id"] for r in out})
        self.assertEqual(rep["mixed_groups"]["count"], 1)
        for part in ("human", "typed"):
            self.assertGreaterEqual(rep["tokens_by_part"][part], 60)
            self.assertLessEqual(rep["tokens_by_part"][part], 90)
        order = [r["id"] for r in self.rows]
        self.assertEqual(
            [r["id"] for r in out], sorted((r["id"] for r in out), key=order.index)
        )
        groups = {}
        for r in out:
            groups.setdefault(r["group_id"], set()).add(parts[r["id"]])
        self.assertTrue(all(len(v) == 1 for v in groups.values()))

    def test_deterministic_and_seeded(self):
        a = [r["id"] for r in self.run_compose()[0]]
        self.assertEqual(a, [r["id"] for r in self.run_compose()[0]])
        self.assertNotEqual(a, [r["id"] for r in self.run_compose(seed="other")[0]])

    def test_budget_over_available(self):
        with self.assertRaises(ValueError):
            mc.compose(
                self.rows,
                self.length,
                quarantine=set(),
                budgets={"human": 10_000, "typed": 10},
                seed="s",
            )


class TeacherSplitTest(unittest.TestCase):
    def test_d1_all_d2_human(self):
        train = [row(1, "squad2_train"), row(2, "dec10:generated_stage4_v2")]
        parts = {"r1": "human", "r2": "typed"}
        labels = {(r["id"], r["input_sha256"]): {"A": 0.3, "B": 0.7} for r in train}
        out = mk.split_teachers(train, parts, labels)
        self.assertEqual([r["id"] for r in out["D1"]], ["r1", "r2"])
        self.assertEqual([r["id"] for r in out["D2"]], ["r1"])
        labels[("r2", train[1]["input_sha256"])] = {"A": 1.0}
        with self.assertRaises(ValueError):
            mk.split_teachers(train, parts, labels)
        del labels[("r2", train[1]["input_sha256"])]
        with self.assertRaises(ValueError):
            mk.split_teachers(train, parts, labels)


class ParityTest(unittest.TestCase):
    def test_compare(self):
        stored = [{"id": "a", "logits": [1.0, 2.0]}, {"id": "b", "logits": [0.5, -1.0]}]
        self.assertEqual(ml.compare_logits(stored, stored)["status"], "PASS")
        near = [
            {"id": "a", "logits": [1.00005, 2.0]},
            {"id": "b", "logits": [0.5, -1.0]},
        ]
        self.assertEqual(ml.compare_logits(near, stored)["status"], "PASS")
        far = [{"id": "a", "logits": [1.01, 2.0]}, {"id": "b", "logits": [0.5, -1.0]}]
        self.assertEqual(ml.compare_logits(far, stored)["status"], "FAIL")
        flip = [{"id": "a", "logits": [2.5, 2.0]}, {"id": "b", "logits": [0.5, -1.0]}]
        res = ml.compare_logits(flip, stored)
        self.assertEqual((res["status"], res["argmax_changes"]), ("FAIL", 1))
        with self.assertRaises(ValueError):
            ml.compare_logits(stored[:1], stored)

    def test_softmax(self):
        p = ml.softmax([0.0, 0.0])
        self.assertAlmostEqual(p[0], 0.5)


def point(c=500, n=270, s=300, fam=70, t=0.6, proxy=47.0):
    return {
        "by_type": {
            "choice": {"correct": c, "n": 800},
            "noul": {"correct": n, "n": 400},
            "score": {"correct": s, "n": 400},
        },
        "by_family": {
            "f1": {"correct": fam, "n": 100},
            "f2": {"correct": 70, "n": 100},
        },
        "T": t,
        "H_mean": 0.43,
        "H": 0.42,
        "proxy": proxy,
    }


class RulesTest(unittest.TestCase):
    def test_floors_and_gate(self):
        ref = point()
        self.assertEqual(mr.floors(point(c=476), ref), [])
        self.assertTrue(mr.floors(point(c=475), ref))
        self.assertTrue(mr.floors(point(s=287), ref))
        self.assertEqual(mr.floors(point(s=288), ref), [])
        self.assertTrue(mr.floors(point(fam=59), ref))
        self.assertTrue(mr.gate(point(), ref, {"delta": -0.02})["reasons"])
        g = mr.gate(point(), ref, {"delta": 0.025})
        self.assertEqual((g["eligible"], g["htdev2"]), (True, "GAIN"))
        self.assertEqual(mr.gate(point(), ref, {"delta": -0.0199})["htdev2"], "TIE")

    def test_pick_prefers_gain_then_larger_alpha(self):
        rows = [
            {
                "step": "1",
                "point": "p1",
                "eligible": True,
                "htdev2": "TIE",
                "proxy": 47,
                "reasons": [],
            },
            {
                "step": "1/2",
                "point": "p2",
                "eligible": True,
                "htdev2": "GAIN",
                "proxy": 47,
                "reasons": [],
            },
            {
                "step": "1/3",
                "point": "p3",
                "eligible": True,
                "htdev2": "GAIN",
                "proxy": 47,
                "reasons": [],
            },
        ]
        self.assertEqual(mr.pick(rows)["point"], "p2")
        rows[1]["eligible"] = rows[2]["eligible"] = False
        self.assertEqual(mr.pick(rows)["point"], "p1")
        rows[0]["eligible"] = False
        self.assertIsNone(mr.pick(rows))

    def test_select_slots_and_proxy_drop(self):
        def r(p, proxy, ok=True):
            return [
                {
                    "step": "1",
                    "point": p,
                    "eligible": ok,
                    "htdev2": "TIE",
                    "proxy": proxy,
                    "reasons": [] if ok else ["x"],
                }
            ]

        out = mr.select(
            {"D1": r("a", 50), "D2": r("b", 41.9), "C": r("c", 49, ok=False)}, {}
        )
        self.assertEqual([f["line"] for f in out["finalists"]], ["D1"])
        self.assertEqual(out["proxy_dropped"], {"D2": "b"})
        out = mr.select(
            {"D1": r("a", 50), "C": r("c", 49)}, {"D2": "stopped by the early rule"}
        )
        self.assertEqual(
            [(f["slot"], f["line"]) for f in out["finalists"]], [(1, "D1"), (2, "C")]
        )

    def test_early_select_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "checkpoint-0000010").mkdir()
            (run / "BEST.json").write_text(
                json.dumps({"checkpoint": "checkpoint-0000010"})
            )
            (run / "checkpoint-0000010" / "checkpoint.json").write_text(
                json.dumps({"dev_metrics": {"family_macro_accuracy": 0.61}})
            )
            self.assertAlmostEqual(mr.final_select(run), 0.61)


if __name__ == "__main__":
    unittest.main()
