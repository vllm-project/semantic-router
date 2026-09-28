from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from benchmark.generate import FINAL_FAMILIES
from jev_arena import compare_v3
from transfer.build import EVALUATION_TASKS
from v2.eval import overlap_effects as oe


def noul_item(item_id: str, family: str, group_id: str, value: bool) -> dict:
    return {
        "id": item_id,
        "family": family,
        "group_id": group_id,
        "questions": {
            "decision": {
                "type": "noul",
                "instructions": "?",
                "criteria": {"true": "t", "false": "f"},
            }
        },
        "gold": {"decision": {"type": "noul", "value": value, "semantic_value": value}},
    }


def typed_panel(rng: random.Random) -> tuple[dict, dict, dict]:
    gold, left, right = {}, {}, {}
    for family in FINAL_FAMILIES:
        for group in range(100):
            for variant in range(4):
                item_id = f"td_{family}_{group:03d}_{variant}"
                value = rng.random() < 0.5
                gold[item_id] = noul_item(item_id, family, f"{family}-{group}", value)
                for predictions, skill in ((left, 0.7), (right, 0.5)):
                    if rng.random() < 0.02:
                        continue
                    right_side = value if rng.random() < skill else not value
                    predictions[item_id] = {
                        "id": item_id,
                        "answers": {
                            "decision": {
                                "type": "noul",
                                "noul": 0.9 if right_side else 0.1,
                            }
                        },
                    }
    return gold, left, right


def css_panel(rng: random.Random) -> tuple[dict, dict, dict]:
    gold, left, right = {}, {}, {}
    tasks = sorted(EVALUATION_TASKS)
    for n in range(6547):
        task = tasks[n % len(tasks)]
        labels = ["a", "b", "c"]
        item_id = f"css/{task}/{n:05d}"
        gold[item_id] = {
            "id": item_id,
            "task": task,
            "role": "evaluation",
            "labels": labels,
            "gold": labels[n // len(tasks) % 3],
        }
        for predictions in (left, right):
            if rng.random() < 0.03:
                continue
            choice = rng.choice(labels)
            probs = {label: (0.8 if label == choice else 0.1) for label in labels}
            predictions[item_id] = {
                "id": item_id,
                "answers": {
                    "label": {
                        "type": "choice",
                        "choice": choice,
                        "probabilities": probs,
                    }
                },
            }
    return gold, left, right


class PairStructuresTest(unittest.TestCase):
    def test_full_panel_structures_equal_compare_v3(self):
        rng = random.Random(7)
        gold, left, right = typed_panel(rng)
        self.assertEqual(
            oe.pair_families(
                gold, oe.typed_counts(gold, left), oe.typed_counts(gold, right), set()
            ),
            compare_v3._typed_groups(gold, left, right),
        )
        gold, left, right = css_panel(rng)
        self.assertEqual(
            oe.pair_tasks(
                gold, oe.css_outcomes(gold, left), oe.css_outcomes(gold, right), set()
            ),
            compare_v3._css_tasks(gold, left, right),
        )

    def test_exclusion_drops_items_and_keeps_label_universe(self):
        rng = random.Random(8)
        gold, left, right = css_panel(rng)
        task = sorted(EVALUATION_TASKS)[0]
        drop = {
            i for i, row in gold.items() if row["task"] == task and row["gold"] == "c"
        }
        coded = oe.pair_tasks(
            gold, oe.css_outcomes(gold, left), oe.css_outcomes(gold, right), drop
        )
        rows, nlabels = coded[task]
        self.assertEqual(nlabels, 3)
        self.assertEqual(
            len(rows), sum(r["task"] == task for r in gold.values()) - len(drop)
        )
        self.assertNotIn(2, {truth for truth, _l, _r in rows})
        f1 = oe.css_task_f1(gold, oe.css_outcomes(gold, left), drop)[task]
        kept = [r for r in gold.values() if r["task"] == task and r["id"] not in drop]
        outcomes = oe.css_outcomes(gold, left)
        expected = oe.macro_f1(
            [r["gold"] for r in kept],
            [outcomes[r["id"]]["choice"] for r in kept],
            ["a", "b", "c"],
        )
        self.assertEqual(f1, expected)

    def test_bootstrap_is_seeded_and_paired(self):
        rng = random.Random(9)
        gold, left, _right = typed_panel(rng)
        counts = oe.typed_counts(gold, left)
        families = oe.pair_families(gold, counts, counts, set())
        cgold, cleft, _cright = css_panel(rng)
        outcomes = oe.css_outcomes(cgold, cleft)
        tasks = oe.pair_tasks(cgold, outcomes, outcomes, set())
        first = oe.v3_bootstrap(families, tasks, 20, 5)
        self.assertEqual(first, oe.v3_bootstrap(families, tasks, 20, 5))
        self.assertEqual(first["ci95"], {"low": 0.0, "high": 0.0})


class FlaggedTest(unittest.TestCase):
    def test_collect_flagged_by_panel_pool_and_role(self):
        gold = {
            "typed-final": {"td_1": {"family": "evidence_join"}},
            "css15": {
                "css/media_ideology/1": {"task": "media_ideology"},
                "css/media_ideology/2": {"task": "media_ideology"},
                "css/ibc/1": {"task": "ibc"},
            },
            "public231": {"hard-x-1": {"tier": "hard"}},
            "mlx-diag": {"mlx-pawsx-en-1": {"type": "noul", "language": "en"}},
            "sha256": {"css15": "x"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scan").mkdir()
            (root / "scan-union").mkdir()

            def group(ids_by_role, methods):
                return {
                    "protected_ids": ids_by_role,
                    "by_method": {m: ids_by_role for m in methods},
                }

            (root / "scan" / "H3.private.json").write_text(
                json.dumps(
                    {
                        "groups": {
                            "g1": group(
                                {"css15_native": ["css/media_ideology/1"]}, ["N"]
                            ),
                            "g2": group({"held_out": ["aho-1"]}, ["N"]),
                            "g3": group(
                                {"decision_bench_v4": ["fin-1", "fin-2"]}, ["S"]
                            ),
                        }
                    }
                )
            )
            (root / "scan" / "V1-A3.private.json").write_text(
                json.dumps(
                    {
                        "groups": {
                            "g4": group(
                                {
                                    "css15_native": [
                                        "css/media_ideology/1",
                                        "css/ibc/1",
                                    ],
                                    "mlx_diag_v1": ["mlx-pawsx-en-1"],
                                },
                                ["L"],
                            )
                        }
                    }
                )
            )
            (root / "scan-union" / "union.private.json").write_text(
                json.dumps(
                    {
                        "groups": {
                            "g4": group({"jevbench_public231": ["hard-x-1"]}, ["N"])
                        }
                    }
                )
            )
            hits = {
                "H3": {
                    "g1": {"roles": ["css15_native"]},
                    "g2": {"roles": ["held_out"]},
                    "g3": {"roles": ["decision_bench_v4"]},
                },
                "V1:A3": {"g4": {"roles": ["css15_native", "mlx_diag_v1"]}},
            }
            (root / "rescreen.private.json").write_text(json.dumps({"hits": hits}))
            wanted = {
                "css15_native",
                "decision_bench_v4",
                "mlx_diag_v1",
                "jevbench_public231",
            }
            result = oe.collect_flagged(root, wanted, gold)
        self.assertEqual(result["groups"], 3)
        self.assertEqual(result["groups_by_pool"], {"H3": 2, "V1-A3": 1})
        self.assertEqual(
            result["panels"]["css15"], ["css/ibc/1", "css/media_ideology/1"]
        )
        self.assertEqual(result["panels"]["public231"], ["hard-x-1"])
        self.assertEqual(result["panels"]["mlx-diag"], ["mlx-pawsx-en-1"])
        self.assertEqual(result["panels"]["typed-final"], [])
        self.assertEqual(result["unscored_by_role"], {"decision_bench_v4": 2})
        self.assertEqual(
            result["strata"]["css15"]["media_ideology"],
            {"items": 1, "pool:H3": 1, "pool:V1-A3": 1},
        )
        self.assertEqual(result["methods"]["css15"], {"L": 1, "L+N": 1})
        self.assertEqual(result["rescreen_private_cross_check"]["groups"], 3)
        self.assertNotIn("aho-1", json.dumps(result))


class AggregatesTest(unittest.TestCase):
    def test_contamination_difference_in_differences(self):
        # task 1: flagged left 2/2, right 1/2; unflagged left 1/2, right 1/2
        # task 2: flagged left 0/1, right 0/1; unflagged left 1/1, right 0/1
        units = [
            ([(1, 1), (1, 0)], [(1, 1), (0, 0)]),
            ([(0, 0)], [(1, 0)]),
        ]
        point = oe.contamination_point(units)
        self.assertAlmostEqual(point["left_flagged_accuracy"], 2 / 3)
        self.assertAlmostEqual(
            point["left_unflagged_accuracy"], 2 / 3 * 0.5 + 1 / 3 * 1.0
        )
        self.assertAlmostEqual(point["right_flagged_accuracy"], 1 / 3)
        self.assertAlmostEqual(point["right_unflagged_accuracy"], 2 / 3 * 0.5)
        self.assertAlmostEqual(
            point["difference_in_differences"],
            (2 / 3 - 2 / 3) - (1 / 3 - 1 / 3),
        )
        boot = oe.contamination_bootstrap(units, 50, 3)
        self.assertEqual(
            set(boot),
            {"left_gap", "right_gap", "delta_flagged", "difference_in_differences"},
        )

    def test_strata_bootstrap_counts_and_type_macro(self):
        strata = [[(1, 0)] * 5, [(1, 1)] * 3]
        self.assertEqual(oe.strata_bootstrap(strata, 10, 1), {"low": 5.0, "high": 5.0})
        macro = oe.strata_bootstrap(strata, 10, 1, kinds=["choice", "noul"])
        self.assertEqual(macro, {"low": 0.5, "high": 0.5})

    def test_mlx_and_public_values_drop_excluded_items(self):
        mlx = {
            "a": {"type": "noul", "language": "en", "correct": True},
            "b": {"type": "noul", "language": "en", "correct": False},
            "c": {"type": "noul", "language": "ko", "correct": True},
            "d": {"type": "choice", "language": "en", "correct": True},
            "e": {"type": "score", "language": "en", "correct": False},
        }
        full = oe.mlx_value(mlx, set())
        reduced = oe.mlx_value(mlx, {"b"})
        self.assertAlmostEqual(
            full["type_macro_accuracy"], ((0.5 + 1.0) / 2 + 1 + 0) / 3
        )
        self.assertAlmostEqual(reduced["type_macro_accuracy"], (1.0 + 1 + 0) / 3)
        self.assertEqual(full["per_language"]["ko"], reduced["per_language"]["ko"])
        public = {
            "x": {"tier": "hard", "correct": True},
            "y": {"tier": "easy", "correct": False},
        }
        self.assertEqual(oe.public_value(public, {"x"})["correct"], 0)
        self.assertEqual(oe.public_value(public, {"x"})["items"], 1)

    def test_ranks_share_places_on_ties(self):
        self.assertEqual(oe.ranks({"a": 2, "b": 3, "c": 2}), [["b"], ["a", "c"]])
        self.assertEqual(oe.ci_status({"low": -0.1, "high": 0.2}), "includes 0")
        self.assertEqual(oe.ci_status({"low": 0.1, "high": 0.2}), "above 0")


if __name__ == "__main__":
    unittest.main()
