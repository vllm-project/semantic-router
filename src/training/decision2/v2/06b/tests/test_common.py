import importlib
import math
import unittest

common = importlib.import_module("v2.06b.common")


def row(task_type, keys, label, family="fam", language="en"):
    return {
        "id": f"{family}-{task_type}-{label}",
        "task_type": task_type,
        "options": [{"key": k, "description": k} for k in keys],
        "label": label,
        "family": family,
        "language": language,
    }


class SelectRecordTest(unittest.TestCase):
    def test_choice_tie_is_incorrect(self):
        r = row("choice", ["a", "b", "c"], 0)
        self.assertFalse(common.select_record(r, [0.4, 0.4, 0.2])["correct"])
        self.assertTrue(common.select_record(r, [0.5, 0.3, 0.2])["correct"])

    def test_noul_uses_true_probability_and_exact_half_abstains(self):
        r = row("noul", ["true", "false"], 1)
        self.assertTrue(common.select_record(r, [0.2, 0.8])["correct"])
        self.assertEqual(common.select_record(r, [0.7, 0.3])["chosen"], 0)
        self.assertFalse(common.select_record(r, [0.7, 0.3])["correct"])
        self.assertIsNone(common.select_record(r, [0.5, 0.5])["chosen"])

    def test_half_brier_and_nll(self):
        r = row("score", ["0", "1", "2"], 2)
        out = common.select_record(r, [0.2, 0.3, 0.5])
        self.assertAlmostEqual(out["brier"], (0.04 + 0.09 + 0.25) / 2)
        self.assertAlmostEqual(out["nll"], -math.log(0.5))

    def test_invalid_vector_rejected(self):
        with self.assertRaises(ValueError):
            common.select_record(row("choice", ["a", "b"], 0), [0.9, 0.3])


class MappingTest(unittest.TestCase):
    def test_noul_maps_native_no_yes_to_original_keys(self):
        r = row("noul", ["true", "false"], 0)
        self.assertEqual(
            common.original_probabilities(r, ["no", "yes"], [0.25, 0.75]), [0.75, 0.25]
        )

    def test_choice_order_must_be_preserved(self):
        r = row("choice", ["x", "y"], 0)
        with self.assertRaises(ValueError):
            common.original_probabilities(r, ["y", "x"], [0.5, 0.5])

    def test_score_is_positional(self):
        r = row("score", ["0", "1", "2"], 1)
        self.assertEqual(
            common.original_probabilities(r, ["0", "1", "2"], [0.1, 0.2, 0.7]),
            [0.1, 0.2, 0.7],
        )


class SummaryTest(unittest.TestCase):
    def test_family_macro_and_best_rule(self):
        records = [
            common.select_record(row("choice", ["a", "b"], 0, family="f1"), [0.9, 0.1]),
            common.select_record(row("choice", ["a", "b"], 1, family="f1"), [0.9, 0.1]),
            common.select_record(
                row("noul", ["false", "true"], 1, family="f2"), [0.2, 0.8]
            ),
        ]
        summary = common.metric_summary(records)
        self.assertEqual(summary["correct"], 2)
        self.assertAlmostEqual(summary["family_macro_accuracy"], (0.5 + 1.0) / 2)
        a = {
            "step": 64,
            "metrics": {"family_macro_accuracy": 0.7, "family_macro_brier": 0.2},
        }
        b = {
            "step": 128,
            "metrics": {"family_macro_accuracy": 0.7, "family_macro_brier": 0.1},
        }
        c = {
            "step": 32,
            "metrics": {"family_macro_accuracy": 0.7, "family_macro_brier": 0.1},
        }
        self.assertTrue(common.better(b, a))
        self.assertTrue(common.better(c, b))
        self.assertFalse(common.better(a, b))


class ScheduleTest(unittest.TestCase):
    def test_one_epoch_deterministic_partition(self):
        ids = [f"r{i}" for i in range(10)]
        plan = common.schedule(ids, 4, "seed")
        self.assertEqual(plan, common.schedule(ids, 4, "seed"))
        self.assertEqual(sorted(i for batch in plan for i in batch), list(range(10)))
        self.assertEqual([len(b) for b in plan], [4, 4, 2])
        self.assertNotEqual(plan, common.schedule(ids, 4, "other"))

    def test_learning_rate_matches_kai_decision_finetune(self):
        total, warmup, base, low = 117, 5, 1e-5, 1e-6
        self.assertAlmostEqual(
            common.learning_rate(1, total, warmup, base, low), base / 5
        )
        self.assertAlmostEqual(common.learning_rate(5, total, warmup, base, low), base)
        progress = (6 - warmup) / (total - warmup)
        self.assertAlmostEqual(
            common.learning_rate(6, total, warmup, base, low),
            low + (base - low) * 0.5 * (1 + math.cos(math.pi * progress)),
        )
        self.assertAlmostEqual(
            common.learning_rate(total, total, warmup, base, low), low
        )


if __name__ == "__main__":
    unittest.main()
