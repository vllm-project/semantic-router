import math
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[3]))

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "requires torch")
class OrdinalHeadTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        from clm9b.heads import OrdinalScoreHead

        self.head = OrdinalScoreHead(16, hidden=8)

    def test_probabilities_are_proper_and_ordinal(self):
        from clm9b.heads import ordinal_probabilities

        state = torch.randn(3, 16)
        levels = torch.randn(3, 5, 16)
        mask = torch.tensor(
            [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.bool
        )
        cumulative = self.head(state, levels * mask[..., None], mask)
        self.assertEqual(cumulative.shape, (3, 4))
        self.assertTrue(
            torch.isinf(cumulative[0, 2:]).all()
            and torch.isinf(cumulative[2, 1:]).all()
        )
        counts = mask.sum(-1)
        for temperature in (0.5, 1.0, 3.0):
            probs = ordinal_probabilities(cumulative, counts, temperature)
            self.assertTrue(torch.allclose(probs.sum(-1), torch.ones(3)))
            self.assertTrue((probs >= 0).all())
            self.assertTrue((probs[0, 3:] == 0).all() and (probs[2, 2:] == 0).all())
        finite = cumulative[1]
        self.assertTrue((finite[:-1] > finite[1:]).all(), "thresholds must increase")

    def test_level_count_independence_of_absent_levels(self):
        state = torch.randn(1, 16)
        levels = torch.randn(1, 3, 16)
        mask = torch.ones(1, 3, dtype=torch.bool)
        padded_levels = torch.cat([levels, torch.randn(1, 2, 16)], dim=1)
        padded_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
        short = self.head(state, levels, mask)
        long = self.head(state, padded_levels * padded_mask[..., None], padded_mask)
        self.assertTrue(torch.allclose(short, long[:, :2], atol=1e-6))


@unittest.skipUnless(HAS_TORCH, "requires torch")
class InfoNCETest(unittest.TestCase):
    def test_own_distractors_are_masked_and_hard_negatives_change_loss(self):
        from clm9b.objectives import bidirectional_infonce

        torch.manual_seed(1)
        state = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
        pool = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)
        gold = torch.tensor([0, 1, 0])
        own = torch.tensor([[False, True], [False, False], [False, False]])
        base = bidirectional_infonce(state, pool, torch.tensor(10.0), gold, own)
        changed = pool.clone()
        changed[1] = -changed[1]
        again = bidirectional_infonce(
            state, changed, torch.tensor(10.0), torch.tensor([0, 1, 0]), own
        )
        self.assertTrue(
            torch.isfinite(base["total"]) and torch.isfinite(again["total"])
        )
        hard = torch.nn.functional.normalize(torch.randn(3, 2, 8), dim=-1)
        hard_mask = torch.tensor([[True, False], [True, True], [False, False]])
        with_hard = bidirectional_infonce(
            state, pool, torch.tensor(10.0), gold, own, hard, hard_mask
        )
        self.assertGreater(float(with_hard["forward"]), float(base["forward"]))
        self.assertAlmostEqual(
            float(with_hard["backward"]), float(base["backward"]), places=6
        )

    def test_masked_pool_entry_does_not_affect_forward_loss(self):
        from clm9b.objectives import bidirectional_infonce

        state = torch.nn.functional.normalize(torch.randn(2, 4), dim=-1)
        pool = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1)
        gold = torch.tensor([0, 1])
        own = torch.tensor([[False, False, True], [False, False, True]])
        first = bidirectional_infonce(
            state, pool, torch.tensor(5.0), torch.tensor([0, 1]), own
        )
        moved = pool.clone()
        moved[2] = state[0]
        second = bidirectional_infonce(state, moved, torch.tensor(5.0), gold, own)
        self.assertAlmostEqual(
            float(first["forward"]), float(second["forward"]), places=6
        )


@unittest.skipUnless(HAS_TORCH, "requires torch")
class CalibrationTest(unittest.TestCase):
    def test_golden_temperature_recovers_scale(self):
        from clm9b.train_heads import golden_temperature, softmax

        torch.manual_seed(2)
        records = []
        for _ in range(400):
            logits = (torch.randn(4) * 2).tolist()
            probs = softmax(logits, 2.0)
            label = int(torch.multinomial(torch.tensor(probs), 1))
            records.append((logits, label))

        def objective(t):
            return sum(-math.log(softmax(l, t)[y]) for l, y in records) / len(records)

        fitted = golden_temperature(objective)
        self.assertLess(abs(fitted - 2.0), 0.5)

    def test_absolute_score_option_mapping(self):
        from clm9b.train_heads import option_probabilities, point

        record = {
            "task_type": "score",
            "keys": ["2", "0", "1"],
            "cumulative": [3.0, -3.0],
            "relative": [0.0, 0.0, 0.0],
        }
        temps = {
            "choice": 1.0,
            "noul": 1.0,
            "score_relative": 1.0,
            "score_absolute": 1.0,
        }
        probs = option_probabilities(record, temps, "absolute")
        self.assertAlmostEqual(sum(probs), 1.0, places=9)
        self.assertEqual(record["keys"][point(record, probs)], "1")


if __name__ == "__main__":
    unittest.main()
