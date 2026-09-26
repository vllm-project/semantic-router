import importlib.util
import math
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "CPU torch is unavailable in this workspace"
)
class LossTest(unittest.TestCase):
    def test_ce_brier_replay_and_padding(self):
        import torch

        from training.model.loss import per_example_loss

        logits = torch.tensor(
            [[0.0, 0.0, float("nan")], [2.0, 0.0, -1.0]], requires_grad=True
        )
        labels = torch.tensor([0, 1])
        mask = torch.tensor([[True, True, False], [True, True, True]])
        teacher = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.8, 0.1]])
        replay = torch.tensor([False, True])
        ce = per_example_loss(logits, labels, mask, objective="ce")
        mix = per_example_loss(
            logits,
            labels,
            mask,
            objective="ce_brier",
            brier_weight=0.5,
            teacher_probs=teacher,
            replay_mask=replay,
            replay_kl_weight=0.3,
        )
        torch.testing.assert_close(
            ce["ce"][0], torch.tensor(math.log(2)), atol=1e-6, rtol=0
        )
        torch.testing.assert_close(
            mix["total"], mix["ce"] + 0.5 * mix["brier"] + 0.3 * mix["replay_kl"]
        )
        self.assertEqual(mix["replay_kl"][0].item(), 0.0)
        mix["total"].mean().backward()
        self.assertTrue(torch.isfinite(logits.grad[mask]).all())
        self.assertEqual(logits.grad[0, 2].item(), 0.0)

    def test_invalid_gold_or_replay_distribution(self):
        import torch

        from training.model.loss import per_example_loss

        logits = torch.zeros(1, 2)
        mask = torch.tensor([[True, False]])
        with self.assertRaisesRegex(ValueError, "2..255"):
            per_example_loss(logits, torch.tensor([0]), mask)
        mask[:] = True
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            per_example_loss(logits, torch.tensor([2]), mask)
        with self.assertRaisesRegex(ValueError, "sum to one"):
            per_example_loss(
                logits,
                torch.tensor([1]),
                mask,
                teacher_probs=torch.tensor([[0.3, 0.3]]),
                replay_mask=torch.tensor([True]),
                replay_kl_weight=0.2,
            )


if __name__ == "__main__":
    unittest.main()
