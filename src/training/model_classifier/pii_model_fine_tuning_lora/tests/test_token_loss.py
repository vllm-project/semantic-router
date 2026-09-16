"""Numerical document weighting and logical-batch accumulation contracts."""

# ruff: noqa: PLC0415

import importlib.util
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from token_loss import document_mean_loss


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional Torch dependency")
class DocumentLossTests(unittest.TestCase):
    def test_unequal_lengths_give_documents_equal_weight(self):
        import torch

        # One difficult token and three easy tokens: documents contribute half
        # each, whereas token mean gives the easy document three quarters.
        logits = torch.tensor([[[0.0, math.log(3.0)]] * 3, [[math.log(3.0), 0.0]] * 3])
        labels = torch.tensor([[0, -100, -100], [0, 0, 0]])
        mask = torch.tensor([[1, 0, 0], [1, 1, 1]])
        actual = document_mean_loss(logits, labels, mask)
        expected = (math.log(4.0) + math.log(4.0 / 3.0)) / 2
        self.assertAlmostEqual(actual.item(), expected, places=6)
        token_mean = (math.log(4.0) + 3 * math.log(4.0 / 3.0)) / 4
        self.assertGreater(abs(actual.item() - token_mean), 0.2)

    def test_ignore_labels_and_padding_have_zero_loss_gradient(self):
        import torch

        logits = torch.tensor(
            [[[0.0, 0.0], [8.0, -8.0], [9.0, -9.0]]], requires_grad=True
        )
        # Special-token ignore and populated padding label are both excluded.
        labels = torch.tensor([[0, -100, 1]])
        mask = torch.tensor([[1, 1, 0]])
        loss = document_mean_loss(logits, labels, mask)
        self.assertAlmostEqual(loss.item(), math.log(2.0), places=6)
        loss.backward()
        torch.testing.assert_close(logits.grad[0, 0], torch.tensor([-0.5, 0.5]))
        self.assertEqual(logits.grad[0, 1:].count_nonzero().item(), 0)

    def test_accumulation_matches_one_logical_batch_with_unequal_lengths(self):
        import torch

        generator = torch.Generator().manual_seed(123)
        original = torch.randn(4, 7, 3, generator=generator)
        lengths = torch.tensor([1, 7, 2, 5])
        mask = (torch.arange(7)[None, :] < lengths[:, None]).long()
        labels = torch.randint(0, 3, (4, 7), generator=generator)
        labels[1, 2] = -100
        whole = original.clone().requires_grad_()
        whole_loss = document_mean_loss(whole, labels, mask)
        whole_loss.backward()
        split = original.clone().requires_grad_()
        total = 0.0
        for start in (0, 2):
            # Actual microbatch padding differs; logical document count does not.
            end = start + 2
            width = int(lengths[start:end].max())
            loss = (
                document_mean_loss(
                    split[start:end, :width],
                    labels[start:end, :width],
                    mask[start:end, :width],
                )
                / 2
            )
            total += loss.item()
            loss.backward()
        self.assertAlmostEqual(total, whole_loss.item(), places=6)
        torch.testing.assert_close(split.grad, whole.grad, atol=1e-7, rtol=1e-6)

    def test_bfloat16_logits_reduce_in_float32(self):
        import torch

        logits = torch.zeros(1, 2, 3, dtype=torch.bfloat16, requires_grad=True)
        loss = document_mean_loss(
            logits, torch.zeros(1, 2, dtype=torch.long), torch.ones(1, 2)
        )
        self.assertEqual(loss.dtype, torch.float32)
        self.assertAlmostEqual(loss.item(), math.log(3.0), places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_empty_supervision_invalid_shape_and_mask_are_rejected(self):
        import torch

        logits = torch.zeros(2, 3, 2)
        labels = torch.zeros(2, 3, dtype=torch.long)
        ignored = labels.clone()
        ignored[1] = -100
        for targets, mask, error in (
            (ignored, torch.ones(2, 3), "Every document"),
            (labels, torch.tensor([[1, 1, 1], [0, 0, 0]]), "Every document"),
            (labels[:, :1], torch.ones(2, 3), "batch/token"),
            (labels, torch.full((2, 3), 0.5), "zero or one"),
        ):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                document_mean_loss(logits, targets, mask)


if __name__ == "__main__":
    unittest.main()
