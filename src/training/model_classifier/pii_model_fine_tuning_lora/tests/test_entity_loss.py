"""Entity/document weighting, explicit span IDs and logical batch gradients."""

# ruff: noqa: PLC0415

import importlib.util
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from token_loss import IGNORE_INDEX, entity_document_mean_loss, pad_entity_ids


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional Torch dependency")
class EntityDocumentLossTests(unittest.TestCase):
    def test_entities_have_equal_mass_despite_unequal_subword_lengths(self):
        import torch

        # The two same-type entities have one hard token versus three easy
        # tokens. Their span IDs keep them distinct; one O token gets half.
        logits = torch.tensor(
            [[[math.log(3), 0.0]] + [[0.0, math.log(3)]] * 3 + [[0.0, 0.0]]],
            requires_grad=True,
        )
        labels = torch.tensor([[1, 1, 1, 1, 0]])
        ids = torch.tensor([[0, 1, 1, 1, -1]])
        loss = entity_document_mean_loss(
            logits, labels, torch.ones_like(labels), ids, o_label_id=0
        )
        expected = (math.log(4) + math.log(4 / 3)) / 4 + math.log(2) / 2
        self.assertAlmostEqual(loss.item(), expected, places=6)
        loss.backward()
        torch.testing.assert_close(logits.grad[0, 0], torch.tensor([0.1875, -0.1875]))
        torch.testing.assert_close(logits.grad[0, 1], torch.tensor([1 / 48, -1 / 48]))
        torch.testing.assert_close(logits.grad[0, 4], torch.tensor([-0.25, 0.25]))

    def test_negative_and_no_o_documents_keep_unit_weight(self):
        import torch

        logits = torch.tensor([[[0.0, math.log(3)]] * 3, [[0.0, math.log(3)]] * 3])
        labels = torch.tensor([[0, 0, 0], [1, 1, 1]])
        ids = torch.tensor([[-1, -1, -1], [0, 0, 1]])
        loss = entity_document_mean_loss(
            logits, labels, torch.ones_like(labels), ids, o_label_id=0
        )
        self.assertAlmostEqual(
            loss.item(), (math.log(4) + math.log(4 / 3)) / 2, places=6
        )

    def test_masks_and_unequal_microbatch_partitions_preserve_gradients(self):
        import torch

        generator = torch.Generator().manual_seed(17)
        values = torch.randn(5, 7, 3, generator=generator)
        labels = torch.tensor(
            [
                [1, 2, 0, -100, 2, 2, 2],
                [0] * 7,
                [1, 2, 1, 0, 0, 0, 0],
                [1, 2, 2, -100, -100, -100, -100],
                [0, 1, 2, 0, 0, 0, 0],
            ]
        )
        ids = torch.tensor(
            [
                [0, 0, -1, 0, 99, 99, 99],
                [-1] * 7,
                [0, 0, 1, -1, -1, -1, -1],
                [0, 0, 0, -1, -1, -1, -1],
                [-1, 0, 0, -1, -1, -1, -1],
            ]
        )
        lengths = [4, 7, 3, 3, 5]
        mask = (torch.arange(7)[None] < torch.tensor(lengths)[:, None]).long()
        whole = values.clone().requires_grad_()
        loss = entity_document_mean_loss(whole, labels, mask, ids, o_label_id=0)
        loss.backward()
        partitioned = values.clone().requires_grad_()
        total = 0.0
        for start, end in [(0, 1), (1, 4), (4, 5)]:
            width = max(lengths[start:end])
            part = entity_document_mean_loss(
                partitioned[start:end, :width],
                labels[start:end, :width],
                mask[start:end, :width],
                ids[start:end, :width],
                o_label_id=0,
            ) * ((end - start) / 5)
            part.backward()
            total += part.item()
        self.assertAlmostEqual(total, loss.item(), places=6)
        torch.testing.assert_close(partitioned.grad, whole.grad, atol=1e-7, rtol=1e-6)
        ignored = (labels == IGNORE_INDEX) | (mask == 0)
        self.assertEqual(whole.grad[ignored].count_nonzero().item(), 0)

    def test_bfloat16_logits_reduce_in_float32(self):
        import torch

        logits = torch.zeros(1, 2, 3, dtype=torch.bfloat16, requires_grad=True)
        loss = entity_document_mean_loss(
            logits,
            torch.tensor([[1, 0]]),
            torch.ones(1, 2),
            torch.tensor([[0, -1]]),
            o_label_id=0,
        )
        self.assertEqual(loss.dtype, torch.float32)
        self.assertAlmostEqual(loss.item(), math.log(3), places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_invalid_entity_metadata_and_empty_supervision_are_rejected(self):
        import torch

        logits = torch.zeros(1, 2, 2)
        labels = torch.tensor([[1, 0]])
        mask = torch.ones(1, 2)
        for ids, error in [
            (torch.tensor([[-1, -1]]), "span IDs"),
            (torch.tensor([[0, 1]]), "observed O"),
            (torch.tensor([[0.0, -1.0]]), "int64"),
            (torch.tensor([[0]]), "label shape"),
        ]:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                entity_document_mean_loss(logits, labels, mask, ids, o_label_id=0)
        with self.assertRaisesRegex(ValueError, "Every document"):
            entity_document_mean_loss(
                logits,
                torch.full_like(labels, -100),
                mask,
                torch.tensor([[-1, -1]]),
                o_label_id=0,
            )

    def test_entity_padding_matches_left_and_right_token_padding(self):
        import torch

        rows = [
            {"input_ids": [2, 4, 1], "entity_ids": [-1, 0, -1]},
            {"input_ids": [2, 4, 5, 1], "entity_ids": [-1, 0, 0, -1]},
        ]
        torch.testing.assert_close(
            pad_entity_ids(rows, 5, "left"),
            torch.tensor([[-1, -1, -1, 0, -1], [-1, -1, 0, 0, -1]]),
        )
        torch.testing.assert_close(
            pad_entity_ids(rows, 5, "right"),
            torch.tensor([[-1, 0, -1, -1, -1], [-1, 0, 0, -1, -1]]),
        )
        with self.assertRaisesRegex(ValueError, "before padding"):
            pad_entity_ids(rows, 2, "right")


if __name__ == "__main__":
    unittest.main()
