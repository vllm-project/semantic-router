"""CPU behavior tests for trained heads and partial accumulation windows."""

from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    import torch

    from src.training.model_embeddings.mmbert_32k import reranker_training
    from src.training.model_embeddings.mmbert_32k.foundation_collators import (
        _special_token_ids,
        _standard_mask,
    )
    from src.training.model_embeddings.mmbert_32k.reranker_model import (
        Matryoshka2DReranker,
    )
except ImportError:
    torch = None


@unittest.skipIf(
    torch is None, "requires the training torch and transformers dependencies"
)
class RerankerBoundaryTest(unittest.TestCase):
    def test_tail_window_steps_and_scales_without_crossing_epochs(self) -> None:
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.zero_()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        scheduler = mock.Mock()
        args = SimpleNamespace(
            epochs=2,
            gradient_accumulation_steps=2,
            max_grad_norm=1e6,
            logging_steps=1000,
            save_steps=0,
        )
        loader = [{"value": torch.tensor(value)} for value in (1.0, 3.0, 5.0)]
        with mock.patch.object(
            reranker_training,
            "_forward_loss",
            side_effect=lambda _args, net, batch, _loss: net.weight.sum()
            * batch["value"],
        ):
            reranker_training._run_training(
                args, model, None, loader, optimizer, scheduler, None
            )
        # Each epoch applies mean(1,3) then mean(5), with no stale tail gradients.
        self.assertAlmostEqual(model.weight.item(), -14.0)
        self.assertEqual(scheduler.step.call_count, 4)

    def test_missing_trained_heads_fail_before_loading_backbone(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(FileNotFoundError, "trained reranker heads"),
        ):
            Matryoshka2DReranker.from_pretrained(directory)

    def test_distinct_bos_and_eos_are_not_mlm_targets(self) -> None:
        tokenizer = SimpleNamespace(
            pad_token_id=0,
            cls_token_id=1,
            sep_token_id=1,
            bos_token_id=2,
            eos_token_id=3,
            unk_token_id=4,
            mask_token_id=5,
        )
        ids = torch.tensor([[2, 10, 3]])
        selected = _standard_mask(
            ids,
            torch.ones_like(ids),
            1.0,
            _special_token_ids(tokenizer, include_mask=True),
        )
        self.assertEqual(selected.tolist(), [[False, True, False]])


if __name__ == "__main__":
    unittest.main()
