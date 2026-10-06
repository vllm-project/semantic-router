"""Tests for the LFM2.5 pooling, wrapper and prediction loop, using a fake encoder."""

import types

import pytest
import torch
import train_lfm25_encoder as trainer
from lfm25_classifier import (
    Lfm2ForModalityClassification,
    hidden_size_of,
    mean_pool,
    predict_labels,
)
from torch import nn


class FakeBody(nn.Module):
    """Returns the one-hot of each token id as its hidden state."""

    def __init__(self, hidden_size=3):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden_size)
        self.hidden_size = hidden_size

    def forward(self, input_ids=None, attention_mask=None):
        hidden = torch.nn.functional.one_hot(input_ids, self.hidden_size).float()
        return types.SimpleNamespace(last_hidden_state=hidden)


def test_mean_pool_ignores_padding():
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    pooled = mean_pool(hidden, torch.tensor([[1, 1, 0]]))
    assert pooled.tolist() == [[2.0, 2.0]]


def test_mean_pool_of_an_empty_sequence_is_finite():
    pooled = mean_pool(torch.ones(1, 2, 2), torch.zeros(1, 2, dtype=torch.long))
    assert torch.isfinite(pooled).all()


def test_hidden_size_is_found_on_a_plain_or_wrapped_body():
    assert hidden_size_of(FakeBody(5)) == 5
    wrapped = types.SimpleNamespace(
        base_model=types.SimpleNamespace(config=types.SimpleNamespace(hidden_size=7))
    )
    assert hidden_size_of(wrapped) == 7


def test_wrapper_returns_logits_and_a_loss_only_when_given_labels():
    model = Lfm2ForModalityClassification(FakeBody(3), num_labels=3, dropout=0.0)
    ids, mask = torch.tensor([[0, 1], [2, 2]]), torch.ones(2, 2, dtype=torch.long)
    without = model(input_ids=ids, attention_mask=mask)
    with_labels = model(
        input_ids=ids, attention_mask=mask, labels=torch.tensor([0, 2]), extra_column=1
    )
    assert without.logits.shape == (2, 3) and without.loss is None
    assert with_labels.loss.item() > 0


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        ids = torch.tensor([[len(t) % 3, len(t) % 3] for t in texts])

        class Batch(dict):
            def to(self, device):
                return self

        return Batch(input_ids=ids, attention_mask=torch.ones_like(ids))


def test_predictions_follow_the_head_and_the_text_order():
    head = nn.Linear(3, 3, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.eye(3))  # class = index of the one-hot the body returns
    texts = ["a", "bb", "ccc", "dddd"]  # lengths 1, 2, 3, 4 -> classes 1, 2, 0, 1
    preds = predict_labels(
        FakeBody(3),
        head,
        FakeTokenizer(),
        texts,
        batch_size=3,
        max_length=8,
        device="cpu",
    )
    assert preds == ["DIFFUSION", "BOTH", "AR", "DIFFUSION"]


def test_class_weights_use_the_shared_rule():
    rows = [{"label": 0}] * 40 + [{"label": 1}] * 40 + [{"label": 2}] * 10
    weights = trainer.class_weights_tensor(rows)
    assert weights.dtype == torch.float32
    assert weights[2] > weights[0]
    assert weights[2].item() == pytest.approx(3**0.5)  # sqrt of 90 / (3 * 10)


def test_tokenized_rows_carry_ids_mask_and_labels():
    class Tok:
        def __call__(self, texts, **kwargs):
            return {
                "input_ids": torch.zeros(len(texts), 4, dtype=torch.long),
                "attention_mask": torch.ones(len(texts), 4, dtype=torch.long),
            }

    dataset = trainer.tokenize_rows(
        [{"text": "a", "label": 2}, {"text": "b", "label": 0}], Tok()
    )
    assert dataset.column_names == ["input_ids", "attention_mask", "labels"]
    assert dataset["labels"] == [2, 0]
