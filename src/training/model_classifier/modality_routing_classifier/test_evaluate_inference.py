"""Tests for checkpoint loading and prediction in evaluate_modality_candidate.py."""

import types

import evaluate_modality_candidate as evaluate
import numpy as np
import pytest
import torch


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    """Maps each text to a one-hot row, so the fake model can pick a class from it."""

    def __init__(self):
        self.calls = []

    def __call__(self, batch, **kwargs):
        self.calls.append(list(batch))
        return FakeBatch(
            features=torch.tensor(
                [[float(len(t) % 3 == k) for k in range(3)] for t in batch]
            )
        )


class FakeModel(torch.nn.Module):
    def __init__(self, num_labels=3):
        super().__init__()
        self.config = types.SimpleNamespace(num_labels=num_labels)

    def forward(self, features):
        return types.SimpleNamespace(logits=features)


def test_predictions_are_mapped_to_canonical_ids_and_batched():
    tokenizer = FakeTokenizer()
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]  # lengths 1..5 -> classes 1, 2, 0, 1, 2
    remap = {0: 2, 1: 0, 2: 1}  # checkpoint class 0 is BOTH, 1 is AR, 2 is DIFFUSION
    preds = evaluate.predict_canonical(
        FakeModel(), tokenizer, remap, texts, batch_size=2, max_length=8, device="cpu"
    )
    assert preds.tolist() == [0, 1, 2, 0, 1]
    assert [len(call) for call in tokenizer.calls] == [2, 2, 1]
    assert preds.dtype == np.int64


def test_load_checkpoint_accepts_a_permuted_mapping(monkeypatch):
    monkeypatch.setattr(
        evaluate,
        "load_sequence_classifier_for_inference",
        lambda path, num_labels: (
            FakeModel(),
            FakeTokenizer(),
            {0: "BOTH", 1: "AR", 2: "DIFFUSION"},
        ),
    )
    _, _, remap = evaluate.load_checkpoint("some/checkpoint")
    assert remap == {0: 2, 1: 0, 2: 1}


@pytest.mark.parametrize(
    ("id2label", "output_size"),
    [
        ({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}, 3),
        ({0: "AR", 1: "DIFFUSION", 2: "BOTH"}, 4),
    ],
)
def test_load_checkpoint_fails_closed(monkeypatch, id2label, output_size):
    monkeypatch.setattr(
        evaluate,
        "load_sequence_classifier_for_inference",
        lambda path, num_labels: (FakeModel(output_size), FakeTokenizer(), id2label),
    )
    with pytest.raises(ValueError):
        evaluate.load_checkpoint("some/checkpoint")


def test_config_lists_the_three_models_in_report_order():
    config = evaluate.EvalConfig("t", "tr", "v", "pub", "clean", "cand")
    assert list(config.model_paths) == [
        "published_baseline",
        "clean_baseline",
        "candidate",
    ]
    args = evaluate.build_parser().parse_args(
        [
            "--test-file",
            "t",
            "--train-file",
            "tr",
            "--val-file",
            "v",
            "--published-baseline-model-path",
            "pub",
            "--clean-baseline-model-path",
            "clean",
            "--candidate-model-path",
            "cand",
        ]
    )
    assert evaluate.config_from_args(args) == config
