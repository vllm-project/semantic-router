"""Task head readouts and decoders, against the legacy binding's rules."""

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from vllm_srun.heads.grounded import GroundingPolicy, PairEnvelope, answer_spans
from vllm_srun.heads.scores import OperatingPoint
from vllm_srun.heads.task import ClassifierHead, Rows, cache_key
from vllm_srun.heads.token import bio_spans, trim
from vllm_srun.testing.fixtures import save
from vllm_srun.testing.task_heads import (
    BOS,
    EOS,
    encoder_tokenizer,
    operating_point,
)

LABELS = ["O", "B-PERSON", "I-PERSON", "B-EMAIL", "I-EMAIL"]


def one_hot(choices, probability=0.9):
    rows = np.full((len(choices), len(LABELS)), (1 - probability) / (len(LABELS) - 1))
    for row, choice in enumerate(choices):
        rows[row, LABELS.index(choice)] = probability
    return rows.astype(np.float32)


def test_bio_decoding_follows_the_legacy_merge():
    text = "Hi Tom Baker tom@x.io ok"
    offsets = [(0, 2), (2, 6), (6, 12), (12, 16), (16, 21), (21, 24)]
    tags = ["O", "B-PERSON", "I-PERSON", "I-EMAIL", "I-EMAIL", "O"]
    probabilities = one_hot(tags)
    probabilities[2, 2] = 0.7
    spans = bio_spans(text, offsets, probabilities, LABELS)
    assert [(s["label"], s["text"], s["start"], s["end"]) for s in spans] == [
        ("PERSON", "Tom Baker", 3, 12),
        ("EMAIL", "tom@x.io", 13, 21),
    ]
    assert spans[0]["probability"] == pytest.approx((0.9 + 0.7) / 2)


def test_an_inside_tag_of_another_type_opens_a_new_entity():
    text = "Tom tom@x.io"
    spans = bio_spans(text, [(0, 3), (3, 12)], one_hot(["B-PERSON", "I-EMAIL"]), LABELS)
    assert [s["label"] for s in spans] == ["PERSON", "EMAIL"]


def test_spans_are_trimmed_with_rusts_whitespace_set():
    assert trim(" \tTom\n", 0, 6) == (2, 5)
    assert trim("\x1cTom", 0, 4) == (0, 4)
    assert trim("   ", 0, 3) == (0, 3)


def test_answer_spans_merge_passing_runs_and_skip_empty_tokens():
    offsets = [(0, 3), (3, 3), (3, 7), (7, 9), (9, 12)]
    probabilities = [0.9, 0.1, 0.6, 0.4, 0.8]
    spans = answer_spans(offsets, probabilities, lambda p: p > 0.5)
    assert spans == [
        {"start": 0, "end": 7, "probability": 0.9},
        {"start": 9, "end": 12, "probability": 0.8},
    ]
    assert answer_spans([(0, 1)], [0.5], lambda p: p > 0.5) == []


def test_operating_points_select_by_their_comparison():
    labels = ["a", "b", "c"]
    point = OperatingPoint.parse(
        {
            "score_type": "independent_sigmoid",
            "comparison": "score >= threshold",
            "labels": labels,
            "thresholds": [0.5, 0.2, 0.9],
            "input_policy": {
                "strategy": "overlapping_content_windows",
                "window_tokens_including_special_tokens": 2048,
                "overlap_content_tokens": 1023,
                "max_document_tokens_including_special_tokens": 32768,
            },
        },
        labels,
    )
    assert point.window == (2048, 1023) and point.max_tokens == 32768
    assert point.select([0.5, 0.1, 0.95]) == ["a", "c"]
    fixture = operating_point("scores", labels, seed=0)
    assert OperatingPoint.parse(fixture, labels).comparison == "score >= threshold"
    with pytest.raises(ValueError, match="labels"):
        OperatingPoint.parse(fixture, ["c", "b", "a"])
    with pytest.raises(ValueError, match="comparison"):
        OperatingPoint.parse({**fixture, "comparison": "score < threshold"}, labels)


def test_grounding_policies_name_the_pair_and_labels():
    labels = ["supported", "hallucinated"]
    policy = GroundingPolicy.parse(operating_point("grounded", labels, 0), labels)
    assert policy.strict and policy.positive == 1 and policy.max_tokens == 512
    assert policy.passes(0.51, 0.5) and not policy.passes(0.5, 0.5)
    document = operating_point("grounded", labels, 0)
    for broken in (
        {"input_pair": ["{question} {other}", "answer"]},
        {"label2id": {"supported": 1, "hallucinated": 0}},
        {"input_pair": ["{question}\n\n{context}", "context"]},
    ):
        with pytest.raises(ValueError):
            GroundingPolicy.parse({**document, **broken}, labels)


def test_pair_framing_is_the_tokenizers(tmp_path):
    encoder_tokenizer(tmp_path)
    tokenizer = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    pair = PairEnvelope.of(tokenizer)
    assert (pair.prefix, pair.middle, pair.suffix) == ((BOS,), (EOS,), (EOS,))


def test_classifier_tensors_must_be_complete(tmp_path):
    config = {"hidden_size": 8, "classifier_activation": "gelu"}
    head = ClassifierHead(config, 3)
    tensors = {
        f"head.{k}" if not k.startswith("classifier") else k: v
        for k, v in head.state_dict().items()
    }
    save(tensors, tmp_path / "model.safetensors")
    loaded = ClassifierHead.load([tmp_path / "model.safetensors"], config, 3)
    x = torch.randn(2, 8)
    assert torch.equal(loaded(x), head.eval()(x))
    save({"classifier.weight": torch.zeros(3, 8)}, tmp_path / "partial.safetensors")
    with pytest.raises(ValueError, match="missing"):
        ClassifierHead.load([tmp_path / "partial.safetensors"], config, 3)


def test_rows_pool_each_packed_sequence():
    hidden = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    rows = Rows({4: hidden}, starts=[0, 2], lengths=[2, 4])
    assert rows.first([1, 0], 4).tolist() == [[4.0, 5.0], [0.0, 1.0]]
    assert rows.mean([1], 4).tolist() == [[7.0, 8.0]]
    assert rows.tokens([1], 4).shape == (4, 2)


def test_cache_keys_cover_identity_head_layer_and_ids():
    base = cache_key("m", "default", 22, [2, 5, 1])
    assert base != cache_key("n", "default", 22, [2, 5, 1])
    assert base != cache_key("m", "other", 22, [2, 5, 1])
    assert base != cache_key("m", "default", 11, [2, 5, 1])
    assert base != cache_key("m", "default", 22, [2, 51])
