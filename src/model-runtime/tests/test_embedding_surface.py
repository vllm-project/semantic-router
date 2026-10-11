"""The shared /v1/embeddings building blocks: parsing, budgets, pooling, views, keys and responses."""

from __future__ import annotations

import base64
import time

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from vllm_srun.heads import embedding
from vllm_srun.plugins.base import DEADLINE, EmbeddingInfo, SurfaceRequest

INFO = EmbeddingInfo(
    dimensions=(8, 4, 2), layers=(1, 3), input_types=("query", "document")
)
MEDIA = EmbeddingInfo(
    dimensions=(6,), layers=(2,), modalities=("text", "image", "audio")
)


def request(body, **options):
    if options:
        body = {**body, "options": options}
    return SurfaceRequest("embeddings", body, None, "exact", True, time.monotonic())


def templated_tokenizer() -> Tokenizer:
    vocab = {"<bos>": 0, "<eos>": 1, "[UNK]": 2, "a": 3, "b": 4, "c": 5, "d": 6}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> $B <eos>",
        special_tokens=[("<bos>", 0), ("<eos>", 1)],
    )
    return tokenizer


def test_defaults_are_the_full_dimension_and_last_layer():
    parsed = embedding.parse_request(request({"input": "a b"}), INFO, 32)
    assert (parsed.dimension, parsed.layer, parsed.input_type) == (8, 3, None)
    assert (parsed.encoding, parsed.overflow, parsed.max_tokens) == (
        "float",
        "reject",
        32,
    )
    assert parsed.inputs == (embedding.EmbeddingInput(0, "text", text="a b"),)


@pytest.mark.parametrize(
    ("body", "options", "message"),
    [
        ({"input": "a", "dimensions": 5}, {}, "dimensions must be one of"),
        ({"input": "a", "dimensions": True}, {}, "positive integer"),
        ({"input": "a", "layer": 2}, {}, "layer must be one of"),
        ({"input": "a", "input_type": "passage"}, {}, "input_type must be one of"),
        ({"input": "a", "encoding_format": "hex"}, {}, "encoding_format"),
        ({"input": "a"}, {"overflow": "window"}, "options.overflow"),
        ({"input": "a"}, {"max_tokens": 64}, "at most 32"),
        ({"input": []}, {}, "nonempty list"),
        ({"input": [3]}, {}, "must be a string or"),
        (
            {"input": [{"type": "image_url", "image_url": {"url": "data:,x"}}]},
            {},
            "embed image",
        ),
    ],
)
def test_invalid_requests(body, options, message):
    with pytest.raises(ValueError, match=message):
        embedding.parse_request(request(body, **options), INFO, 32)


def test_input_types_are_refused_by_uninstructed_models():
    with pytest.raises(ValueError, match="takes no input_type"):
        embedding.parse_request(
            request({"input": "a", "input_type": "query"}), MEDIA, 32
        )


def test_content_parts_decode_media_and_mark_bad_payloads_per_item():
    png = base64.b64encode(b"\x89PNG-bytes").decode()
    wav = base64.b64encode(b"RIFF-bytes").decode()
    body = {
        "input": [
            {"type": "text", "text": "a"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}},
            {"type": "input_audio", "input_audio": {"data": wav, "format": "WAV"}},
            {"type": "input_audio", "input_audio": {"data": "!!", "format": "wav"}},
        ]
    }
    parsed = embedding.parse_request(request(body), MEDIA, 32)
    text, image, audio, broken = parsed.inputs
    assert (
        text.text == "a"
        and image.data == b"\x89PNG-bytes"
        and image.media_type == "image/png"
    )
    assert (
        audio.data == b"RIFF-bytes"
        and audio.media_type == "wav"
        and audio.error is None
    )
    assert broken.error == "invalid_input" and broken.data is None
    with pytest.raises(ValueError, match="data URL"):
        embedding.parse_request(
            request(
                {"input": [{"type": "image_url", "image_url": {"url": "https://x"}}]}
            ),
            MEDIA,
            32,
        )


def test_budgets_reject_or_truncate_inside_the_special_tokens():
    tokenizer = templated_tokenizer()
    ids, usage = embedding.encode_text(tokenizer, "a b c d", 8, "reject")
    assert ids == [0, 3, 4, 5, 6, 1]
    assert usage == {"tokens": 6, "processed_tokens": 6, "truncated": False}
    assert (
        embedding.encode_text(tokenizer, "a b c d", 5, "reject")
        == "max_length_exceeded"
    )
    ids, usage = embedding.encode_text(tokenizer, "a b c d", 4, "truncate")
    assert ids == [0, 3, 4, 1]
    assert usage == {"tokens": 6, "processed_tokens": 4, "truncated": True}
    assert embedding.encode_text(tokenizer, "a", 2, "truncate") == "max_length_exceeded"


def test_pooling_reads_only_real_tokens():
    hidden = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    torch.testing.assert_close(
        embedding.pool(hidden, mask, "mean"), torch.tensor([[2.0, 3.0], [7.0, 8.0]])
    )
    torch.testing.assert_close(embedding.pool(hidden, mask, "cls"), hidden[:, 0])
    torch.testing.assert_close(
        embedding.pool(hidden, mask, "last_token"),
        torch.tensor([[4.0, 5.0], [8.0, 9.0]]),
    )
    with pytest.raises(ValueError, match="unknown pooling"):
        embedding.pool(hidden, mask, "max")


def test_matryoshka_truncates_before_normalizing():
    vectors = torch.tensor([[3.0, 4.0, 12.0], [0.0, 0.0, 0.0]])
    view = embedding.matryoshka(vectors, 2)
    torch.testing.assert_close(view[0], torch.tensor([0.6, 0.8]))
    assert torch.equal(view[1], torch.zeros(2))
    assert torch.equal(
        embedding.matryoshka(vectors, 2, normalize=False), vectors[:, :2]
    )


def test_content_keys_separate_part_boundaries():
    assert embedding.content_key("ab", "c") != embedding.content_key("a", "bc")
    assert embedding.content_key("m", 3, [1, 2]) == embedding.content_key(
        "m", 3, [1, 2]
    )
    assert embedding.content_key(b"\x00", [0]) != embedding.content_key([0], b"\x00")


def test_finish_keeps_input_order_errors_and_representation():
    parsed = embedding.parse_request(
        request({"input": ["a", "b", "c"], "encoding_format": "base64"}), INFO, 32
    )
    usage = {"tokens": 3, "processed_tokens": 3, "truncated": False}
    items = [
        (embedding.EmbedItem(0, "text", [0, 3, 1], "k0"), usage),
        "max_length_exceeded",
        (embedding.EmbedItem(2, "text", [0, 5, 1], "k2"), usage),
    ]
    rep = embedding.representation("sha", 3, 8, True)
    plan = embedding.plan(request({"input": ["a", "b", "c"]}), parsed, items, rep)
    assert plan.input_tokens == 6 and len(plan.items) == 2
    body = embedding.finish(plan, [[0.5, -1.0], None])
    first, second, third = body["data"]
    decoded = np.frombuffer(base64.b64decode(first["embedding"]), dtype="<f4")
    assert decoded.tolist() == [0.5, -1.0] and first["input"] == usage
    assert second == {"object": "embedding", "index": 1, "error": "max_length_exceeded"}
    assert third["error"] == "invalid_model_output"
    assert body["usage"] == {"prompt_tokens": 6, "total_tokens": 6}
    assert body["meta"]["representation"] == {
        "model_sha256": "sha",
        "layer": 3,
        "dimension": 8,
        "normalized": True,
    }
    expired = embedding.finish(plan, DEADLINE)
    assert [entry.get("error") for entry in expired["data"]] == [
        "deadline_exceeded",
        "max_length_exceeded",
        "deadline_exceeded",
    ]
