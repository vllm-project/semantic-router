"""The pooled and relevance heads: package layouts, request planning, readouts and responses."""

from __future__ import annotations

import json
import math
import time

import pytest
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from vllm_srun.accel.cpu import CPUAccelerator
from vllm_srun.engines.native import models
from vllm_srun.engines.native.models.modernbert import padded_layout
from vllm_srun.errors import PackageError
from vllm_srun.heads.pooled import EmbeddingSurface, PooledHead, PooledLayout
from vllm_srun.heads.relevance import (
    LOGITS,
    RelevanceHead,
    RelevanceLayout,
    RerankSurface,
)
from vllm_srun.heads.task import Rows
from vllm_srun.plugins.base import DEADLINE, DeviceInfo, SurfaceRequest
from vllm_srun.testing import embed_packages

CPU = DeviceInfo(accelerator="cpu", index=None, name="cpu")


def request(surface, body):
    return SurfaceRequest(surface, body, None, "exact", True, time.monotonic())


def config_of(root):
    return json.loads((root / "config.json").read_text())


def tokenizer_of(root):
    return Tokenizer.from_file(str(root / "tokenizer.json"))


def backbone(root):
    config = config_of(root)
    module = models.build(config["model_type"], config)
    module.load_state_dict(load_file(str(root / "model.safetensors")), strict=False)
    module.kernels = CPUAccelerator().kernels(CPU)
    return module.float().eval()


def packed_rows(hidden, mask):
    """Padded [B, T, H] hidden states as the family's packed ``Rows``."""
    lengths = mask.sum(1).tolist()
    starts = [sum(lengths[:row]) for row in range(len(lengths))]
    return lambda layer: Rows({layer: hidden[mask.bool()]}, starts, lengths)


def padded(rows):
    width = max(len(row) for row in rows)
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    mask = torch.zeros(len(rows), width, dtype=torch.long)
    for index, row in enumerate(rows):
        ids[index, : len(row)] = torch.tensor(row)
        mask[index, : len(row)] = 1
    return ids, mask


@pytest.fixture(scope="module")
def embedder(tmp_path_factory):
    return embed_packages.write_embedding_package(
        tmp_path_factory.mktemp("emb") / "pkg"
    )


@pytest.fixture(scope="module")
def reranker(tmp_path_factory):
    return embed_packages.write_reranker_package(tmp_path_factory.mktemp("rr") / "pkg")


@pytest.fixture(scope="module")
def qwen(tmp_path_factory):
    return embed_packages.write_qwen3_embedding_package(
        tmp_path_factory.mktemp("qw") / "pkg"
    )


def test_vela_embedding_layout(embedder):
    layout = PooledLayout.read(embedder, config_of(embedder))
    assert layout.pooling == "mean" and layout.normalize and not layout.normalize_exits
    assert layout.layers == (1, 2, 4) and layout.dimensions == (64,)
    assert set(layout.graphs) == {1, 2} and layout.prompts == {}
    assert (
        "onnx/model_layer_1.onnx" in layout.files
        and "onnx/weights.data" not in layout.files
    )
    assert layout.info([4]).layers == (4,) and layout.info().input_types == ()


def test_matryoshka_dimensions_follow_the_contract(tmp_path):
    root = embed_packages.write_embedding_package(tmp_path / "wide", exits=())
    config = config_of(root)
    config["hidden_size"] = 768
    pooling = json.loads((root / "1_Pooling/config.json").read_text())
    (root / "1_Pooling/config.json").write_text(
        json.dumps({**pooling, "word_embedding_dimension": 768})
    )
    assert PooledLayout.read(root, config).dimensions == (768, 512, 256, 128, 64)
    for change in (
        {"truncate_before_l2_normalize": False},
        {"intermediate_normalization": "typo"},
    ):
        contract = {**embed_packages.EMBED_CONTRACT, **change}
        with pytest.raises(PackageError):
            PooledLayout.read(root, {**config, "representation_contract": contract})


def test_qwen3_embedding_layout(qwen):
    layout = PooledLayout.read(qwen, config_of(qwen))
    assert layout.pooling == "last_token" and layout.layers == (2,)
    assert layout.info().input_types == ("document", "query")


def test_embedding_plans_prompts_budgets_and_shared_keys(qwen, embedder):
    surface = EmbeddingSurface(
        PooledLayout.read(qwen, config_of(qwen)), tokenizer_of(qwen), [2]
    )
    query = surface.plan(
        request("embeddings", {"input": ["hello"] * 2, "input_type": "query"}),
        "sha",
        64,
    )
    plain = surface.plan(
        request("embeddings", {"input": "hello", "dimensions": 64}), "sha", 64
    )
    hello = tokenizer_of(qwen).token_to_id("hello")
    assert plain.items[0].ids == (hello, 1) and query.items[0].ids[-2:] == (hello, 1)
    assert (
        query.items[0].cache_key == query.items[1].cache_key != plain.items[0].cache_key
    )
    vela = EmbeddingSurface(
        PooledLayout.read(embedder, config_of(embedder)),
        tokenizer_of(embedder),
        [1, 2, 4],
    )
    first = vela.plan(request("embeddings", {"input": "hello", "layer": 1}), "sha", 64)
    full = vela.plan(request("embeddings", {"input": "hello"}), "sha", 64)
    assert (first.items[0].head, first.items[0].layer) == ("pooled@1", 1)
    assert (
        full.items[0].head == "pooled@4"
        and first.items[0].cache_key != full.items[0].cache_key
    )
    long = vela.plan(request("embeddings", {"input": "hello " * 80}), "sha", 64)
    assert long.items == [] and long.state.slots == ["max_length_exceeded"]


def test_pooled_readout_and_matryoshka_views(embedder):
    module = backbone(embedder)
    layout = PooledLayout.read(embedder, config_of(embedder))
    ids, mask = padded([[2, 5, 6, 7, 1], [2, 19, 1]])
    with torch.inference_mode():
        hidden = module.encode(
            ids, padded_layout(mask, 2, 5, module.window, "cpu"), (4,)
        )[4]
    vectors = PooledHead(layout, 4).readout(packed_rows(hidden, mask)(4), [0, 1])
    torch.testing.assert_close(vectors[1], hidden[1, :3].sum(0) / 3)
    surface = EmbeddingSurface(layout, tokenizer_of(embedder), [4])
    plan = surface.plan(request("embeddings", {"input": ["a", "b"]}), "sha", 64)
    body = surface.finish(plan, vectors)
    first = torch.tensor(body["data"][0]["embedding"])
    torch.testing.assert_close(first, vectors[0] / (vectors[0].norm() + 1e-12))


def test_last_token_readout_on_the_qwen3_backbone(qwen):
    module = backbone(qwen)
    layout = PooledLayout.read(qwen, config_of(qwen))
    ids, mask = padded([[5, 6, 7, 1], [19, 1]])
    with torch.inference_mode():
        hidden = module(ids, mask)
        alone = module(ids[1:, :2], None)
    (_, last) = PooledHead(layout, 2).readout(packed_rows(hidden, mask)(2), [0, 1])
    torch.testing.assert_close(last, alone[0, -1], atol=1e-5, rtol=1e-5)


def test_reranker_layout(reranker):
    config = config_of(reranker)
    layout = RelevanceLayout.read(reranker, config)
    assert layout.exits == ((2, 64), (2, 32), (4, 64), (4, 32)) and layout.default == (
        4,
        64,
    )
    assert RelevanceLayout.read(reranker, config, (2, 32)).default == (2, 32)
    assert "special_tokens_map.json" not in layout.files(reranker, [(4, 64)])
    with pytest.raises(PackageError, match="no trained pair scorer"):
        RelevanceLayout.read(reranker, config, (3, 64))
    contract = {**config["representation_contract"], "pooling": "mean"}
    with pytest.raises(PackageError, match="representation contract"):
        RelevanceLayout.read(reranker, {**config, "representation_contract": contract})


def surface_of(reranker, exits=None):
    layout = RelevanceLayout.read(reranker, config_of(reranker))
    exits = exits or layout.exits
    scorers = layout.scorers(exits)
    heads = {exit: RelevanceHead(exit, scorers[exit]) for exit in exits}
    return RerankSurface(layout, tokenizer_of(reranker), heads)


def test_rerank_plan_uses_the_pair_template_once_per_query(reranker):
    surface = surface_of(reranker)
    tokenizer = tokenizer_of(reranker)
    body = {
        "query": "how do i reset my password",
        "documents": ["open settings security", "  ", "offices are closed today " * 4],
    }
    plan = surface.plan(request("rerank", body), "sha", 24)
    assert plan.items[0].ids == tuple(
        tokenizer.encode(body["query"], body["documents"][0]).ids
    )
    assert plan.items[0].head == "relevance@4x64"
    assert plan.state.slots == [0, "invalid_input", "max_length_exceeded"]
    with pytest.raises(ValueError, match="served pair scorer"):
        surface.plan(request("rerank", {**body, "layer": 3}), "sha", 24)
    with pytest.raises(ValueError, match="rejects over-long"):
        surface.plan(
            request("rerank", {**body, "options": {"overflow": "truncate"}}), "sha", 24
        )
    other = surface.plan(
        request("rerank", {**body, "layer": 2, "dimensions": 32}), "sha", 24
    )
    assert other.items[0].head == "relevance@2x32"
    assert other.items[0].cache_key != plan.items[0].cache_key


def test_relevance_readout_matches_the_trained_mlp(reranker):
    module = backbone(reranker)
    surface = surface_of(reranker)
    tensors = load_file(str(reranker / "classification_heads.safetensors"))
    ids, mask = padded([[2, 5, 6, 1, 11, 12, 1], [2, 19, 1, 22, 1]])
    with torch.inference_mode():
        hidden = module.encode(
            ids, padded_layout(mask, 2, 7, module.window, "cpu"), (2,), True
        )[2]
        logits = surface.heads[(2, 32)].readout(packed_rows(hidden, mask)(2), [0, 1])
    cls = hidden[:, 0, :32]
    inner = torch.nn.functional.gelu(
        cls @ tensors["2.32.0.weight"].T + tensors["2.32.0.bias"]
    )
    expected = (inner @ tensors["2.32.3.weight"].T + tensors["2.32.3.bias"]).squeeze(-1)
    torch.testing.assert_close(torch.tensor(logits), expected)
    graph_rows = Rows({}, [], [7, 5], {LOGITS: torch.tensor([[0.25], [-1.5]])})
    assert RelevanceHead((4, 64), None).readout(graph_rows, [1, 0]) == [-1.5, 0.25]


def test_rerank_response_orders_by_logit_and_keeps_failures(reranker):
    surface = surface_of(reranker)
    body = {
        "query": "cat",
        "documents": ["a cat", "the router", "", "an animal"],
        "top_n": 2,
        "return_documents": True,
    }
    plan = surface.plan(request("rerank", body), "sha", 64)
    response = surface.finish(plan, [0.5, 2.0, 0.5])
    results = response["results"]
    assert [entry["index"] for entry in results] == [1, 0, 2]
    assert results[0]["relevance_score"] == pytest.approx(1 / (1 + math.exp(-2.0)))
    assert results[0]["document"] == "the router"
    assert results[2] == {"index": 2, "error": "invalid_input"}
    assert response["usage"]["input_tokens"] == plan.input_tokens
    assert response["meta"]["pair_scorer"] == {"layer": 4, "dimension": 64}
    expired = surface.finish(plan, DEADLINE)["results"]
    assert {entry.get("error") for entry in expired} == {
        "deadline_exceeded",
        "invalid_input",
    }
