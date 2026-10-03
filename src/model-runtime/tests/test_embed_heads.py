"""The pooled and relevance heads: package detection, planning, readouts and responses."""

from __future__ import annotations

import json
import math
import time

import pytest
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from vllm_sr_runtime.accel.cpu import CPUAccelerator
from vllm_sr_runtime.engines.native import models
from vllm_sr_runtime.engines.native.models.modernbert import padded_layout
from vllm_sr_runtime.errors import PackageError
from vllm_sr_runtime.heads import embedding
from vllm_sr_runtime.heads.pooled import PooledHead
from vllm_sr_runtime.heads.relevance import RelevanceHead
from vllm_sr_runtime.plugins.base import DEADLINE, DeviceInfo, SurfaceRequest
from vllm_sr_runtime.testing import embed_packages

CPU = DeviceInfo(accelerator="cpu", index=None, name="cpu")


def request(surface, body):
    return SurfaceRequest(surface, body, None, "exact", True, time.monotonic())


def backbone(root):
    config = json.loads((root / "config.json").read_text())
    module = models.build(config["model_type"], config)
    module.load_state_dict(load_file(str(root / "model.safetensors")), strict=False)
    module.kernels = CPUAccelerator().kernels(CPU)
    return module.float().eval(), config


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


def test_vela_embedding_layout_is_detected(embedder):
    head = PooledHead.detect(
        embedder, json.loads((embedder / "config.json").read_text())
    )
    assert head.pooling == "mean" and head.normalize and not head.normalize_exits
    assert head.layers == (1, 2, 4) and head.dimensions == (64,)
    assert set(head.graphs) == {1, 2} and head.prompts == {}
    info = head.info()
    assert info.layers == (1, 2, 4) and info.input_types == ()


def test_matryoshka_dimensions_follow_the_contract(tmp_path):
    root = embed_packages.write_embedding_package(tmp_path / "wide", exits=())
    config = json.loads((root / "config.json").read_text())
    config["hidden_size"] = 768
    pooling = json.loads((root / "1_Pooling/config.json").read_text())
    (root / "1_Pooling/config.json").write_text(
        json.dumps({**pooling, "word_embedding_dimension": 768})
    )
    head = PooledHead.detect(root, config)
    assert head.dimensions == (768, 512, 256, 128, 64)
    for change in (
        {
            "representation_contract": {
                **embed_packages.EMBED_CONTRACT,
                "truncate_before_l2_normalize": False,
            }
        },
        {
            "representation_contract": {
                **embed_packages.EMBED_CONTRACT,
                "intermediate_normalization": "typo",
            }
        },
    ):
        with pytest.raises(PackageError):
            PooledHead.detect(root, {**config, **change})


def test_qwen3_embedding_layout_is_detected(qwen):
    config = json.loads((qwen / "config.json").read_text())
    head = PooledHead.detect(qwen, config)
    assert (
        head.pooling == "last_token"
        and head.layers == (2,)
        and head.dimensions == (64,)
    )
    assert head.info().input_types == ("document", "query")
    assert PooledHead.detect(qwen.parent, config) is None


def test_pooled_plans_prompts_budgets_and_keys(qwen, embedder):
    head = PooledHead.detect(qwen, json.loads((qwen / "config.json").read_text()))
    tokenizer = Tokenizer.from_file(str(qwen / "tokenizer.json"))
    plan = head.plan(
        request("embeddings", {"input": ["hello", "hello"], "input_type": "query"}),
        tokenizer,
        64,
        "sha",
    )
    plain = head.plan(request("embeddings", {"input": "hello"}), tokenizer, 64, "sha")
    query_ids, plain_ids = plan.items[0].ids, plain.items[0].ids
    assert plain_ids == [tokenizer.token_to_id("hello"), 1]
    assert len(query_ids) > len(plain_ids) and query_ids[-2:] == plain_ids
    assert (
        plan.items[0].cache_key == plan.items[1].cache_key != plain.items[0].cache_key
    )
    vela = PooledHead.detect(
        embedder, json.loads((embedder / "config.json").read_text())
    )
    vela_tokens = Tokenizer.from_file(str(embedder / "tokenizer.json"))
    first = vela.plan(
        request("embeddings", {"input": "hello", "layer": 1}), vela_tokens, 64, "sha"
    )
    full = vela.plan(request("embeddings", {"input": "hello"}), vela_tokens, 64, "sha")
    assert first.items[0].layer == 1 and full.items[0].layer == 4
    assert first.items[0].cache_key != full.items[0].cache_key
    long = vela.plan(
        request("embeddings", {"input": "hello " * 80}), vela_tokens, 64, "sha"
    )
    assert long.items == [] and long.state.slots == ["max_length_exceeded"]


def test_pooled_readout_is_masked_mean_then_view(embedder):
    module, config = backbone(embedder)
    head = PooledHead.detect(embedder, config)
    ids, mask = padded([[2, 5, 6, 7, 1], [2, 19, 1]])
    with torch.inference_mode():
        hidden = module.encode(ids, padded_layout(mask, 2, 5, module.window, "cpu"))[4]
        vectors = head.readout(hidden, mask, 64)
        alone = head.readout(
            module.encode(ids[1:, :3], padded_layout(None, 1, 3, module.window, "cpu"))[
                4
            ],
            mask[1:, :3],
            64,
        )
    expected = hidden[1, :3].sum(0) / 3
    torch.testing.assert_close(vectors[1], expected / (expected.norm() + 1e-12))
    torch.testing.assert_close(vectors[1:], alone, atol=1e-5, rtol=1e-5)
    assert torch.allclose(vectors.norm(dim=-1), torch.ones(2))


def test_last_token_readout_on_the_qwen3_backbone(qwen):
    module, config = backbone(qwen)
    head = PooledHead.detect(qwen, config)
    ids, mask = padded([[5, 6, 7, 1], [19, 1]])
    with torch.inference_mode():
        hidden = module(ids, mask)
        vectors = head.readout(hidden, mask, 64)
        alone = head.readout(module(ids[1:, :2], None), mask[1:, :2], 64)
    torch.testing.assert_close(vectors[1:], alone, atol=1e-5, rtol=1e-5)


def test_reranker_exits_and_contract(reranker):
    config = json.loads((reranker / "config.json").read_text())
    head = RelevanceHead.detect(reranker, config)
    assert head.exits == ((2, 64), (2, 32), (4, 64), (4, 32)) and head.default == (
        4,
        64,
    )
    assert RelevanceHead.detect(reranker, config, (2, 32)).default == (2, 32)
    with pytest.raises(PackageError, match="no trained pair scorer"):
        RelevanceHead.detect(reranker, config, (3, 64))
    broken = {
        **config,
        "representation_contract": {
            **config["representation_contract"],
            "pooling": "mean",
        },
    }
    with pytest.raises(PackageError, match="representation contract"):
        RelevanceHead.detect(reranker, broken)
    assert RelevanceHead.detect(reranker.parent, config) is None


def test_rerank_plan_uses_the_pair_template_once_per_query(reranker):
    head = RelevanceHead.detect(
        reranker, json.loads((reranker / "config.json").read_text())
    )
    tokenizer = Tokenizer.from_file(str(reranker / "tokenizer.json"))
    body = {
        "query": "how do i reset my password",
        "documents": ["open settings security", "  ", "offices are closed today " * 4],
    }
    plan = head.plan(request("rerank", body), tokenizer, head.exits, 24, "sha")
    assert [item.index for item in plan.items] == [0]
    assert (
        plan.items[0].ids == tokenizer.encode(body["query"], body["documents"][0]).ids
    )
    assert plan.state.slots == [0, "invalid_input", "max_length_exceeded"]
    with pytest.raises(ValueError, match="served pair scorer"):
        head.plan(
            request("rerank", {**body, "layer": 3}), tokenizer, head.exits, 24, "sha"
        )
    with pytest.raises(ValueError, match="rejects over-long"):
        head.plan(
            request("rerank", {**body, "options": {"overflow": "truncate"}}),
            tokenizer,
            head.exits,
            24,
            "sha",
        )
    other = head.plan(
        request("rerank", {**body, "layer": 2, "dimensions": 32}),
        tokenizer,
        head.exits,
        24,
        "sha",
    )
    assert (
        other.items[0].exit == (2, 32)
        and other.items[0].cache_key != plan.items[0].cache_key
    )


def test_relevance_readout_matches_the_trained_mlp(reranker):
    module, config = backbone(reranker)
    head = RelevanceHead.detect(reranker, config)
    scorers = head.load(head.exits, torch.device("cpu"))
    tensors = load_file(str(reranker / "classification_heads.safetensors"))
    ids, mask = padded([[2, 5, 6, 1, 11, 12, 1], [2, 19, 1, 22, 1]])
    with torch.inference_mode():
        hidden = module.encode(
            ids, padded_layout(mask, 2, 7, module.window, "cpu"), (2,)
        )[2]
        logits = head.readout(scorers, hidden[:, 0], (2, 32))
    cls = hidden[:, 0, :32]
    inner = torch.nn.functional.gelu(
        cls @ tensors["2.32.0.weight"].T + tensors["2.32.0.bias"]
    )
    expected = (inner @ tensors["2.32.3.weight"].T + tensors["2.32.3.bias"]).squeeze(-1)
    torch.testing.assert_close(logits, expected)


def test_rerank_response_orders_by_logit_and_keeps_failures(reranker):
    head = RelevanceHead.detect(
        reranker, json.loads((reranker / "config.json").read_text())
    )
    tokenizer = Tokenizer.from_file(str(reranker / "tokenizer.json"))
    body = {
        "query": "cat",
        "documents": ["a cat", "the router", "", "an animal"],
        "top_n": 2,
        "return_documents": True,
    }
    plan = head.plan(request("rerank", body), tokenizer, head.exits, 64, "sha")
    response = head.finish(plan, [0.5, 2.0, 0.5])
    results = response["results"]
    assert [entry["index"] for entry in results] == [1, 0, 2]
    assert results[0]["relevance_score"] == pytest.approx(1 / (1 + math.exp(-2.0)))
    assert results[0]["document"] == "the router" and results[2] == {
        "index": 2,
        "error": "invalid_input",
    }
    assert response["usage"]["input_tokens"] == plan.input_tokens
    assert response["meta"]["pair_scorer"] == {"layer": 4, "dimension": 64}
    expired = head.finish(plan, DEADLINE)["results"]
    assert {entry.get("error") for entry in expired} == {
        "deadline_exceeded",
        "invalid_input",
    }


def test_embedding_views_share_one_forward(embedder):
    module, config = backbone(embedder)
    head = PooledHead.detect(embedder, config)
    ids, mask = padded([[2, 5, 6, 1]])
    with torch.inference_mode():
        hidden = module.encode(
            ids, padded_layout(None, 1, 4, module.window, "cpu"), (4,)
        )[4]
    full, small = head.readout(hidden, mask, 64), embedding.matryoshka(
        embedding.pool(hidden, mask, "mean"), 32
    )
    torch.testing.assert_close(full[:, :32] / full[:, :32].norm(), small)
