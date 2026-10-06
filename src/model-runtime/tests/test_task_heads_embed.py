"""task_heads embedders and rerankers end to end: verification, both engines, surfaces and bundles."""

from __future__ import annotations

import importlib.util
import time
from dataclasses import replace

import numpy as np
import pytest
import torch
from vllm_srun.accel import onednn
from vllm_srun.accel.cpu import CPUAccelerator
from vllm_srun.engines.native.engine import NativeEngine
from vllm_srun.errors import PackageError
from vllm_srun.families.task_heads.family import (
    TaskHeadsFamily,
    batch_invariant,
    length_buckets,
)
from vllm_srun.heads.task import identical
from vllm_srun.plugins.base import (
    DeviceInfo,
    EngineOptions,
    PackageRef,
    RegistryOptions,
    SurfaceRequest,
    UnsupportedSurfaceError,
)
from vllm_srun.profiles.exact import ExactProfile
from vllm_srun.testing import embed_packages

CPU = DeviceInfo(accelerator="cpu", index=None, name="cpu")


# Graph fixtures need the onnx package; native tests run without it.
HAS_ONNX = importlib.util.find_spec("onnx") is not None


def load(root, engine=None, **model_options):
    family = TaskHeadsFamily(RegistryOptions(model_options=model_options))
    assert family.detect(PackageRef(root))
    package = family.verify(PackageRef(root))
    spec = family.describe(package)
    engine = engine or NativeEngine()
    assert engine.supports(spec, CPU) is None
    engine_model = engine.load(spec, CPUAccelerator(), CPU, EngineOptions(threads=1))
    return family, package, family.load(package, spec, engine_model)


def serve(model, surface, body):
    request = SurfaceRequest(surface, body, None, "exact", True, time.monotonic())
    plan = model.plan_surface(surface, request)
    return plan, model.finish_surface(plan, model.run(plan.items))


@pytest.fixture(scope="module")
def embedder(tmp_path_factory):
    return embed_packages.write_embedding_package(
        tmp_path_factory.mktemp("emb") / "pkg", exits=(1, 2), graphs=HAS_ONNX
    )


@pytest.fixture(scope="module")
def reranker(tmp_path_factory):
    return embed_packages.write_reranker_package(
        tmp_path_factory.mktemp("rr") / "pkg", graphs=HAS_ONNX
    )


def test_buckets_bound_padding_and_tokens():
    lengths = [16, 250, 17, 240, 18, 128, 130]
    buckets = length_buckets(lengths, 1024)
    assert sorted(i for bucket in buckets for i in bucket) == list(range(len(lengths)))
    for bucket in buckets:
        widest = max(lengths[i] for i in bucket)
        real = sum(lengths[i] for i in bucket)
        assert widest * len(bucket) <= 1024 and widest * len(bucket) <= real * 1.25
    assert [len(bucket) for bucket in length_buckets([8] * 10, 32)] == [4, 4, 2]


def test_native_embedder_serves_every_exit_from_one_forward(embedder):
    family, package, model = load(embedder)
    assert model.info.surfaces == ("embeddings",)
    assert model.info.embedding.layers == (1, 2, 4) and model.normalize_exits is False
    assert sorted(model.heads) == ["pooled@1", "pooled@2", "pooled@4"]
    _, full = serve(model, "embeddings", {"input": ["hello", "a cat is an animal"]})
    _, early = serve(model, "embeddings", {"input": ["hello"], "layer": 1})
    vectors = [np.asarray(entry["embedding"]) for entry in full["data"]]
    assert all(abs(np.linalg.norm(v) - 1) < 1e-5 for v in vectors)
    assert not np.allclose(early["data"][0]["embedding"], vectors[0])
    assert full["meta"]["representation"] == {
        "model_sha256": package.model_sha256,
        "layer": 4,
        "dimension": 64,
        "normalized": True,
    }
    with pytest.raises(UnsupportedSurfaceError):
        model.plan_surface(
            "classify",
            SurfaceRequest("classify", {"input": "x"}, None, "exact", True, 0.0),
        )
    (golden,) = family.golden(package)
    assert golden["surface"] == "embeddings"


def embeddings_request(body):
    return SurfaceRequest("embeddings", body, None, "exact", True, 0.0)


def test_bundled_exits_share_the_packed_forward(embedder):
    _, _, model = load(embedder)
    first = model.plan_surface(
        "embeddings", embeddings_request({"input": ["hello"], "layer": 1})
    )
    last = model.plan_surface(
        "embeddings", embeddings_request({"input": ["hello", "a cat"]})
    )
    results = model.run(first.items + last.items)
    fused = model.finish_surface(last, results[1:])
    _, alone = serve(model, "embeddings", {"input": ["hello", "a cat"]})
    np.testing.assert_allclose(
        [e["embedding"] for e in fused["data"]],
        [e["embedding"] for e in alone["data"]],
        atol=1e-6,
    )


def test_native_reranker_serves_every_trained_exit(reranker):
    _, _, model = load(reranker)
    assert model.info.surfaces == ("rerank",) and model.normalize_exits is True
    assert model.info.rerank.exits == ((2, 64), (2, 32), (4, 64), (4, 32))
    body = {
        "query": "how do i reset my password",
        "documents": ["open settings security", "offices are closed"],
    }
    _, response = serve(model, "rerank", body)
    _, other = serve(model, "rerank", {**body, "layer": 2, "dimensions": 32})
    assert [r["index"] for r in response["results"]] in ([0, 1], [1, 0])
    assert response["results"][0]["logit"] >= response["results"][1]["logit"]
    assert other["meta"]["pair_scorer"] == {"layer": 2, "dimension": 32}
    _, _, pinned = load(reranker, pair_scorer={"layer": 2, "dimension": 32})
    _, default = serve(pinned, "rerank", body)
    assert default["meta"]["pair_scorer"] == {"layer": 2, "dimension": 32}
    with pytest.raises(PackageError, match="pair_scorer"):
        load(reranker, pair_scorer=[{"layer": 2}])


def test_exact_merges_native_cpu_embedders_and_rerankers_where_the_probe_holds(
    embedder, reranker
):
    if not onednn.available():
        pytest.skip("oneDNN's packed linear needs an x86 CPU")
    docs = ["open settings security", "offices are closed today", "the router"]
    bodies = {
        "embeddings": [
            {"input": text}
            for text in ("hi", "reset my password", "the router " * 40, "a cat")
        ],
        "rerank": [
            {"query": "reset my password", "documents": docs},
            {"query": "when do the offices open", "documents": docs[:1]},
        ],
    }
    for root, surface in ((embedder, "embeddings"), (reranker, "rerank")):
        _, _, model = load(root)
        profile = ExactProfile()
        profile.bind(model)
        assert profile.merge == model.batch_invariant
        assert model.packs_rows and not profile.banded
        if not model.batch_invariant:
            continue
        plans = [
            model.plan_surface(
                surface,
                SurfaceRequest(surface, body, None, "exact", True, time.monotonic()),
            )
            for body in bodies[surface]
        ]
        alone = [value for plan in plans[::-1] for value in model.run(plan.items)]
        together = model.run([item for plan in plans[::-1] for item in plan.items])
        assert len(together) == len(alone)
        assert all(map(identical, alone, together))
    _, package, model = load(embedder)
    head = model.heads["pooled@4"]
    readout = head.readout
    head.readout = lambda rows, sequences: [
        vector + 1e-7 * len(sequences) for vector in readout(rows, sequences)
    ]
    assert not batch_invariant(model, package.details["package"].config["vocab_size"])


def test_graph_engine_serves_the_configured_exits(embedder, reranker):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from vllm_srun.engines.onnxruntime.engine import OnnxRuntimeEngine

    _, _, model = load(embedder, OnnxRuntimeEngine(), layers=[2, 4])
    assert model.info.embedding.layers == (2, 4)
    texts = ["hello", "a cat is an animal", "reset my password"]
    _, body = serve(model, "embeddings", {"input": texts, "layer": 2})
    assert [len(entry["embedding"]) for entry in body["data"]] == [64] * 3
    with pytest.raises(ValueError, match="layer must be one of"):
        serve(model, "embeddings", {"input": "hello", "layer": 1})
    with pytest.raises(PackageError, match="no exit graph"):
        load(embedder, OnnxRuntimeEngine(), layers=[3])
    _, _, scorer = load(
        reranker, OnnxRuntimeEngine(), pair_scorers=[{"layer": 2, "dimension": 32}]
    )
    assert scorer.info.rerank.exits == ((4, 64), (2, 32))
    docs = ["open settings security", "offices are closed today", "the router"]
    _, response = serve(
        scorer, "rerank", {"query": "reset my password", "documents": docs}
    )
    logits = [r["logit"] for r in response["results"]]
    assert logits == sorted(logits, reverse=True) and all(np.isfinite(logits))


def test_auto_tries_the_tables_engine_first_where_it_runs(embedder):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from vllm_srun.runtime import choose_engine

    family = TaskHeadsFamily(RegistryOptions())
    spec = family.describe(family.verify(PackageRef(embedder)))
    assert choose_engine("auto", spec, CPU)[0] == "native"
    assert choose_engine("auto", spec, CPU, "onnxruntime")[0] == "onnxruntime"
    assert (
        choose_engine("auto", replace(spec, graphs={}), CPU, "onnxruntime")[0]
        == "native"
    )
    assert choose_engine("native", spec, CPU, "onnxruntime")[0] == "native"


def test_graph_runs_keep_torch_on_one_thread(embedder):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from vllm_srun.engines.onnxruntime.engine import OnnxRuntimeEngine

    _, _, model = load(embedder, OnnxRuntimeEngine(), layers=[2])
    seen = []
    encode = model.engine_model.encode

    def spy(batch):
        seen.append(torch.get_num_threads())
        return encode(batch)

    model.engine_model.encode = spy
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        serve(model, "embeddings", {"input": ["hello", "a cat"], "layer": 2})
        assert seen == [1] and torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(threads)


def test_qwen3_embedder_on_the_native_decoder(tmp_path):
    root = embed_packages.write_qwen3_embedding_package(tmp_path / "qw")
    _, _, model = load(root)
    assert model.info.embedding.pooling == "last_token"
    assert model.info.embedding.input_types == ("document", "query")
    plan, body = serve(model, "embeddings", {"input": ["hello", "a cat is an animal"]})
    _, query = serve(model, "embeddings", {"input": ["hello"], "input_type": "query"})
    assert len(plan.items) == 2 and not np.allclose(
        query["data"][0]["embedding"], body["data"][0]["embedding"]
    )
    alone = model.engine_model.backbone(
        torch.tensor([list(plan.items[0].ids)]),
        torch.ones(1, len(plan.items[0].ids), dtype=torch.long),
    )[0, -1]
    expected = (alone / (alone.norm() + 1e-12)).tolist()
    np.testing.assert_allclose(body["data"][0]["embedding"], expected, atol=1e-5)
