"""The onnxruntime engine: provider choice, graph inputs and outputs, shared weights, metadata."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from vllm_srun.accel.cpu import CPUAccelerator
from vllm_srun.engines.onnxruntime import graphs, providers
from vllm_srun.engines.onnxruntime.engine import OnnxRuntimeEngine
from vllm_srun.plugins.base import (
    BackboneSpec,
    DeviceInfo,
    DtypePolicy,
    EncoderBatch,
    EngineOptions,
    ModelSpec,
)
from vllm_srun.testing import onnx_graphs

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

CPU = DeviceInfo(accelerator="cpu", index=None, name="cpu")
ROCM = DeviceInfo(accelerator="rocm", index=1, name="MI325X", bf16=True)
SCORER = {"version": 1, "layer": 2, "dimension": 4, "score_type": "relevance_logit"}


def spec(graph_paths: dict[str, Path]) -> ModelSpec:
    return ModelSpec(
        name="tiny",
        backbone=BackboneSpec(model_type="modernbert", config={}, weight_files=()),
        dtype=DtypePolicy(autocast=None),
        max_input_tokens=64,
        graphs=graph_paths,
        encoder=True,
    )


def load(graph_paths: dict[str, Path], threads: int | None = 1):
    engine = OnnxRuntimeEngine()
    model_spec = spec(graph_paths)
    assert engine.supports(model_spec, CPU) is None
    return engine.load(
        model_spec, CPUAccelerator(), CPU, EngineOptions(threads=threads)
    )


def batch(rows: list[list[int]], graph: str = "default", **kwargs) -> EncoderBatch:
    width = max(len(row) for row in rows)
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    mask = torch.zeros(len(rows), width, dtype=torch.long)
    for index, row in enumerate(rows):
        ids[index, : len(row)] = torch.tensor(row)
        mask[index, : len(row)] = 1
    return EncoderBatch(input_ids=ids, attention_mask=mask, graph=graph, **kwargs)


def reference_hidden(rows: list[list[int]], seed: int = 0) -> np.ndarray:
    table = onnx_graphs.weights((32, 8), seed)
    width = max(len(row) for row in rows)
    out = np.zeros((len(rows), width, 8), dtype=np.float32)
    for index, row in enumerate(rows):
        out[index, : len(row)] = table[row]
    return out + 0.01 * np.arange(width, dtype=np.float32)[None, :, None]


def test_provider_choice_per_accelerator():
    assert (
        providers.choose(CPU, ["CPUExecutionProvider"]).name == "CPUExecutionProvider"
    )
    rocm = providers.choose(
        ROCM, ["MIGraphXExecutionProvider", "ROCMExecutionProvider"]
    )
    assert rocm.name == "MIGraphXExecutionProvider"
    assert rocm.options == {"device_id": "1"} and rocm.gpu and not rocm.validated
    assert "none of" in providers.choose(ROCM, ["CPUExecutionProvider"])
    openvino = providers.choose(
        CPU,
        ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
        "OpenVINOExecutionProvider",
    )
    assert openvino.options == {"device_type": "CPU"} and not openvino.gpu
    assert "cannot run" in providers.choose(
        CPU, ["CPUExecutionProvider"], "CUDAExecutionProvider"
    )
    assert "no execution provider" in providers.choose(
        DeviceInfo(accelerator="mps", index=None, name="mps"), ["CPUExecutionProvider"]
    )


def test_descriptor_lists_providers_per_device():
    descriptor = OnnxRuntimeEngine.descriptor()
    assert descriptor["outputs"] == ["graph_outputs"]
    assert descriptor["providers"]["rocm"] == [
        "MIGraphXExecutionProvider",
        "ROCMExecutionProvider",
    ]
    assert descriptor["validated"] == ["CPUExecutionProvider"]
    assert "CPUExecutionProvider" in descriptor["installed"]


def test_cpu_pools_are_sized_to_the_process_and_capped():
    assert providers.cpu_threads(3) == 3 and providers.cpu_threads(None) >= 1
    assert providers.cpu_threads(16, 8) == 8 and providers.cpu_threads(4, 8) == 4
    assert providers.cpu_threads(None, 1) == 1


def test_gpu_sessions_never_fall_back_to_the_cpu():
    gpu = providers.session_options(
        providers.ProviderChoice("CUDAExecutionProvider"), 2
    )
    assert gpu.get_session_config_entry("session.disable_cpu_ep_fallback") == "1"
    with pytest.raises(RuntimeError):
        gpu.get_session_config_entry("session.intra_op.spin_duration_us")


@pytest.mark.parametrize(
    ("neighbors", "graph_spin", "spin"),
    [
        (False, None, providers.SPIN_US),
        (False, 10_000, 10_000),
        (False, 0, 0),
        (True, None, providers.NEIGHBOR_SPIN_US),
        (True, 10_000, providers.NEIGHBOR_SPIN_US),
        (True, 500, 500),
        (True, 0, 0),
    ],
)
def test_cpu_sessions_get_own_pools_whose_idle_threads_spin_briefly(
    neighbors, graph_spin, spin
):
    choice = providers.ProviderChoice("CPUExecutionProvider")
    options = providers.session_options(choice, 2, neighbors, graph_spin)
    assert options.use_per_session_threads is True
    assert options.intra_op_num_threads == 2 and options.inter_op_num_threads == 1
    assert options.get_session_config_entry(providers.SPIN_ENTRY) == str(spin)


def test_a_negative_spin_is_refused_rather_than_left_to_onnxruntimes_default():
    choice = providers.ProviderChoice("CPUExecutionProvider")
    with pytest.raises(ValueError, match="0 or more"):
        providers.session_options(choice, 2, False, -1)


def test_a_graph_that_asks_for_no_spin_never_spins(tmp_path):
    paths = {"text": onnx_graphs.token_graph(tmp_path / "text.onnx")}
    model_spec = dataclasses.replace(spec(paths), graph_spin_us={"text": 0})
    model = OnnxRuntimeEngine().load(
        model_spec, CPUAccelerator(), CPU, EngineOptions(threads=2)
    )
    assert model.receipt()["spin_us"] == {"text": 0}


@pytest.mark.parametrize(
    ("neighbors", "beside_another_engine"),
    [
        (frozenset(), False),
        (frozenset({"onnxruntime"}), False),
        (frozenset({"onnxruntime", "native"}), True),
        (frozenset({"auto"}), True),
    ],
)
def test_each_graph_gets_its_own_capped_pool(
    tmp_path, monkeypatch, neighbors, beside_another_engine
):
    seen = {}
    options = providers.session_options

    def recording(choice, threads, cpu_neighbors=False, spin_us=None):
        seen[threads] = cpu_neighbors
        return options(choice, threads, cpu_neighbors, spin_us)

    monkeypatch.setattr(providers, "session_options", recording)
    paths = {
        "text": onnx_graphs.token_graph(tmp_path / "text.onnx"),
        "image": onnx_graphs.token_graph(tmp_path / "image.onnx"),
    }
    model_spec = dataclasses.replace(
        spec(paths), graph_threads={"text": 1}, graph_spin_us={"text": 10_000}
    )
    model = OnnxRuntimeEngine().load(
        model_spec,
        CPUAccelerator(),
        CPU,
        EngineOptions(threads=2, cpu_neighbors=neighbors),
    )
    assert seen == {1: beside_another_engine, 2: beside_another_engine}
    assert model.receipt()["threads"] == {"text": 1, "image": 2}
    neighbor = providers.NEIGHBOR_SPIN_US
    assert model.receipt()["spin_us"] == (
        {"text": neighbor, "image": neighbor}
        if beside_another_engine
        else {"text": 10_000, "image": providers.SPIN_US}
    )


def test_unsupported_specs_say_why(tmp_path):
    engine = OnnxRuntimeEngine()
    assert "no ONNX graph" in engine.supports(spec({}), CPU)
    path = onnx_graphs.token_graph(tmp_path / "model.onnx")
    assert "none of" in engine.supports(spec({"default": path}), ROCM)


def test_hidden_states_with_row_and_batch_position_ids(tmp_path):
    rows = [[1, 2, 3, 4], [5, 6]]
    for layout in ("row", "batch"):
        path = onnx_graphs.token_graph(tmp_path / f"{layout}.onnx", position_ids=layout)
        model = load({"default": path})
        out = model.encode(batch(rows))
        hidden = out.outputs["last_hidden_state"]
        assert hidden.dtype == torch.float32 and hidden.shape == (2, 4, 8)
        np.testing.assert_allclose(hidden.numpy(), reference_hidden(rows), atol=1e-6)


def test_named_graphs_share_external_weights_and_report_metadata(tmp_path):
    full = onnx_graphs.token_graph(
        tmp_path / "onnx/model.onnx",
        external="weights.data",
        scorer=4,
        output="logits",
        metadata={"semantic_router.pair_scorer": json.dumps(SCORER)},
    )
    model = load(
        {"layer:2/dim:4": full, "hidden": onnx_graphs.token_graph(tmp_path / "h.onnx")}
    )
    logits = model.encode(batch([[1, 2], [3, 4, 5]], graph="layer:2/dim:4")).outputs[
        "logits"
    ]
    hidden = reference_hidden([[1, 2], [3, 4, 5]])[:, 0, :4]
    expected = hidden @ onnx_graphs.weights((4, 1), 1)
    np.testing.assert_allclose(logits.numpy(), expected, atol=1e-5)
    assert (
        json.loads(
            model.graphs["layer:2/dim:4"].facts.metadata["semantic_router.pair_scorer"]
        )
        == SCORER
    )
    receipt = model.receipt()
    assert receipt["provider"] == "CPUExecutionProvider" and receipt["validated"]
    assert set(receipt["graphs"]) == {"layer:2/dim:4", "hidden"}
    with pytest.raises(ValueError, match="no graph"):
        model.encode(batch([[1]], graph="missing"))
    with pytest.raises(ValueError, match="no outputs"):
        model.encode(batch([[1]], graph="hidden", outputs=("logits",)))


def test_parameters_count_shared_storage_once(tmp_path):
    first = onnx_graphs.token_graph(tmp_path / "onnx/a.onnx", external="weights.data")
    facts = graphs.read_graph(first)
    embedded = graphs.read_graph(onnx_graphs.token_graph(tmp_path / "b.onnx"))
    table = 32 * 8
    assert graphs.parameters([facts]) == graphs.parameters([embedded]) == table + 1 + 1
    # The same weights stored once (two graphs, one file) count once; separate storage counts twice.
    assert graphs.parameters([facts, facts]) == table + 2
    assert graphs.parameters([facts, embedded]) == 2 * (table + 2)
    model = load({"a": first})
    assert model.parameter_count() == table + 2


def test_extra_graph_inputs_and_missing_inputs(tmp_path):
    path = onnx_graphs.projection_graph(
        tmp_path / "image.onnx", {"pixel_values": [1, 3, 4, 4]}, dimension=6
    )
    model = load({"image": path})
    pixels = torch.rand(1, 3, 4, 4)
    rows = EncoderBatch(
        input_ids=torch.zeros(1, 1, dtype=torch.long),
        attention_mask=torch.ones(1, 1, dtype=torch.long),
        graph="image",
        graph_inputs={"pixel_values": pixels},
    )
    embedding = model.encode(rows).outputs["embedding"]
    assert embedding.shape == (1, 6)
    assert abs(float(embedding.norm()) - 1.0) < 1e-5
    rows.graph_inputs = {}
    with pytest.raises(ValueError, match="pixel_values"):
        model.encode(rows)


def test_truncated_protobuf_is_a_package_error(tmp_path):
    path = tmp_path / "broken.onnx"
    path.write_bytes(b"\x0a\xff")
    from vllm_srun.errors import PackageError

    with pytest.raises(PackageError):
        graphs.read_graph(path)


def test_graphs_linked_into_a_blob_store_load_like_a_hub_snapshot(tmp_path):
    exported = onnx_graphs.token_graph(
        tmp_path / "export/onnx/model.onnx", external="weights.data"
    )
    blobs, snapshot = tmp_path / "blobs", tmp_path / "snapshots/rev/onnx"
    blobs.mkdir()
    snapshot.mkdir(parents=True)
    for name in ("model.onnx", "weights.data"):
        target = blobs / f"blob-{name}"
        target.write_bytes((exported.parent / name).read_bytes())
        (snapshot / name).symlink_to(target)
    facts = graphs.read_graph(snapshot / "model.onnx")
    assert [tensor.name for tensor in facts.externals] == ["embed.weight"]
    model = load({"default": snapshot / "model.onnx"})
    hidden = model.encode(batch([[1, 2, 3]])).outputs["last_hidden_state"]
    np.testing.assert_allclose(hidden.numpy(), reference_hidden([[1, 2, 3]]), atol=1e-6)


def test_external_data_without_a_length_runs_to_the_end_of_its_file(tmp_path):
    import onnx
    from vllm_srun.errors import PackageError

    path = onnx_graphs.token_graph(
        tmp_path / "onnx/model.onnx", external="weights.data"
    )
    model = onnx.load(str(path), load_external_data=False)
    (tensor,) = [
        t
        for t in model.graph.initializer
        if t.data_location == onnx.TensorProto.EXTERNAL
    ]
    location = [entry for entry in tensor.external_data if entry.key == "location"]
    del tensor.external_data[:]
    tensor.external_data.extend(location)
    bare = tmp_path / "onnx/bare.onnx"
    onnx.save_model(model, str(bare))
    hidden = load({"default": bare}).encode(batch([[1, 2, 3]]))
    np.testing.assert_allclose(
        hidden.outputs["last_hidden_state"].numpy(),
        reference_hidden([[1, 2, 3]]),
        atol=1e-6,
    )
    (tmp_path / "onnx/weights.data").unlink()
    with pytest.raises(PackageError, match="is missing"):
        graphs.read_graph(bare)


def test_external_data_must_stay_next_to_the_graph(tmp_path):
    import onnx
    from vllm_srun.errors import PackageError

    path = onnx_graphs.token_graph(
        tmp_path / "onnx/model.onnx", external="weights.data"
    )
    model = onnx.load(str(path), load_external_data=False)
    for change, message in (
        ({"location": "../escape.data"}, "stay next to"),
        ({"length": "4"}, "wrong length"),
    ):
        edited = onnx.ModelProto()
        edited.CopyFrom(model)
        (tensor,) = [
            t
            for t in edited.graph.initializer
            if t.data_location == onnx.TensorProto.EXTERNAL
        ]
        for entry in tensor.external_data:
            entry.value = change.get(entry.key, entry.value)
        broken = tmp_path / "onnx/broken.onnx"
        onnx.save_model(edited, str(broken))
        with pytest.raises(PackageError, match=message):
            graphs.read_graph(broken)
    weights = graphs.WeightFiles()
    (tensor,) = graphs.read_graph(path).externals
    beyond = graphs.ExternalTensor(
        tensor.name, tensor.file, 1 << 20, tensor.length, tensor.dtype, tensor.shape
    )
    with pytest.raises(PackageError, match="outside"):
        weights.tensor(beyond)
