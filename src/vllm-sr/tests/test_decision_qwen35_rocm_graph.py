"""Dependency-light isolation and failure tests for optional Qwen ROCm graphs."""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision_runtime import qwen35_rocm_graph as graph_module  # noqa: E402
from decision_runtime.qwen35_torch import Qwen35TorchRuntime  # noqa: E402
from decision_runtime.row_executor import _finish_thread_inference  # noqa: E402


class _Tensor:
    def __init__(self, shape, value, *, dtype="torch.int64"):
        self.shape = shape
        self.value = value
        self.dtype = dtype
        self.device = "cuda:0"
        self.copies = []

    def copy_(self, other):
        self.value = other.value
        self.copies.append(other.value)
        return self


class _Model:
    def __init__(self):
        self.backbone = SimpleNamespace(
            config=SimpleNamespace(_attn_implementation="sdpa")
        )
        self.eager = []

    def __call__(self, **batch):
        value = batch["input_ids"].value
        self.eager.append(value)
        return f"eager:{value}"


class _Graph:
    def __init__(self):
        self.replays = 0

    def replay(self):
        self.replays += 1


def _runtime_and_graphs(monkeypatch):
    model = _Model()
    cuda = SimpleNamespace(
        mem_get_info=lambda device: (16 << 30, 256 << 30),
        memory_allocated=lambda device: 0,
        memory_reserved=lambda device: 0,
        max_memory_allocated=lambda device: 0,
        max_memory_reserved=lambda device: 0,
        reset_peak_memory_stats=lambda device: None,
    )
    runtime = Qwen35TorchRuntime(
        model=model,
        tokenizer=object(),
        torch=SimpleNamespace(cuda=cuda),
        device="cuda:0",
        temperature=1.0,
        max_length=16384,
        backend="rocm",
        rocm_profile_binding=SimpleNamespace(
            profile_sha256="1" * 64,
            kernel_config_sha256="2" * 64,
            guard_contract_sha256="3" * 64,
            strict=True,
            source_hashes_verified=True,
            strict_cache_enforced=True,
            unknown_key_guard_enforced=True,
        ),
    )
    graphs = graph_module.QwenRocmBackboneGraphs(runtime, artifact_content_id="a" * 64)
    runtime.rocm_graphs = graphs
    monkeypatch.setattr(graphs, "_exact_masks", lambda batch: batch["masks"])
    monkeypatch.setattr(
        graphs, "_same_valid_logits_and_probabilities", lambda *args: True
    )
    monkeypatch.setattr(
        graphs,
        "_logits_from_hidden",
        lambda batch, hidden: f"graph:{batch['input_ids'].value}",
    )
    entries = []

    def capture(batch, masks):
        entry = graph_module._GraphEntry(
            graph=_Graph(),
            hidden=object(),
            static_ids=_Tensor(batch["input_ids"].shape, "initial"),
            static_masks={
                name: None if value is None else _Tensor(value.shape, "initial")
                for name, value in masks.items()
            },
            allocated_bytes=0,
        )
        entries.append(entry)
        return entry

    monkeypatch.setattr(graphs, "_capture", capture)
    monkeypatch.setattr(
        graph_module, "assert_qwen_rocm_graph_replay_safe", lambda *args, **kw: None
    )
    return runtime, graphs, entries


def _batch(value: str, *, rows: int = 8, tokens: int = 128, padded=True):
    masks = {
        "full_attention": (
            _Tensor((rows, 1, tokens, tokens), f"full:{value}", dtype="torch.bool")
            if padded
            else None
        ),
        "linear_attention": (
            _Tensor((rows, tokens), f"linear:{value}") if padded else None
        ),
    }
    return {"input_ids": _Tensor((rows, tokens), value), "masks": masks}


def test_changed_content_replays_copy_every_static_id_and_mask(monkeypatch):
    _, graphs, entries = _runtime_and_graphs(monkeypatch)
    assert graphs.logits(_batch("A")) == "eager:A"  # qualification stays eager
    assert graphs.logits(_batch("B")) == "graph:B"
    assert graphs.logits(_batch("A")) == "graph:A"
    assert len(entries) == 1
    assert entries[0].graph.replays == 3  # qualification, B, then A
    assert entries[0].static_ids.copies == ["A", "B", "A"]
    assert entries[0].static_masks["full_attention"].copies == [
        "full:A",
        "full:B",
        "full:A",
    ]
    assert entries[0].static_masks["linear_attention"].copies == [
        "linear:A",
        "linear:B",
        "linear:A",
    ]


def test_replay_guard_failure_never_calls_captured_graph(monkeypatch):
    _, graphs, entries = _runtime_and_graphs(monkeypatch)
    graphs.logits(_batch("A"))
    before = entries[0].graph.replays

    def reject(*args, **kwargs):
        raise graph_module.QwenRocmBindingError("profile changed")

    monkeypatch.setattr(graph_module, "assert_qwen_rocm_graph_replay_safe", reject)
    with pytest.raises(graph_module.QwenRocmBindingError, match="profile changed"):
        graphs.logits(_batch("B"))
    assert entries[0].graph.replays == before


def test_shape_mask_and_memory_bounds_fall_back_to_eager(monkeypatch):
    _, graphs, entries = _runtime_and_graphs(monkeypatch)
    assert graphs.logits(_batch("short", rows=7)) == "eager:short"
    assert graphs.logits(_batch("long", tokens=288)) == "eager:long"
    assert entries == []
    graphs.logits(_batch("A"))
    graphs.logits(_batch("unpadded", padded=False))
    assert len(entries) == 2  # different mask topology has a distinct key
    assert graphs.logits(_batch("third", tokens=64)) == "eager:third"
    assert len(entries) == graph_module._MAX_GRAPHS


def test_overbudget_capture_returns_ordinary_eager_and_is_not_retried(monkeypatch):
    runtime, graphs, entries = _runtime_and_graphs(monkeypatch)
    allocated = [0]
    runtime.torch.cuda.memory_allocated = lambda device: allocated[0]
    original_capture = graphs._capture

    def costly_capture(batch, masks):
        allocated[0] += graph_module._MAX_CAPTURE_BYTES + 1
        return original_capture(batch, masks)

    monkeypatch.setattr(graphs, "_capture", costly_capture)
    assert graphs.logits(_batch("A")) == "eager:A"
    assert len(entries) == 1
    assert not graphs._graphs
    assert graphs.logits(_batch("B")) == "eager:B"
    assert len(entries) == 1


def test_unsupported_capture_returns_ordinary_eager_and_is_not_retried(monkeypatch):
    _, graphs, _ = _runtime_and_graphs(monkeypatch)

    def fail_capture(batch, masks):
        raise RuntimeError("unsupported HIP capture")

    monkeypatch.setattr(graphs, "_capture", fail_capture)
    assert graphs.logits(_batch("A")) == "eager:A"
    assert not graphs._graphs
    assert graphs.logits(_batch("B")) == "eager:B"


def test_insufficient_hbm_returns_ordinary_eager_and_is_not_retried(monkeypatch):
    runtime, graphs, entries = _runtime_and_graphs(monkeypatch)
    runtime.torch.cuda.mem_get_info = lambda device: (7 << 30, 256 << 30)
    assert graphs.logits(_batch("A")) == "eager:A"
    assert not entries
    assert graphs.logits(_batch("B")) == "eager:B"
    assert not entries


def test_postcapture_hbm_shortage_discards_graph_and_returns_eager(monkeypatch):
    runtime, graphs, entries = _runtime_and_graphs(monkeypatch)
    readings = iter((16 << 30, 7 << 30))
    runtime.torch.cuda.mem_get_info = lambda device: (next(readings), 256 << 30)
    assert graphs.logits(_batch("A")) == "eager:A"
    assert len(entries) == 1
    assert not graphs._graphs
    assert graphs.logits(_batch("B")) == "eager:B"


def test_strict_binding_failure_during_capture_is_not_hidden(monkeypatch):
    _, graphs, _ = _runtime_and_graphs(monkeypatch)

    def reject_capture(batch, masks):
        raise graph_module.QwenRocmBindingError("strict FLA tamper")

    monkeypatch.setattr(graphs, "_capture", reject_capture)
    with pytest.raises(graph_module.QwenRocmBindingError, match="strict FLA tamper"):
        graphs.logits(_batch("A"))


@pytest.mark.parametrize(
    ("qualifying_mask", "stage"),
    ((True, "exact-mask"), (False, "captured-replay")),
)
def test_parity_failure_rejects_graph_and_returns_eager_with_content_free_warning(
    monkeypatch, caplog, qualifying_mask, stage
):
    _, graphs, entries = _runtime_and_graphs(monkeypatch)
    calls = []

    def mismatch_on_selected_pass(*args):
        calls.append(True)
        return len(calls) != (1 if qualifying_mask else 2)

    monkeypatch.setattr(
        graphs, "_same_valid_logits_and_probabilities", mismatch_on_selected_pass
    )
    with caplog.at_level("WARNING"):
        assert graphs.logits(_batch("PRIVATE-QUESTION")) == "eager:PRIVATE-QUESTION"
        assert graphs.logits(_batch("PRIVATE-QUESTION")) == "eager:PRIVATE-QUESTION"
    assert not graphs._graphs
    assert len(entries) == (0 if qualifying_mask else 1)
    assert len(caplog.records) == 1
    assert stage in caplog.text
    assert "B8/T128" in caplog.text
    assert "PRIVATE-QUESTION" not in caplog.text


def test_parity_warning_count_is_bounded_across_rejected_shapes(monkeypatch, caplog):
    _, graphs, _ = _runtime_and_graphs(monkeypatch)
    monkeypatch.setattr(
        graphs, "_same_valid_logits_and_probabilities", lambda *args: False
    )
    with caplog.at_level("WARNING"):
        for tokens in (32, 64, 96, 128, 160):
            assert graphs.logits(_batch("private", tokens=tokens)) == "eager:private"
    assert len(caplog.records) == graph_module._MAX_PARITY_WARNINGS
    assert "private" not in caplog.text


def test_cancelled_thread_keeps_graph_lock_until_host_completion(monkeypatch):
    entered = []
    started = threading.Event()
    release = threading.Event()
    runtime = Qwen35TorchRuntime(
        model=None,
        tokenizer=None,
        torch=None,
        device=None,
        temperature=1.0,
        max_length=128,
        backend="rocm",
        rocm_profile_binding=None,
        rocm_graphs=SimpleNamespace(lock=threading.RLock()),
    )

    def blocked(self, rows):
        entered.append(rows)
        if rows == ("first",):
            started.set()
            assert release.wait(2)
        return rows

    monkeypatch.setattr(Qwen35TorchRuntime, "_predict_encoded_locked", blocked)

    async def exercise():
        first = asyncio.create_task(
            _finish_thread_inference(runtime.predict_encoded, ("first",))
        )
        assert await asyncio.to_thread(started.wait, 2)
        first.cancel()
        second = asyncio.create_task(
            _finish_thread_inference(runtime.predict_encoded, ("second",))
        )
        try:
            await asyncio.sleep(0.05)
            assert entered == [("first",)]
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert await second == ("second",)

    asyncio.run(exercise())
