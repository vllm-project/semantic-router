"""Gated LoRA (Vela 3.0 d3 sizes): one adapter on per forward, on some rows only, base(x) bit for bit elsewhere."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from vllm_srun.accel.cpu import CPUAccelerator
from vllm_srun.engines.native.engine import NativeEngine
from vllm_srun.engines.native.models.gated_lora import Gate, GatedLinear
from vllm_srun.families.decision3.family import Decision3Family
from vllm_srun.plugins.base import (
    AdapterGate,
    EngineOptions,
    ForwardBatch,
    GatedAdapterSpec,
    PackageRef,
)
from vllm_srun.testing import vela3_d3 as fixture

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def reference(base: torch.nn.Linear, x, a, b, scaling):
    """The released ``MultiLoRA`` with that adapter on: base(x) + ((x.float() @ A.T) @ B.T * s).to(y.dtype)."""
    y = base(x)
    return y + ((x.float() @ a.T) @ b.T * scaling).to(y.dtype)


@pytest.fixture
def linear():
    torch.manual_seed(0)
    base = torch.nn.Linear(8, 6).to(torch.bfloat16)
    gate = Gate()
    layer = GatedLinear(base, gate)
    a, b = torch.randn(2, 8), torch.randn(6, 2)
    layer.add("one", a, b, 1.5)
    layer.add("two", torch.randn(2, 8), torch.randn(6, 2), 0.5)
    x = torch.randn(3, 4, 8).to(torch.bfloat16)
    return base, gate, layer, a, b, x


def test_off_is_the_base_bit_for_bit(linear) -> None:
    base, _gate, layer, _a, _b, x = linear
    assert torch.equal(layer(x), base(x))


def test_on_is_the_released_adapter_math(linear) -> None:
    base, gate, layer, a, b, x = linear
    with gate.on(AdapterGate("one")):
        out = layer(x)
    expected = reference(base, x, a, b, 1.5)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, expected)
    assert not torch.equal(out, base(x))
    assert torch.equal(layer(x), base(x))  # the gate closes after the block


def test_rows_off_the_gate_keep_the_base(linear) -> None:
    base, gate, layer, a, b, x = linear
    with gate.on(AdapterGate("one", (True, False, True))):
        out = layer(x)
    expected = reference(base, x, a, b, 1.5)
    assert torch.equal(out[0], expected[0]) and torch.equal(out[2], expected[2])
    assert torch.equal(out[1], base(x)[1])


def test_a_layer_without_that_adapter_is_the_base(linear) -> None:
    base, gate, layer, _a, _b, x = linear
    with gate.on(AdapterGate("absent")):
        assert torch.equal(layer(x), base(x))


def test_a_row_gate_must_cover_the_batch(linear) -> None:
    _base, gate, layer, _a, _b, x = linear
    with gate.on(AdapterGate("one", (True, False))), pytest.raises(ValueError):
        layer(x)


@pytest.fixture(scope="module")
def written(tmp_path_factory) -> Path:
    return fixture.write_fixture(tmp_path_factory.mktemp("gated") / "pkg", "text", 5)


def load(written: Path, gated: bool):
    base = written.parent / f"{written.name}-base"
    family = Decision3Family()
    verified = family.verify(PackageRef(base))
    spec = family.describe(verified)
    if gated:
        adapters = {
            name: GatedAdapterSpec(written / file, rank, alpha)
            for name, (file, rank, alpha, _first) in fixture.ADAPTERS.items()
        }
        spec = replace(spec, backbone=replace(spec.backbone, gated=adapters))
    accelerator = CPUAccelerator()
    return NativeEngine().load(
        spec, accelerator, accelerator.devices()[0], EngineOptions()
    )


def batch(adapter: AdapterGate | None = None) -> ForwardBatch:
    ids = torch.tensor([[5, 9, 12, 7, 3, 4], [0, 0, 8, 11, 6, 2]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]])
    return ForwardBatch(
        input_ids=ids,
        attention_mask=mask,
        gather=torch.tensor([[1, 5], [3, 5]]),
        query=torch.tensor([5, 5]),
        lengths=[6, 4],
        adapter=adapter,
    )


@pytest.fixture(scope="module")
def engines(written):
    return load(written, gated=False), load(written, gated=True)


def test_engine_keeps_the_base_with_every_adapter_off(engines) -> None:
    plain, gated = engines
    assert gated.parameter_count() == plain.parameter_count()
    want, got = plain.forward(batch()), gated.forward(batch())
    assert torch.equal(got.query, want.query)
    assert torch.equal(got.gathered, want.gathered)


def test_engine_gates_rows(engines) -> None:
    plain, gated = engines
    base = plain.forward(batch()).query
    every = gated.forward(batch(AdapterGate("decision"))).query
    some = gated.forward(batch(AdapterGate("decision", (False, True)))).query
    assert not torch.equal(every[0], base[0]) and not torch.equal(every[1], base[1])
    assert torch.equal(some[0], base[0])
    assert torch.equal(some[1], every[1])
    other = gated.forward(batch(AdapterGate("decision_df"))).query
    assert not torch.equal(other, every)


def test_engine_refuses_an_unknown_adapter_and_packed_gates(engines) -> None:
    _plain, gated = engines
    with pytest.raises(ValueError, match="adapter"):
        gated.forward(batch(AdapterGate("missing")))
    packed = batch(AdapterGate("decision", (True, False)))
    packed.packed = True
    with pytest.raises(ValueError, match="adapter"):
        gated.forward(packed)
