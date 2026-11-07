"""The direct convolution probe preserves numerical witnesses and independent calls."""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import causal_conv1d_parity as parity


def test_bfloat16_ulp_distance_handles_both_signs_and_signed_zero():
    left = torch.tensor([0x3F80, 0xBF80, 0], dtype=torch.int64).to(torch.int16)
    right = torch.tensor([0x3F81, 0xBF81, 0x8000], dtype=torch.int64).to(torch.int16)
    result = parity.comparison(left.view(torch.bfloat16), right.view(torch.bfloat16))
    assert result["mismatches"] == 3
    assert result["max_ulp_distance"] == 1
    assert result["exact"] is False
    assert result["max_abs"] == 0.0078125


def test_float32_adjacent_values_have_one_ulp_distance():
    value = torch.tensor([1.0, -1.0])
    other = torch.nextafter(value, torch.tensor([float("inf"), -float("inf")]))
    result = parity.comparison(value, other)
    assert result["mismatches"] == 2
    assert result["max_ulp_distance"] == 1


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_raw_witness_round_trips_values_and_hashes(dtype):
    value = torch.tensor([[1.0, -0.0], [-2.0, 0.0]], dtype=dtype).T
    record = parity.tensor_record(value, True)
    integer_dtype = parity.INTEGER_DTYPES[dtype]
    restored = torch.tensor(record["bits"], dtype=torch.int64).to(integer_dtype)
    restored = restored.view(dtype).reshape(record["shape"])
    assert parity.tensor_record(restored, True) == record
    assert "bits" not in parity.tensor_record(value, False)


def test_nonfinite_results_are_not_exact_or_json_nan():
    result = parity.comparison(torch.tensor([float("nan")]), torch.zeros(1))
    assert result["finite"] is False
    assert result["exact"] is False
    assert result["max_abs"] is None
    assert result["max_ulp_distance"] is None
    assert result["samples"][0]["causal_conv1d"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("mode", parity.MODES)
def test_seeded_inputs_preserve_layout_and_values(mode):
    shape = (2, 8, 16, 4)
    a, aw = parity.inputs(shape, mode, "contiguous", 17, "cpu")
    b, bw = parity.inputs(shape, mode, "transpose", 17, "cpu")
    assert a.stride() == (128, 16, 1)
    assert b.stride() == (128, 1, 8)
    assert torch.equal(a, b) and torch.equal(aw, bw)
    assert a.dtype == parity.MODES[mode][0]
    assert aw.dtype == parity.MODES[mode][1]


def test_probe_calls_both_implementations_even_when_outputs_differ(monkeypatch):
    calls = {"external": 0, "reference": 0}

    def external(value, weight, **kwargs):
        calls["external"] += 1
        return value + 1

    def reference(value, weight, **kwargs):
        calls["reference"] += 1
        return value.clone()

    monkeypatch.setattr(parity, "causal_conv1d_ref", reference)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    row = parity.run_case(
        external, (1, 8, 16, 4), "float32", "transpose", "silu", 17, 3, 256, "cpu"
    )
    assert calls == {"external": 3, "reference": 3}
    assert all(row["deterministic"].values())
    assert row["comparison"]["mismatches"] == 128
    assert row["comparison"]["exact"] is False
    assert "bits" in row["tensors"]["input"]
