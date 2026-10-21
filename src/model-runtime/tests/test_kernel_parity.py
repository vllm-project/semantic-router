"""A bit-exact record must reject numeric equality with different bits."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from kernel_parity import compare_tensors


def test_signed_zero_is_not_bit_exact():
    report = compare_tensors(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert report["max_abs"] == 0.0
    assert report["mismatches"] == 1
    assert report["exact"] is False


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_outputs_cannot_pass(value):
    tensor = torch.tensor([value])
    report = compare_tensors(tensor, tensor.clone())
    assert report["exact"] is False
    assert report["max_abs"] is None


def test_noncontiguous_bfloat16_outputs_compare_by_value_bits():
    tensor = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).T
    report = compare_tensors(tensor, tensor.contiguous())
    assert report["n"] == 12
    assert report["mismatches"] == 0
    assert report["exact"] is True


@pytest.mark.parametrize("other", [torch.ones(2), torch.ones(1, dtype=torch.bfloat16)])
def test_layout_or_dtype_mismatch_fails(other):
    with pytest.raises(ValueError):
        compare_tensors(torch.ones(1), other)
