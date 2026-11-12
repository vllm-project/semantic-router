"""Compare floating-point kernel outputs without hiding dtype or bit differences."""

from __future__ import annotations

from typing import Any

import torch


def compare_tensors(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    """Count differing elements by their bytes, including signed zero."""
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("kernel outputs must have the same shape and dtype")
    a = left.detach().cpu().contiguous()
    b = right.detach().cpu().contiguous()
    width = a.element_size()
    a_bits = a.reshape(-1).view(torch.uint8).reshape(-1, width)
    b_bits = b.reshape(-1).view(torch.uint8).reshape(-1, width)
    mismatches = int((a_bits != b_bits).any(dim=-1).sum().item())
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    max_abs = float((a.double() - b.double()).abs().max()) if a.numel() else 0.0
    return {
        "n": a.numel(),
        "mismatches": mismatches,
        "max_abs": max_abs if finite else None,
        "exact": finite and mismatches == 0,
    }
