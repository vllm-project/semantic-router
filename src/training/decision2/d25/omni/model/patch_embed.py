"""Run the Qwen3.5 vision patch embedding as a matrix product instead of a Conv3d.

The patch embedding is a Conv3d whose kernel equals its stride over inputs already cut into single
patches, so it is exactly a linear map of each flattened patch. On ROCm, MIOpen searches for a Conv3d
kernel for every new number of patches, and every image has a different count, which stalls
inference for minutes per image; the matrix product has no such search and gives the same result up
to floating-point summation order. The module keeps its weights, so checkpoints are unchanged.
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F


def _linear_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    weight = self.proj.weight
    flat = hidden_states.reshape(-1, weight[0].numel()).to(weight.dtype)
    out = F.linear(flat, weight.reshape(weight.shape[0], -1), self.proj.bias)
    return out.view(-1, self.embed_dim)


def linearize_patch_embed(model: torch.nn.Module) -> int:
    """Patch every vision patch embedding under ``model``; returns how many were patched."""
    count = 0
    for module in model.modules():
        proj = getattr(module, "proj", None)
        if (
            isinstance(proj, torch.nn.Conv3d)
            and tuple(proj.kernel_size) == tuple(proj.stride)
            and hasattr(module, "embed_dim")
        ):
            module.forward = types.MethodType(_linear_forward, module)
            count += 1
    return count
