"""Opt-in, fixed-order task-gradient projection for the 0.6B control arm.

Only the shared backbone is projected. The candidate head always receives
the ordinary sum. The disabled path is available for exact one-step parity
tests; the default trainer never constructs this accumulator.
"""

from __future__ import annotations

import math
from typing import Any

from .decision_model import TASK_TYPES

PROJECTION_VERSION = "decision2-06b-backbone-original-reference-norm-matched-v1"
NORM_EPSILON = 1e-12


def _dot(left: list[Any], right: list[Any]) -> float:
    import torch

    if not left or len(left) != len(right):
        raise ValueError("Task gradient vectors are misaligned")
    total = torch.zeros((), dtype=torch.float64, device=left[0].device)
    for a, b in zip(left, right):
        total.add_(torch.dot(a.reshape(-1), b.reshape(-1)).double())
    value = float(total.item())
    if not math.isfinite(value):
        raise ValueError("Nonfinite task-gradient dot product")
    return value


def _norm(values: list[Any]) -> float:
    return math.sqrt(max(0.0, _dot(values, values)))


def project_backbone_gradients(
    typed: dict[str, list[Any]], counts: dict[str, int], *, enabled: bool
) -> tuple[list[Any], dict[str, Any]]:
    """Project each task against original other-task vectors in frozen order."""
    import torch

    if set(typed) != set(TASK_TYPES) or set(counts) != set(TASK_TYPES):
        raise ValueError("Projection needs Choice, Noul and Score buffers")
    width = len(typed[TASK_TYPES[0]])
    if width == 0 or any(len(typed[kind]) != width for kind in TASK_TYPES):
        raise ValueError("Projection buffers are misaligned")
    if any(type(count) is not int or count < 0 for count in counts.values()):
        raise ValueError("Invalid per-task window counts")
    if not any(counts.values()):
        raise ValueError("Empty optimizer window")
    original = [torch.zeros_like(value) for value in typed[TASK_TYPES[0]]]
    for kind in TASK_TYPES:
        torch._foreach_add_(original, typed[kind])
    original_norm = _norm(original)
    norms = {kind: _norm(typed[kind]) if counts[kind] else 0.0 for kind in TASK_TYPES}
    cosines: dict[str, float | None] = {}
    for left, right in (("choice", "noul"), ("choice", "score"), ("noul", "score")):
        denominator = norms[left] * norms[right]
        cosines[f"{left}_{right}"] = (
            max(-1.0, min(1.0, _dot(typed[left], typed[right]) / denominator))
            if denominator > NORM_EPSILON
            else None
        )
    projected_pairs = 0
    result = original
    pre_match_norm = original_norm
    norm_scale = 1.0
    if enabled:
        combined = [torch.zeros_like(value) for value in original]
        squared = {kind: norms[kind] ** 2 for kind in TASK_TYPES}
        for task in TASK_TYPES:
            if not counts[task]:
                continue
            vector = [value.clone() for value in typed[task]]
            for reference in TASK_TYPES:
                if (
                    reference == task
                    or not counts[reference]
                    or norms[reference] < NORM_EPSILON
                ):
                    continue
                coefficient = min(0.0, _dot(vector, typed[reference])) / (
                    squared[reference] + NORM_EPSILON
                )
                if coefficient < 0.0:
                    torch._foreach_add_(vector, typed[reference], alpha=-coefficient)
                    projected_pairs += 1
            torch._foreach_add_(combined, vector)
        pre_match_norm = _norm(combined)
        if original_norm >= NORM_EPSILON and pre_match_norm >= NORM_EPSILON:
            norm_scale = original_norm / pre_match_norm
            if not math.isfinite(norm_scale):
                raise ValueError("Nonfinite gradient norm matching factor")
            torch._foreach_mul_(combined, norm_scale)
            result = combined
        # A zero norm takes the frozen ordinary-sum fallback.
    if any(not bool(torch.isfinite(value).all().item()) for value in result):
        raise ValueError("Nonfinite projected backbone gradient")
    summary = {
        "projection_version": PROJECTION_VERSION,
        "enabled": enabled,
        "task_counts": dict(counts),
        "task_norms": norms,
        "pairwise_cosines": cosines,
        "projected_pairs": projected_pairs,
        "ordinary_backbone_norm": original_norm,
        "pre_match_backbone_norm": pre_match_norm,
        "norm_scale": norm_scale,
        "final_backbone_norm": _norm(result),
    }
    return result, summary


class TaskGradientAccumulator:
    """Capture each one-example backward before the next microbatch."""

    def __init__(self, model: Any):
        import torch

        self.backbone = list(model.backbone.parameters())
        self.head = list(model.head.parameters())
        if (
            not self.backbone
            or not self.head
            or any(param.dtype != torch.float32 for param in self.backbone + self.head)
        ):
            raise ValueError("Projection needs a full FP32 backbone and shared head")
        self.typed = {
            kind: [torch.zeros_like(param) for param in self.backbone]
            for kind in TASK_TYPES
        }
        self.head_sum = [torch.zeros_like(param) for param in self.head]
        self.reset()

    def reset(self) -> None:
        import torch

        for kind in TASK_TYPES:
            torch._foreach_zero_(self.typed[kind])
        torch._foreach_zero_(self.head_sum)
        self.counts = dict.fromkeys(TASK_TYPES, 0)
        self.backbone_used = [False] * len(self.backbone)
        self.head_used = [False] * len(self.head)

    def capture(self, kind: str) -> None:
        import torch

        if kind not in TASK_TYPES:
            raise ValueError("Unknown native decision task type")
        backbone_buffers = []
        backbone_gradients = []
        for index, param in enumerate(self.backbone):
            if param.grad is not None:
                backbone_buffers.append(self.typed[kind][index])
                backbone_gradients.append(param.grad)
                self.backbone_used[index] = True
        head_buffers = []
        head_gradients = []
        for index, param in enumerate(self.head):
            if param.grad is not None:
                head_buffers.append(self.head_sum[index])
                head_gradients.append(param.grad)
                self.head_used[index] = True
        gradients = backbone_gradients + head_gradients
        if not gradients or not bool(
            torch.isfinite(torch.stack(torch._foreach_norm(gradients))).all().item()
        ):
            raise ValueError("Nonfinite gradient before projection")
        if backbone_gradients:
            torch._foreach_add_(backbone_buffers, backbone_gradients)
        if head_gradients:
            torch._foreach_add_(head_buffers, head_gradients)
        self.counts[kind] += 1

    def finalize(self, *, enabled: bool) -> dict[str, Any]:
        gradients, summary = project_backbone_gradients(
            self.typed, self.counts, enabled=enabled
        )
        for index, param in enumerate(self.backbone):
            param.grad = gradients[index] if self.backbone_used[index] else None
        for index, param in enumerate(self.head):
            param.grad = self.head_sum[index] if self.head_used[index] else None
        return summary
