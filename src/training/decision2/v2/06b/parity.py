"""Padded versus one-row micro-batch parity for the 0.6B trainer families.

The trainer sums per-row losses over type-homogeneous micro-batches padded to
one length. For identical rows and weights the padded micro-batch must give
the same summed loss and parameter gradients as the rows taken one at a time.
In FP32 any difference beyond rounding is an implementation defect (readout
indices versus padding side, attention or position handling, pooling or loss
masks). Under BF16 autocast both paths carry rounding error of their own, so
each is measured against the FP32 one-row reference rather than against the
other.
"""

from __future__ import annotations

import contextlib
import math
from typing import Any

FP32_LOSS_TOLERANCE = 1e-5
FP32_GRAD_TOLERANCE = 1e-3
BF16_LOSS_TOLERANCE = 1e-2
BF16_MIN_COSINE = 0.95


def module_of(family: Any) -> Any:
    return family.native.model if hasattr(family, "native") else family.model


@contextlib.contextmanager
def without_dropout(model: Any) -> Any:
    """Zero every dropout rate (training flags untouched); padding parity is deterministic."""
    import torch

    saved = []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            saved.append((module, "p", module.p))
        elif isinstance(module, torch.nn.MultiheadAttention):
            saved.append((module, "dropout", module.dropout))
        if isinstance(getattr(module, "attention_dropout", None), float):
            saved.append((module, "attention_dropout", module.attention_dropout))
    for module, name, _ in saved:
        setattr(module, name, 0.0)
    try:
        yield sum(value > 0 for _, _, value in saved)
    finally:
        for module, name, value in saved:
            setattr(module, name, value)


@contextlib.contextmanager
def fp32_compute() -> Any:
    """Disable the families' BF16 autocast; their heads already compute in FP32."""
    import torch

    original = torch.autocast

    def disabled(*args: Any, **kwargs: Any) -> Any:
        kwargs["enabled"] = False
        return original(*args, **kwargs)

    torch.autocast = disabled
    try:
        yield
    finally:
        torch.autocast = original


def gradients(
    family: Any,
    records: list[dict[str, Any]],
    teacher: list[Any],
    device: str,
    *,
    padded: bool,
) -> tuple[float, dict[str, Any]]:
    """Summed loss and every parameter gradient, as one padded batch or row by row."""
    model = module_of(family)
    model.zero_grad(set_to_none=True)
    groups = (
        [(records, teacher)]
        if padded
        else [([r], [t]) for r, t in zip(records, teacher)]
    )
    total = 0.0
    for rows, targets in groups:
        loss, _ = family.loss(rows, targets, device)
        loss.backward()
        total += float(loss.detach())
    grads = {
        name: p.grad.detach().float().clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    model.zero_grad(set_to_none=True)
    return total, grads


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    """Gradient set `a` against reference `b`: relative L2 error, cosine, norm ratio."""
    import torch

    diff = ref = dot = own = 0.0
    for name in set(a) | set(b):
        x, y = a.get(name), b.get(name)
        x = torch.zeros_like(y) if x is None else x
        y = torch.zeros_like(x) if y is None else y
        x, y = x.double(), y.double()
        diff += float((x - y).square().sum())
        ref += float(y.square().sum())
        dot += float((x * y).sum())
        own += float(x.square().sum())
    return {
        "rel_err": math.sqrt(diff / ref) if ref else (0.0 if diff == 0 else math.inf),
        "cosine": dot / math.sqrt(own * ref) if own and ref else float(own == ref),
        "norm_ratio": math.sqrt(own / ref) if ref else (1.0 if own == 0 else math.inf),
    }


def relative(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-12)


def micro_batch_parity(
    family: Any,
    records: list[dict[str, Any]],
    teacher: list[Any],
    device: str,
    *,
    bf16: bool = True,
) -> dict[str, Any]:
    """FP32 padded vs one-row (exact), then BF16 paths against the FP32 one-row reference."""
    if len(records) < 2:
        raise ValueError("Parity needs a micro-batch of at least two rows")
    with without_dropout(module_of(family)) as zeroed:
        report = _parity(family, records, teacher, device, bf16=bf16)
    report["dropout_modules_zeroed"] = zeroed
    return report


def _parity(
    family: Any,
    records: list[dict[str, Any]],
    teacher: list[Any],
    device: str,
    *,
    bf16: bool,
) -> dict[str, Any]:
    with fp32_compute():
        loss_pad, g_pad = gradients(family, records, teacher, device, padded=True)
        loss_one, reference = gradients(family, records, teacher, device, padded=False)
    report: dict[str, Any] = {
        "rows": len(records),
        "fp32": {
            "loss_padded": loss_pad,
            "loss_rows": loss_one,
            "loss_rel_diff": relative(loss_pad, loss_one),
            "grad_padded_vs_rows": compare(g_pad, reference),
        },
    }
    del g_pad
    passed = (
        report["fp32"]["loss_rel_diff"] <= FP32_LOSS_TOLERANCE
        and report["fp32"]["grad_padded_vs_rows"]["rel_err"] <= FP32_GRAD_TOLERANCE
    )
    if bf16:
        loss_pad16, g_pad16 = gradients(family, records, teacher, device, padded=True)
        loss_one16, g_one16 = gradients(family, records, teacher, device, padded=False)
        report["bf16"] = {
            "loss_padded": loss_pad16,
            "loss_rows": loss_one16,
            "loss_rel_diff": relative(loss_pad16, loss_one16),
            "grad_padded_vs_fp32": compare(g_pad16, reference),
            "grad_rows_vs_fp32": compare(g_one16, reference),
            "grad_padded_vs_rows": compare(g_pad16, g_one16),
        }
        passed = (
            passed
            and report["bf16"]["loss_rel_diff"] <= BF16_LOSS_TOLERANCE
            and report["bf16"]["grad_padded_vs_fp32"]["cosine"] >= BF16_MIN_COSINE
        )
    report["tolerances"] = {
        "fp32_loss_rel": FP32_LOSS_TOLERANCE,
        "fp32_grad_rel_err": FP32_GRAD_TOLERANCE,
        "bf16_loss_rel": BF16_LOSS_TOLERANCE,
        "bf16_padded_vs_fp32_min_cosine": BF16_MIN_COSINE,
    }
    report["passed"] = passed
    return report
