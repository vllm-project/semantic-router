"""Reduced copies of an encoder backbone's linear layers (the ``max_speed`` profile).

``reduced_view`` returns a view of a backbone that shares every parameter
(embeddings, norms, rotary tables) except its ``nn.Linear`` layers, which the
view holds in a reduced form; the backbone keeps its FP32 weights, so
``exact`` batches never see the copy. Kinds (``DtypePolicy.reduced_gpu`` /
``reduced_cpu``):

- ``bfloat16``: BF16 weights, run under BF16 autocast (norms, softmax and the
  heads stay FP32): GPUs, and CPUs with BF16 instructions;
- ``int8``: dynamic int8 (per-channel weight scales, per-batch activation
  scales; x86 CPUs);
- ``float32-packed``: FP32 weights reordered once for oneDNN (x86 CPUs,
  ``accel/onednn.py``), for models whose exact path keeps ``F.linear``.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, cast

import torch
from torch import nn

from ...accel import onednn
from ...accel.onednn import PackedLinear

if TYPE_CHECKING:
    from torch.ao.quantization import QConfig

REDUCED_AUTOCAST = {"bfloat16": torch.bfloat16}


class Quantizable(Protocol):
    """A float module that ``from_float`` quantizes with the ``qconfig`` set on it."""

    qconfig: QConfig


class QuantizedLinear(Protocol):
    """A dynamic int8 linear layer, whose weight and bias are methods over its packed parameters."""

    def weight(self) -> torch.Tensor: ...

    def bias(self) -> torch.Tensor | None: ...


def bf16_linear(linear: nn.Linear) -> nn.Module:
    layer = nn.Linear(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        layer.weight.copy_(linear.weight)
        if linear.bias is not None:
            layer.bias.copy_(linear.bias)
    return layer.eval()


def int8_linear(linear: nn.Linear) -> nn.Module:
    from torch.ao.nn.quantized.dynamic import Linear as DynamicLinear
    from torch.ao.quantization import per_channel_dynamic_qconfig

    source = copy.copy(linear)
    cast(Quantizable, source).qconfig = per_channel_dynamic_qconfig
    quantized: nn.Module = DynamicLinear.from_float(source)  # type: ignore[no-untyped-call]  # torch leaves from_float unannotated
    return quantized


COPIES: dict[str, Callable[[nn.Linear], nn.Module]] = {
    "bfloat16": bf16_linear,
    "int8": int8_linear,
    "float32-packed": PackedLinear,
}


def unavailable(
    kind: str, device: torch.device, native_bf16: bool = True
) -> str | None:
    """Why ``kind`` cannot run on ``device``; None when it can.

    ``native_bf16`` is ``DeviceInfo.bf16``: a CPU BF16 copy needs it.
    """
    if kind not in COPIES:
        return f"unknown reduced copy {kind!r} (one of {', '.join(COPIES)})"
    if kind in ("int8", "float32-packed") and device.type != "cpu":
        return f"{kind} copies run on CPUs"
    if kind == "bfloat16" and device.type == "cpu" and not native_bf16:
        return (
            "this CPU computes BF16 without native instructions (AVX-512 BF16 or AMX)"
        )
    if kind == "float32-packed" and not onednn.available():
        return "oneDNN's packed linear needs an x86 CPU and a PyTorch build with oneDNN"
    if kind == "int8" and "fbgemm" not in torch.backends.quantized.supported_engines:
        return "this PyTorch build has no fbgemm"
    return None


def reduced_view(module: nn.Module, kind: str) -> nn.Module:
    """``module`` with every ``nn.Linear`` replaced by its ``kind`` copy, every other parameter shared."""
    reason = unavailable(kind, next(module.parameters()).device)
    if reason:
        raise ValueError(reason)
    with torch.inference_mode(False), torch.no_grad():
        return _view(module, COPIES[kind])


def _view(module: nn.Module, make: Callable[[nn.Linear], nn.Module]) -> nn.Module:
    if isinstance(module, nn.Linear):
        return make(module)
    view = copy.copy(module)
    view._parameters = dict(module._parameters)
    view._buffers = dict(module._buffers)
    view._modules = type(module._modules)(
        (name, None if child is None else _view(child, make))
        for name, child in module._modules.items()
    )
    return view


def linear_bytes(module: nn.Module) -> int:
    """Bytes the view's reduced linear layers hold (shared parameters excluded)."""
    total = 0
    for layer in module.modules():
        if isinstance(layer, PackedLinear):
            tensors: list[torch.Tensor | None] = [layer.packed, layer.bias]
        elif isinstance(layer, nn.Linear):
            tensors = [layer.weight, layer.bias]
        elif callable(getattr(layer, "weight", None)):
            quantized = cast(QuantizedLinear, layer)
            tensors = [quantized.weight(), quantized.bias()]
        else:
            continue
        total += sum(t.numel() * t.element_size() for t in tensors if t is not None)
    return total
