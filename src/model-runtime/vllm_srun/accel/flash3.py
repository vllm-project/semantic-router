"""FlashAttention-3 variable-length attention for state-once branches on Hopper.

``flash_attn_interface`` is optional, as FlashInfer is. The CUDA accelerator
registers this kernel on compute capability 9.x when the import succeeds. It
runs only when the operator sets the ``flash_attention_3`` engine argument.
Non-causal calls pass ``num_splits=1`` so a long state does not allocate
FlashAttention-3's split workspace (about 2.75 GiB at 60K tokens). Hopper
kernels accept head sizes 64-256. A call outside that, another dtype, or a
function that cannot take the scale, uses ``fallback``. The kernel registered
for the engine argument passes ``refuse``, so the card's ``flash-attn-3``
source is not a silent FlashAttention-2 call. A model whose attention head
size is outside 64-256 refuses to load.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from typing import Any

import torch

from ..plugins.base import DeviceInfo

SOURCE = "flash-attn-3"
# Hopper FlashAttention-3 accepts this range. ``HEAD_DIM`` is the width the
# numeric test and the released hybrid Decision 3.0 packages use.
HEAD_DIM_MIN = 64
HEAD_DIM_MAX = 256
HEAD_DIM = HEAD_DIM_MAX
# ``sm_{major}{minor}`` keeps the minor version in the last digit (``sm_90``, ``sm_100``).
_ARCH_MINOR_DIGITS = 1
HOPPER_MAJOR = 9
DTYPES = (torch.bfloat16, torch.float16)
# Positional arguments of ``flash_attn_varlen_func``, in order.
_VARLEN_ARGS = (
    "q",
    "k",
    "v",
    "cu_seqlens_q",
    "cu_seqlens_k",
    "max_seqlen_q",
    "max_seqlen_k",
)

Varlen = Callable[..., torch.Tensor]


def capability_major(arch: str | None) -> int | None:
    """The CUDA major version in an ``sm_90`` or ``sm_100`` architecture name.

    ``CUDAAccelerator.arch`` encodes compute capability as ``sm_{major}{minor}``
    with a one-digit minor (``sm_90`` is 9.0, ``sm_100`` is 10.0).
    """
    text = (arch or "").removeprefix("sm_")
    if len(text) <= _ARCH_MINOR_DIGITS or not text.isdigit():
        return None
    return int(text[:-_ARCH_MINOR_DIGITS])


def load() -> Varlen | None:
    """FlashAttention-3's variable-length function, or None when it is not installed."""
    try:
        function = getattr(
            importlib.import_module("flash_attn_interface"),
            "flash_attn_varlen_func",
            None,
        )
    except Exception:
        return None
    return function if callable(function) else None


def unavailable(device: DeviceInfo) -> str | None:
    """Why ``device`` cannot run FlashAttention-3, or None when it can."""
    if device.accelerator != "cuda":
        return f"{device.label} is not a CUDA device"
    major = capability_major(device.arch)
    if major != HOPPER_MAJOR:
        shown = device.arch or "unknown"
        return (
            f"{device.label} is {shown}; FlashAttention-3 runs on Hopper "
            "(compute capability 9.x)"
        )
    if load() is None:
        return "flash_attn_interface is not installed"
    return None


def head_sizes(module: Any) -> set[int]:
    """Attention ``head_dim`` values on ``module`` and its children."""
    walk = getattr(module, "modules", None)
    children = walk() if callable(walk) else (module,)
    found: set[int] = set()
    for child in children:
        size = getattr(child, "head_dim", None)
        if isinstance(size, int):
            found.add(size)
    return found


def unsupported_heads(module: Any) -> str | None:
    """Why ``module``'s attention head sizes cannot run FlashAttention-3, or None."""
    outside = sorted(
        size
        for size in head_sizes(module)
        if size < HEAD_DIM_MIN or size > HEAD_DIM_MAX
    )
    if not outside:
        return None
    shown = ", ".join(str(size) for size in outside)
    return (
        f"attention head size {shown} is outside {HEAD_DIM_MIN}-{HEAD_DIM_MAX}; "
        "FlashAttention-3 does not run it and does not fall back to FlashAttention-2"
    )


def opt_in_refusal(device: DeviceInfo, module: Any) -> str | None:
    """Why ``flash_attention_3`` cannot load for ``module`` on ``device``, or None."""
    reason = unavailable(device)
    if reason is not None:
        return reason
    return unsupported_heads(module)


def refuse(*_args: object, **_kwargs: object) -> torch.Tensor:
    """Fail a call FlashAttention-3 cannot run.

    The registered kernel uses this as ``varlen``'s fallback. FlashAttention-2
    here would leave the card saying ``flash-attn-3`` for a call that never
    reached it.
    """
    raise RuntimeError(
        "flash_attention_3: this call cannot run on FlashAttention-3 "
        f"(needs bf16 or fp16, head size {HEAD_DIM_MIN}-{HEAD_DIM_MAX}, "
        "and a scale argument) and does not fall back to FlashAttention-2"
    )


def varlen(fallback: Varlen) -> Varlen:
    """A kernel with ``branch._flash``'s signature that calls FlashAttention-3, else ``fallback``.

    Head sizes 64-256 in BF16 or FP16 call FlashAttention-3. GQA is the
    head counts of ``q`` and ``k`` (``[T, H, D]``). Non-causal calls set
    ``num_splits=1`` when the installed function accepts it. Anything else calls
    ``fallback`` (``refuse`` on the registered kernel).
    """
    function = load()
    accepted = set(inspect.signature(function).parameters) if function else set()

    def attend(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        max_q: int,
        max_k: int,
        causal: bool,
        scale: float,
    ) -> torch.Tensor:
        dim = query.shape[-1]
        if (
            function is None
            or not set(_VARLEN_ARGS) <= accepted
            or query.dtype not in DTYPES
            or dim < HEAD_DIM_MIN
            or dim > HEAD_DIM_MAX
            or key.shape[-1] != dim
        ):
            return fallback(query, key, value, cu_q, cu_k, max_q, max_k, causal, scale)
        call: dict[str, Any] = dict(
            zip(
                _VARLEN_ARGS,
                (query, key, value, cu_q, cu_k, max_q, max_k),
                strict=True,
            )
        )
        if "softmax_scale" in accepted:
            call["softmax_scale"] = scale
        elif "scale" in accepted:
            call["scale"] = scale
        elif scale != query.shape[-1] ** -0.5:
            return fallback(query, key, value, cu_q, cu_k, max_q, max_k, causal, scale)
        if "causal" in accepted:
            call["causal"] = causal
        if "num_splits" in accepted and not causal:
            call["num_splits"] = 1
        out = function(**call)
        if isinstance(out, tuple):
            out = out[0]
        return out

    return attend
