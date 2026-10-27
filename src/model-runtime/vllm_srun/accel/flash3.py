"""FlashAttention-3 variable-length attention for state-once branches on Hopper.

``flash_attn_interface`` is optional, as FlashInfer is. The CUDA accelerator
registers this kernel on compute capability 9.x when the import succeeds. It
runs only when the operator sets the ``flash_attention_3`` engine argument.
Non-causal calls pass ``num_splits=1`` so a long state does not allocate
FlashAttention-3's split workspace (about 2.75 GiB at 60K tokens). A call it
cannot run (another dtype or head size) uses the FlashAttention-2 function
passed in.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from typing import Any

import torch

from ..plugins.base import DeviceInfo

SOURCE = "flash-attn-3"
HEAD_DIM = 256
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
    if len(text) < 2 or not text.isdigit():
        return None
    return int(text[:-1])


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
    if major != 9:
        shown = device.arch or "unknown"
        return (
            f"{device.label} is {shown}; FlashAttention-3 runs on Hopper "
            "(compute capability 9.x)"
        )
    if load() is None:
        return "flash_attn_interface is not installed"
    return None


def varlen(fallback: Varlen) -> Varlen:
    """A kernel with ``branch._flash``'s signature that calls FlashAttention-3, else ``fallback``.

    GQA is the head counts of ``q`` and ``k`` (``[T, H, D]``). Non-causal calls
    set ``num_splits=1`` when the installed function accepts it.
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
        if (
            function is None
            or not set(_VARLEN_ARGS) <= accepted
            or query.dtype not in DTYPES
            or query.shape[-1] != HEAD_DIM
            or key.shape[-1] != HEAD_DIM
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
