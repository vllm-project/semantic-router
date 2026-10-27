"""The fla-npu Ascend C chunk kernel as an NPU kernel-slot candidate.

Approximate slot: the Ascend C chunk rounds 1-2 bf16 ulp away from the
reference per call -- both implementations sit at max 1 ulp from the FP64
reference, so this is two equally-correct rounding paths, not a wrong kernel.
Measured on Ascend 950DT
(``docs/records/npu-fla-chunk-ascend950dt.md``): the 60-answer label-diff
flips 4 of 60 requests (row agreement 93.3%), every flip on a borderline leaf
(two-option probability gap <= 0.02), the two flip directions opposite; the
divergence does not compound across the 24 GDN layers (measured growth 0.96x)
and does not grow with sequence length. The slot therefore registers
approximate and selects only under ``DtypePolicy.approximate_kernels`` -- the
released BF16 decision packages keep exact kernels and are unchanged.

fla-npu (https://github.com/flashserve/flash-linear-attention-npu) ships the
Ascend C GDN ops vllm-ascend binds for Qwen3.5. Its README documents the
wrapper (``examples/flash_gated_delta_rule.py``) as the integration entry; this
module loads it from the ``FLA_NPU_EXAMPLES`` directory (the fla-npu checkout's
``examples/``) or from the installed ``fla_npu`` package, whichever provides
``flash_gated_delta_rule``. Other SoCs and installs without it run the
reference kernels.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Any

import torch

from .kernels import chunk_gated_delta_rule_ref

# SoCs the slot was measured on: registered (approximate) only there -- the
# rocm.py precedent of exact/approximate only where measured.
MEASURED = frozenset({"Ascend950DT"})

_CACHE: dict[str, Callable[..., Any] | None] = {}


def _load_wrapper() -> Callable[..., Any] | None:
    """``flash_gated_delta_rule`` from the installed package or the checkout's examples."""
    try:
        from fla_npu.flash_gated_delta_rule import (  # type: ignore[attr-defined]
            flash_gated_delta_rule,
        )

        return flash_gated_delta_rule
    except ImportError:
        pass
    path = os.environ.get("FLA_NPU_EXAMPLES")
    if path:
        sys.path.insert(0, path)
        try:
            from flash_gated_delta_rule import flash_gated_delta_rule

            return flash_gated_delta_rule
        except ImportError:
            pass
    return None


def available_wrapper() -> Callable[..., Any] | None:
    """The fla-npu chunk wrapper this install provides, loaded once, else ``None``."""
    if "wrapper" not in _CACHE:
        try:
            _CACHE["wrapper"] = _load_wrapper()
        except Exception:
            _CACHE["wrapper"] = None
    return _CACHE["wrapper"]


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The fla-npu Ascend C chunk, the slot contract's ``[B, T, H, D]`` inputs.

    The wrapper reads the head-major (NTD) layout for the projections and
    returns the output already in ``[B, T, H, D]`` (measured against the
    reference at the served shape on Ascend 950DT: max |diff| 3.9e-3 — the
    BF16 GDN rounding floor — on bf16 inputs, with the recurrent state
    ``[B, H, K, V]`` as the reference's). The kernel is bf16-only — other
    dtypes fall back to the reference so the slot never breaks an fp32/fp16
    backbone.
    """
    wrapper = available_wrapper()
    if wrapper is None or query.dtype != torch.bfloat16:
        return chunk_gated_delta_rule_ref(
            query,
            key,
            value,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
    out = wrapper(
        query.transpose(1, 2).contiguous(),
        key.transpose(1, 2).contiguous(),
        value.transpose(1, 2).contiguous(),
        g=g,
        beta=beta,
        scale=query.shape[-1] ** -0.5,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    output = out[0].to(query.dtype)
    state = out[1] if output_final_state else None
    return output, state
