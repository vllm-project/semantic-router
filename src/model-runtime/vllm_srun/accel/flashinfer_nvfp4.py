"""NVFP4 products in FlashInfer's CUTLASS FP4 GEMM (``engines/native/quantized.py``, ``NVFP4Linear``).

For the few hundred rows of a request, cuBLASLt's block-scaled FP4 product
(``torch.nn.functional.scaled_mm``) reads d3's NVFP4 weights below the GPU's
memory bandwidth. FlashInfer's CUTLASS kernel for compute capability 12.x,
its tile tuned per weight shape and power of two of rows (``tune``), reads
them faster up to ``ROWS`` rows, and writes the bytes cuBLAS writes for every
one of its tiles (``docs/records/decision3-quantized.md``). A call goes
through FlashInfer's Python wrapper, about three times the host time of
``scaled_mm``, so only graph captures take it, where that time is spent once;
eager forwards keep cuBLAS.

FlashInfer is optional (``flashinfer-python``). Its tuner times each tile
after a delay kernel that FlashInfer 0.7 compiles ahead of time for other
GPUs only, so on 12.x it compiles it at first use, with a CUDA toolkit of
12.9 or later; without one, every product would take FlashInfer's default
tile, which is no faster than cuBLAS's and fails in a graph after a few
captures, so ``load`` refuses. FlashInfer takes the architecture its kernels
target from PyTorch's CUDA version and refuses 12.x before CUDA 12.9, although
its prebuilt kernels (``flashinfer-jit-cache``) and a newer toolkit run there;
``FLASHINFER_CUDA_ARCH_LIST`` then names the family target for it.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Iterable

import torch

# Rows up to which graph captures take FlashInfer's product. FlashInfer times its tiles at
# each power of two up to here, and a call takes the tile of the power its rows round to.
ROWS = 512
ARCH_LIST = "FLASHINFER_CUDA_ARCH_LIST"

Product = Callable[..., torch.Tensor]


def load(device: torch.device) -> tuple[Product | None, str | None]:
    """FlashInfer's FP4 GEMM for ``device``, or None and why it cannot run there."""
    major, minor = torch.cuda.get_device_capability(device)
    if major != 12:
        return None, (
            f"FlashInfer's FP4 GEMM is measured on compute capability 12.x, not {major}.{minor}"
        )
    cuda = tuple(int(part) for part in (torch.version.cuda or "0.0").split(".")[:2])
    if cuda < (12, 9) and ARCH_LIST not in os.environ:
        # read when FlashInfer is imported; only the prebuilt kernels run then
        os.environ[ARCH_LIST] = f"{major}.{minor}f"
    try:
        mm_fp4 = importlib.import_module("flashinfer").mm_fp4
    except Exception as e:
        return None, f"FlashInfer is not installed ({type(e).__name__}: {e})"
    try:
        # the tuner's delay kernel (compiled here at first use)
        importlib.import_module("flashinfer.tllm_utils").get_trtllm_utils_module()
    except Exception as e:
        return None, (
            "FlashInfer cannot tune its FP4 GEMM here, and its default tile is no "
            f"faster than cuBLAS's ({type(e).__name__}: {e})"
        )
    return mm_fp4, None


def tune(
    mm_fp4: Product, shapes: Iterable[tuple[int, int]], device: torch.device
) -> None:
    """Time FlashInfer's tiles for each [out, in] weight shape up to ``ROWS`` rows; later calls take the fastest."""
    autotune = importlib.import_module("flashinfer").autotune
    for out_features, in_features in sorted(set(shapes)):
        q = torch.randint(
            0, 256, (ROWS, in_features // 2), dtype=torch.uint8, device=device
        )
        weight = torch.randint(
            0, 256, (out_features, in_features // 2), dtype=torch.uint8, device=device
        )
        # the default buckets: calls outside this context look their tiles up by them
        with autotune(True):
            product(
                mm_fp4,
                q,
                _scales(ROWS, in_features, device),
                weight.t(),
                _scales(out_features, in_features, device)
                .view(_padded(out_features), -1)
                .t(),
                torch.ones(1, device=device),
            )


def product(
    mm_fp4: Product,
    q: torch.Tensor,
    scale: torch.Tensor,
    weight_t: torch.Tensor,
    weight_scale_t: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """``q`` [rows, K / 2] times ``weight_t`` [K / 2, N] in BF16, their block scales in cuBLAS's 32x4x4 layout.

    ``scale`` is flat (``triton_nvfp4.quantize``) and ``weight_scale_t`` the
    transposed [N padded, K / 16 padded] matrix of the weight's; ``alpha`` is
    the product of the two global decode scales.
    """
    return mm_fp4(
        q,
        weight_t,
        scale.view(_padded(q.shape[0]), -1),
        weight_scale_t,
        alpha,
        torch.bfloat16,
        backend="cutlass",
    )


def _padded(rows: int) -> int:
    return -(-rows // 128) * 128


def _scales(rows: int, in_features: int, device: torch.device) -> torch.Tensor:
    """Random flat block scales of ``rows`` rows in cuBLAS's padded layout, for timing."""
    groups = -(-(in_features // 16) // 4) * 4
    return (torch.rand(_padded(rows) * groups, device=device) + 0.5).to(
        torch.float8_e4m3fn
    )
