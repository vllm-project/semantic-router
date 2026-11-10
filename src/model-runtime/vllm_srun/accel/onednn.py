"""oneDNN's pre-packed FP32 linear: the ``linear`` slot's ``float32-packed`` variant on x86 CPUs.

The weight is reordered once, at load. On EPYC its GEMMs are 2.5-4.4x faster
than ``F.linear`` (MKL) at every batch size and stay within 3.3e-6 of it.

Whether a row's result is the same alone or inside a batch depends on the
oneDNN build, the product shape and the thread count, so ``invariant`` measures
it on the model's own layers at load instead of assuming it. On the EPYC the
kernel was first measured on (oneDNN 3.7, the pinned torch 2.10) every batch
size agreed; on a Xeon Gold 6530 and a Zen 4 Ryzen with the same build the
wide-K down-projections did not once a batch held 81 rows or more (1152-to-768
at 12 and 64 threads, 3072-to-768 at 4 to 16 threads, every count under AVX2),
while MKL's ``F.linear`` differs below 32 rows; with oneDNN 3.10 (torch 2.11)
on that Xeon every shape differs between 2-row and 17-row batches at any
thread count, and a Raptor Lake host on torch 2.14 differs only between a
single row and any batch. The same ops called directly on a GB10's Grace
cores (aarch64 wheels of torch 2.10 and 2.11; ``available`` does not admit
that machine yet, #5437 widens it) agreed at every count from 1 to 20. The
property is one of the build, the host and the thread count, which is why it
is probed rather than assumed.
"""

from __future__ import annotations

import platform
from collections.abc import Iterable

import torch
from torch import nn

PACKED = "float32-packed"

# Row counts a packed product is probed at, each against 2-row products: the
# batch sizes the families' own probes use, 17 where oneDNN 3.10 first splits
# differently, 96 inside the band from 81 rows where oneDNN 3.7 changes its
# split on wide products (the band's end moves with the thread count), 127 to
# 129, and larger batches; nothing new appeared between 512 and 8,192 rows.
PROBE_ROWS = (3, 8, 17, 64, 96, 127, 129, 208, 512)


def available() -> bool:
    """Whether this host and PyTorch build run the packed linear."""
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return False
    if not torch.backends.mkldnn.is_available():
        return False
    try:
        torch.ops.mkldnn._linear_pointwise  # noqa: B018 - resolves the op or raises
    except (AttributeError, RuntimeError):
        return False
    return True


class PackedLinear(nn.Module):
    """An FP32 ``nn.Linear`` whose weight oneDNN reordered once (x86 CPUs)."""

    packed: torch.Tensor

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.weight_elements = linear.weight.numel()
        # Reordered for single rows: the layout every batch size then shares.
        packed = torch.ops.mkldnn._reorder_linear_weight(linear.weight.detach(), 1)
        self.register_buffer("packed", packed, persistent=False)
        self.bias = (
            None
            if linear.bias is None
            else nn.Parameter(linear.bias.detach(), requires_grad=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.ops.mkldnn._linear_pointwise(
            x.reshape(-1, self.in_features), self.packed, self.bias, "none", [], ""
        )
        return out.reshape(*x.shape[:-1], self.out_features)


def probe_rows(in_features: int, generator: torch.Generator) -> torch.Tensor:
    """The seeded rows a product shape is probed with: ``max(PROBE_ROWS)`` of them."""
    return torch.randn(max(PROBE_ROWS), in_features, generator=generator)


def agrees(layer: PackedLinear, rows: torch.Tensor) -> bool:
    """Whether each probe batch of ``rows`` gives every row its 2-row-product result.

    A packed row never runs alone: an encoder sequence carries at least its
    two special tokens, so the smallest product a backbone row sees has two
    rows, and that is the reference. Single-row products (a pooled head over
    one request) are the families' whole-forward probes' concern.
    """
    with torch.inference_mode():
        pairs = torch.cat([layer(rows[i : i + 2]) for i in range(0, rows.shape[0], 2)])
        return all(torch.equal(layer(rows[:n]), pairs[:n]) for n in PROBE_ROWS)


def invariant(layers: Iterable[PackedLinear]) -> bool:
    """Whether each packed product shape answers a row the same alone or in a batch.

    Each ``(in_features, out_features)`` is probed once (``agrees``) at the
    intra-op thread count in force when called; the engine calls it at the
    count its forwards use. Adaptive threads at serving level 2 may later run
    a forward at a learned count, a risk that level accepts for every kernel
    (``engines/native/threads.py``). A model whose shapes all pass may still
    fail its family's whole-forward probe; one that fails here cannot be
    batch-invariant.
    """
    generator = torch.Generator().manual_seed(0)
    seen: set[tuple[int, int]] = set()
    for layer in layers:
        shape = (layer.in_features, layer.out_features)
        if shape in seen:
            continue
        seen.add(shape)
        if not agrees(layer, probe_rows(layer.in_features, generator)):
            return False
    return True
