"""Gated LoRA: several adapters on one backbone's Linears, at most one on per forward, on some rows only.

A family switches an adapter on for some rows of a batch
(``ForwardBatch.adapter``) and leaves the base computation untouched
everywhere else. ``GatedLinear`` computes, in eval mode:

    y = base(x) + ((x.float() @ A.T) @ B.T * alpha / r).to(y.dtype)

on the gated rows, and ``torch.where`` keeps ``base(x)`` bit for bit on the
others; with no adapter on (or one this layer does not hold) it returns
``base(x)`` itself. Factors stay FP32 whatever the backbone holds, and are
plain tensors, not parameters: the backbone's parameter count, dtype casts
and checkpoint layout are the base's.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator, Mapping
from pathlib import Path

import torch
from torch import nn

from ....plugins.base import AdapterGate, GatedAdapterSpec


class Gate(threading.local):
    """The adapter the current thread's forward runs, and the rows it covers."""

    name: str | None = None
    rows: tuple[bool, ...] | None = None
    _mask: torch.Tensor | None = None

    @contextlib.contextmanager
    def on(self, adapter: AdapterGate | None) -> Iterator[None]:
        saved = (self.name, self.rows, self._mask)
        self.name = adapter.name if adapter is not None else None
        self.rows = (
            adapter.rows
            if adapter is not None
            and adapter.rows is not None
            and not all(adapter.rows)
            else None
        )
        self._mask = None
        try:
            yield
        finally:
            self.name, self.rows, self._mask = saved

    def mask(self, batch: int, device: torch.device) -> torch.Tensor | None:
        """The row mask as a device tensor (built once per forward), or None for every row."""
        if self.rows is None:
            return None
        if len(self.rows) != batch:
            raise ValueError(
                f"the adapter gate covers {len(self.rows)} rows; the batch has {batch}"
            )
        if self._mask is None or self._mask.device != device:
            self._mask = torch.tensor(self.rows, dtype=torch.bool, device=device)
        return self._mask


class GatedLinear(nn.Module):
    """A Linear with gated LoRA adapters (``GatedAdapterSpec``)."""

    def __init__(self, base: nn.Module, gate: Gate):
        super().__init__()
        self.base = base
        self.gate = gate
        # name -> (A [r, in], B [out, r], scaling); FP32 plain tensors.
        self.factors: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}

    def add(self, name: str, a: torch.Tensor, b: torch.Tensor, scaling: float) -> None:
        self.factors[name] = (a.float(), b.float(), float(scaling))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y: torch.Tensor = self.base(x)
        name = self.gate.name
        if name is None or name not in self.factors:
            return y
        a, b, scaling = self.factors[name]
        delta = ((x.float() @ a.T) @ b.T * scaling).to(y.dtype)
        rows = self.gate.mask(y.shape[0], y.device)
        if rows is None:
            return y + delta
        return torch.where(rows.view(-1, *([1] * (y.dim() - 1))), y + delta, y)


def attach_gated(
    backbone: nn.Module,
    adapters: Mapping[str, GatedAdapterSpec],
    gate: Gate,
    device: torch.device,
) -> dict[str, int]:
    """Wrap every Linear an adapter names and load its factors onto ``device``; returns modules per adapter."""
    from safetensors.torch import load_file

    counts = {}
    for name, spec in adapters.items():
        tensors = load_file(str(Path(spec.weight_file)))
        modules = sorted({key.rsplit(".", 1)[0] for key in tensors})
        if len(tensors) != 2 * len(modules):
            raise ValueError(f"adapter {name}: unpaired A / B tensors")
        for module in modules:
            parent_path, _, attr = module.rpartition(".")
            parent = backbone.get_submodule(parent_path)
            layer = getattr(parent, attr)
            if not isinstance(layer, GatedLinear):
                if not isinstance(layer, nn.Linear):
                    raise ValueError(f"adapter {name}: {module} is not a Linear layer")
                layer = GatedLinear(layer, gate)
                setattr(parent, attr, layer)
            base = layer.base
            a, b = tensors[f"{module}.A"], tensors[f"{module}.B"]
            if a.shape != (spec.rank, base.in_features) or b.shape != (
                base.out_features,
                spec.rank,
            ):
                raise ValueError(f"adapter {name}: factor shapes do not fit {module}")
            layer.add(name, a.to(device), b.to(device), spec.alpha / spec.rank)
        counts[name] = len(modules)
    return counts
