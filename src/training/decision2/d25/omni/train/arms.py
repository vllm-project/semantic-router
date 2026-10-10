"""Training arms and parameter groups.

Parameters split into four groups: ``encoder`` (ViT patch embedding, position table and blocks),
``merger`` (patch merger into the LM width), ``language`` (the Qwen3.5 text model) and ``readout``.
Each group trains at ``base_lr * scale``; scale 0 freezes it (``requires_grad=False``, no optimizer
state).

| Arm | Init | encoder | merger |
| --- | --- | --- | --- |
| O-graft-frozen | Vega 2.5 language + stock vision | 0 | 1 |
| O-graft-lowlr | Vega 2.5 language + stock vision | 0.1 | 1 |
| O-fresh | stock Qwen3.8-27B | 0 | 1 |

The two graft arms differ only in the encoder rate; O-fresh differs from O-graft-frozen only in the
init. ``--merger-lr-scale 0`` gives a fully frozen vision tower.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

GROUPS = ("encoder", "merger", "language", "readout")


@dataclass(frozen=True)
class Arm:
    init: str
    encoder: float
    merger: float
    language: float = 1.0
    readout: float = 1.0

    def scales(self) -> dict[str, float]:
        return {
            "encoder": self.encoder,
            "merger": self.merger,
            "language": self.language,
            "readout": self.readout,
        }


ARMS = {
    "O-graft-frozen": Arm(init="vega", encoder=0.0, merger=1.0),
    "O-graft-lowlr": Arm(init="vega", encoder=0.1, merger=1.0),
    "O-fresh": Arm(init="stock", encoder=0.0, merger=1.0),
}


def group_of(name: str) -> str:
    if name.startswith("readout."):
        return "readout"
    if name.startswith("backbone.visual.merger."):
        return "merger"
    if name.startswith("backbone.visual."):
        return "encoder"
    if name.startswith("backbone.language_model."):
        return "language"
    raise ValueError(f"parameter outside the known groups: {name}")


def apply(model: torch.nn.Module, scales: dict[str, float]) -> dict[str, int]:
    """Freeze zero-scale groups; returns trainable parameter counts per group."""
    if set(scales) != set(GROUPS) or any(value < 0 for value in scales.values()):
        raise ValueError(f"scales must give a non-negative value for each of {GROUPS}")
    counts = dict.fromkeys(GROUPS, 0)
    for name, parameter in model.named_parameters():
        group = group_of(name)
        parameter.requires_grad_(scales[group] > 0)
        if scales[group] > 0:
            counts[group] += parameter.numel()
    return counts


def param_groups(
    model: torch.nn.Module,
    scales: dict[str, float],
    weight_decay: float,
    decay_1d: bool = True,
) -> list[dict]:
    """AdamW groups carrying ``lr_scale``; call after ``apply`` (and after sharding)."""
    buckets: dict[tuple[str, bool], list[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        decays = decay_1d or parameter.ndim >= 2
        buckets.setdefault((group_of(name), decays), []).append(parameter)
    return [
        {
            "params": params,
            "name": f"{group}{'' if decays else '-no-decay'}",
            "lr_scale": scales[group],
            "weight_decay": weight_decay if decays else 0.0,
        }
        for (group, decays), params in sorted(buckets.items())
    ]
