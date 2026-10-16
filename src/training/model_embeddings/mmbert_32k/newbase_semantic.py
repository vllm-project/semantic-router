"""Absolute semantic anchors and full logical-batch geometry distillation."""

from __future__ import annotations

import torch
from torch.nn import functional

from .newbase_objectives import relational_cosine_loss

_MATRIX_DIMENSIONS = 2


def pointwise_cosine_loss(student, teacher):
    """Mean angular distance, without an implicit division by embedding width.

    This same-width objective transfers absolute coordinates. A different-width
    teacher requires relation distillation or an explicit learned projection.
    """
    if (
        student.ndim != _MATRIX_DIMENSIONS
        or student.shape != teacher.shape
        or not student.numel()
    ):
        raise ValueError("Pointwise anchors require nonempty matching matrices")
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise ValueError("Pointwise anchors must be finite")
    if (student.norm(dim=-1) == 0).any() or (teacher.norm(dim=-1) == 0).any():
        raise ValueError("A zero vector has no angular target")
    left = functional.normalize(student.float(), dim=-1)
    right = functional.normalize(teacher.detach().float(), dim=-1)
    # Squared distance avoids negative loss from a dot product rounded above 1.
    return (left - right).square().sum(-1).mean() / 2


def all_exit_anchor_loss(values, teacher, exits, *, target_layers=None):
    """Matryoshka prefixes of an explicitly shared or layer-matched target.

    Prefixes are re-normalized, never used to hide different teacher widths.
    Without target_layers every depth shares one teacher matrix. Otherwise the
    declared layer order binds the second axis of [input, layer, dimension];
    missing layers never fall back to the teacher's final layer.
    """
    if target_layers is None:
        if (
            teacher.ndim != _MATRIX_DIMENSIONS
            or teacher.shape[1] != exits.dimensions[0]
        ):
            raise ValueError("Anchor teacher must match the full student width")
        references = dict.fromkeys(exits.layers, teacher)
    else:
        if (
            not isinstance(target_layers, (list, tuple))
            or not target_layers
            or any(type(layer) is not int or layer <= 0 for layer in target_layers)
            or len(set(target_layers)) != len(target_layers)
            or set(target_layers) != set(exits.layers)
            or teacher.ndim != _MATRIX_DIMENSIONS + 1
            or teacher.shape[1:] != (len(target_layers), exits.dimensions[0])
        ):
            raise ValueError(
                "Layer anchors must match every declared student depth and width"
            )
        references = {
            layer: teacher[:, index] for index, layer in enumerate(target_layers)
        }
    return sum(
        weight * pointwise_cosine_loss(values[key], references[key[0]][:, : key[1]])
        for key, weight in exits.weighted()
    )


def full_batch_relation_loss(student, teacher, component_ids, components):
    """Cross-parent relations across the entire logical batch, with stop-gradient."""
    if len(component_ids) != len(student):
        raise ValueError("Relation identities and logical batch differ")
    mask = torch.tensor(
        [
            [
                components[a]["normalized_sha256"] != components[b]["normalized_sha256"]
                and not set(components[a]["parent_groups"])
                & set(components[b]["parent_groups"])
                for b in component_ids
            ]
            for a in component_ids
        ],
        dtype=torch.bool,
        device=student.device,
    )
    return relational_cosine_loss(student, student, teacher, teacher, mask)
