"""Fit a full-head ridge mapper and write an A1 artifact directory."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.training.kv_mapper.artifact import (
    CompatibilitySpec,
    Manifest,
    write_artifact,
)
from src.training.kv_mapper.mapper_id import make_mapper_id
from src.training.kv_mapper.ridge import RidgeAccumulator, ols_r2_per_head


def stack_source_full_head(
    src_layers: list[np.ndarray], source_idxs: list[int]
) -> np.ndarray:
    """(seq, k * n_kv * d) from selected source layers."""
    parts = [src_layers[i].reshape(src_layers[i].shape[0], -1) for i in source_idxs]
    return np.concatenate(parts, axis=-1)


def select_source_layers(
    source_keys: list[np.ndarray],
    target_keys: list[np.ndarray],
    source_values: list[np.ndarray],
    target_values: list[np.ndarray],
    k: int,
) -> list[list[int]]:
    """One shared top-k list per target, from per-head OLS R² averaged over K/V."""
    if not source_keys or not target_keys or len(source_keys) != len(source_values) or len(target_keys) != len(target_values):
        raise ValueError("K/V layer counts must match and be nonempty")
    if not 1 <= k <= len(source_keys):
        raise ValueError(f"k must be between 1 and {len(source_keys)}")
    selected = []
    for target_k, target_v in zip(target_keys, target_values):
        scores = []
        for source_k, source_v in zip(source_keys, source_values):
            head_scores = np.concatenate((ols_r2_per_head(source_k, target_k), ols_r2_per_head(source_v, target_v)))
            scores.append(float(head_scores.mean()))
        selected.append([int(i) for i in np.argsort(-np.asarray(scores), kind="stable")[:k]])
    return selected


def fit_full_head(
    pairs: list[tuple[list[np.ndarray], list[np.ndarray]]],
    source_idxs: list[list[int]],
    *,
    n_kv_heads: int,
    head_dim: int,
    ridge_alpha: float,
    channel: str = "k",
) -> dict[str, np.ndarray]:
    """pairs: list of (src_layers, tgt_layers), each layer (seq, n_kv, d)."""
    n_tgt = len(source_idxs)
    k = len(source_idxs[0])
    dx = k * n_kv_heads * head_dim
    dy = n_kv_heads * head_dim
    acc = [RidgeAccumulator(dx, dy) for _ in range(n_tgt)]
    for src_layers, tgt_layers in pairs:
        for t, idxs in enumerate(source_idxs):
            x = stack_source_full_head(src_layers, idxs)
            y = tgt_layers[t].reshape(-1, dy)
            if x.shape[0] != y.shape[0]:
                raise ValueError(f"source and target token counts differ for target layer {t}: {x.shape[0]} vs {y.shape[0]}")
            acc[t].add(x, y)
    tensors: dict[str, np.ndarray] = {}
    for t, bank in enumerate(acc):
        weight, bias = bank.solve_affine(ridge_alpha)
        tensors[f"target.{t}.{channel}.W"] = weight
        tensors[f"target.{t}.{channel}.b"] = bias
    return tensors


def write_fitted_artifact(
    out_dir: Path,
    *,
    compat: CompatibilitySpec,
    tensors: dict[str, np.ndarray],
    topk: int,
    ridge_alpha: float,
    source_layers_per_target: dict[str, dict[str, list[int]]],
    pair_slug: str,
    bundle_version: int = 1,
    calibration: dict | None = None,
) -> str:
    mapper_id = make_mapper_id(
        pair_slug=pair_slug,
        variant=compat.variant,
        precision=compat.precision,
        source_revision=compat.source_revision,
        target_revision=compat.target_revision,
        source_tp=compat.source_tp,
        target_tp=compat.target_tp,
        n_kv_heads=compat.num_kv_heads,
        bundle_version=bundle_version,
    )
    manifest = Manifest(
        mapper_id=mapper_id,
        compatibility=compat,
        topk=topk,
        ridge_alpha=ridge_alpha,
        centered_inputs=True,
        rope_stripped_on_keys=True,
        source_layers_per_target=source_layers_per_target,
        calibration=dict(calibration or {}),
    )
    dest = out_dir / mapper_id
    write_artifact(dest, manifest, tensors)
    return mapper_id
