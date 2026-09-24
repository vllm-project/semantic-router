#!/usr/bin/env python3
"""Fit an artifact from checked calibration chunks. Requires a high-memory host."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.training.kv_mapper.artifact import CompatibilitySpec, read_artifact
from src.training.kv_mapper.collect import (
    read_run_metadata,
    token_fingerprint,
    validate_activation_chunk,
)
from src.training.kv_mapper.fit import fit_full_head, write_fitted_artifact
from src.training.kv_mapper.mapper_id import make_mapper_id


def load_captures(run_dir: Path):
    """Validate every chunk, then allocate the sampled tensors once in host RAM."""
    meta = read_run_metadata(run_dir)
    tokens = np.load(run_dir / "tokens.npy", allow_pickle=False)
    if (
        tokens.shape != (meta.num_sequences, meta.seq_len)
        or token_fingerprint(tokens) != meta.token_sha256
    ):
        raise ValueError("token windows do not match run metadata")
    rows = len(range(0, meta.seq_len, meta.fitting_token_step))
    captures = {}
    for side, layers in (
        ("source", meta.source_layers),
        ("target", meta.target_layers),
    ):
        shape = (
            len(layers),
            meta.num_sequences * rows,
            meta.num_kv_heads,
            meta.head_dim,
        )
        keys = np.empty(shape, dtype=np.float32)
        values = np.empty(shape, dtype=np.float32)
        for sequence in range(meta.num_sequences):
            path = run_dir / side / f"{sequence:06d}.npz"
            if not validate_activation_chunk(
                path, len(layers), rows, meta.num_kv_heads, meta.head_dim
            ):
                raise FileNotFoundError(f"missing activation chunk: {path}")
            with np.load(path, allow_pickle=False) as chunk:
                sl = slice(sequence * rows, (sequence + 1) * rows)
                keys[:, sl] = chunk["keys"]
                values[:, sl] = chunk["values"]
        captures[side] = (keys, values)
    return meta, captures


def _channel_scores(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """All layer-pair per-head affine OLS R², using shared covariance GEMMs."""
    n_src, rows, heads, dim = source.shape
    n_tgt = target.shape[0]
    if target.shape[1:] != (rows, heads, dim):
        raise ValueError("source and target capture shapes differ")
    scores = np.empty((n_src, n_tgt, heads), dtype=np.float64)
    for head in range(heads):
        x = np.ascontiguousarray(
            source[:, :, head, :].transpose(1, 0, 2).reshape(rows, n_src * dim),
            dtype=np.float64,
        )
        y = np.ascontiguousarray(
            target[:, :, head, :].transpose(1, 0, 2).reshape(rows, n_tgt * dim),
            dtype=np.float64,
        )
        sx = x.sum(axis=0).reshape(n_src, dim)
        sy = y.sum(axis=0).reshape(n_tgt, dim)
        cross = (x.T @ y).reshape(n_src, dim, n_tgt, dim).transpose(0, 2, 1, 3)
        cross -= sx[:, None, :, None] * sy[None, :, None, :] / rows
        total = (
            np.square(y).sum(axis=0).reshape(n_tgt, dim).sum(axis=1)
            - np.square(sy).sum(axis=1) / rows
        )
        for source_layer in range(n_src):
            xs = x[:, source_layer * dim : (source_layer + 1) * dim]
            cov = xs.T @ xs - np.outer(sx[source_layer], sx[source_layer]) / rows
            rhs = cross[source_layer].transpose(1, 0, 2).reshape(dim, n_tgt * dim)
            weight, *_ = np.linalg.lstsq(cov, rhs, rcond=None)
            explained = np.sum(rhs * weight, axis=0).reshape(n_tgt, dim).sum(axis=1)
            scores[source_layer, :, head] = np.where(
                total > 0,
                1 - np.maximum(total - explained, 0) / np.maximum(total, 1e-30),
                0,
            )
        del x, y, cross
    return scores


def select_shared_layers(captures: dict, k: int) -> list[list[int]]:
    source_k, source_v = captures["source"]
    target_k, target_v = captures["target"]
    if not 1 <= k <= source_k.shape[0]:
        raise ValueError("topk exceeds the number of captured source layers")
    scores = (
        _channel_scores(source_k, target_k) + _channel_scores(source_v, target_v)
    ) / 2
    scores = scores.mean(axis=2)
    return [
        [int(i) for i in np.argsort(-scores[:, t], kind="stable")[:k]]
        for t in range(target_k.shape[0])
    ]


def _work_dir(
    run_dir: Path, out_dir: Path, meta, topk: int, ridge_alpha: float
) -> Path:
    recipe = {
        "run": meta.to_dict(),
        "topk": topk,
        "ridge_alpha": ridge_alpha,
        "chunks": [],
    }
    for side in ("source", "target"):
        for sequence in range(meta.num_sequences):
            recipe["chunks"].append(
                (run_dir / side / f"{sequence:06d}.sha256").read_text().strip()
            )
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[
        :16
    ]
    work = out_dir / ".fit-work" / digest
    work.mkdir(parents=True, exist_ok=True)
    return work


def _partial(
    path: Path, weight: np.ndarray | None = None, bias: np.ndarray | None = None
):
    checksum = path.with_suffix(".sha256")
    if weight is not None and bias is not None:
        temp = path.with_suffix(".tmp")
        with temp.open("wb") as stream:
            np.savez(stream, weight=weight, bias=bias)
        digest = hashlib.sha256(temp.read_bytes()).hexdigest()
        os.replace(temp, path)
        checksum.write_text(digest + "\n")
    if not path.exists():
        return None
    if (
        not checksum.exists()
        or hashlib.sha256(path.read_bytes()).hexdigest() != checksum.read_text().strip()
    ):
        raise ValueError(f"corrupt fit partial: {path}")
    with np.load(path, allow_pickle=False) as data:
        return data["weight"], data["bias"]


def fit_capture(
    run_dir: Path,
    out_dir: Path,
    *,
    pair_slug: str,
    topk: int = 8,
    ridge_alpha: float = 0.01,
    bundle_version: int = 1,
) -> Path:
    if bundle_version < 1:
        raise ValueError("bundle version must be positive")
    meta, captures = load_captures(run_dir)
    if meta.target_layers != list(range(len(meta.target_layers))):
        raise ValueError("artifact requires all target layers in order")
    compatibility = CompatibilitySpec(
        source_model=meta.source_model,
        source_revision=meta.source_revision,
        target_model=meta.target_model,
        target_revision=meta.target_revision,
        variant="full_head",
        precision=meta.precision,
        source_tp=1,
        target_tp=1,
        head_order="contiguous",
        num_kv_heads=meta.num_kv_heads,
        head_dim=meta.head_dim,
    )
    mapper_id = make_mapper_id(
        pair_slug=pair_slug,
        variant=compatibility.variant,
        precision=compatibility.precision,
        source_revision=compatibility.source_revision,
        target_revision=compatibility.target_revision,
        source_tp=1,
        target_tp=1,
        n_kv_heads=compatibility.num_kv_heads,
        bundle_version=bundle_version,
    )
    artifact_dir = out_dir / mapper_id
    if artifact_dir.exists():
        raise FileExistsError(
            f"artifact already exists: {artifact_dir}; choose another --bundle-version"
        )
    work = _work_dir(run_dir, out_dir, meta, topk, ridge_alpha)
    selection_path = work / "selection.json"
    if selection_path.exists():
        chosen = json.loads(selection_path.read_text())
    else:
        chosen = select_shared_layers(captures, topk)
        selection_path.write_text(json.dumps(chosen) + "\n")
    source_k, source_v = captures["source"]
    target_k, target_v = captures["target"]
    tensors = {}
    for target_layer, indices in enumerate(chosen):
        for channel, source, target in (
            ("k", source_k, target_k),
            ("v", source_v, target_v),
        ):
            path = work / f"target.{target_layer}.{channel}.npz"
            partial = _partial(path)
            if partial is None:
                fitted = fit_full_head(
                    [
                        (
                            [source[i] for i in range(source.shape[0])],
                            [target[target_layer]],
                        )
                    ],
                    [indices],
                    n_kv_heads=meta.num_kv_heads,
                    head_dim=meta.head_dim,
                    ridge_alpha=ridge_alpha,
                    channel=channel,
                )
                partial = _partial(
                    path,
                    fitted[f"target.0.{channel}.W"],
                    fitted[f"target.0.{channel}.b"],
                )
            tensors[f"target.{target_layer}.{channel}.W"] = partial[0]
            tensors[f"target.{target_layer}.{channel}.b"] = partial[1]
    mapping = {
        str(t): [meta.source_layers[i] for i in indices]
        for t, indices in enumerate(chosen)
    }
    staging = out_dir / f".{mapper_id}.publishing"
    if staging.exists():
        shutil.rmtree(staging)
    written_id = write_fitted_artifact(
        staging,
        compat=compatibility,
        tensors=tensors,
        topk=topk,
        ridge_alpha=ridge_alpha,
        source_layers_per_target={"k": mapping, "v": mapping},
        pair_slug=pair_slug,
        bundle_version=bundle_version,
        calibration=meta.to_dict(),
    )
    if written_id != mapper_id:
        raise RuntimeError("artifact ID changed while publishing")
    read_artifact(staging / mapper_id)
    os.replace(staging / mapper_id, artifact_dir)
    staging.rmdir()
    return artifact_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-slug", required=True)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--ridge-alpha", type=float, default=0.01)
    parser.add_argument("--bundle-version", type=int, default=1)
    args = parser.parse_args()
    result = fit_capture(
        args.run_dir,
        args.output_dir,
        pair_slug=args.pair_slug,
        topk=args.topk,
        ridge_alpha=args.ridge_alpha,
        bundle_version=args.bundle_version,
    )
    print(json.dumps({"artifact": str(result)}))


if __name__ == "__main__":
    main()
