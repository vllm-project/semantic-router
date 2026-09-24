"""Calibration windows and activation-run metadata. No model weights in git."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Iterator

import numpy as np


def resolve_stride(stride: int, seq_len: int) -> int:
    """Token step between window starts. <=0 or >= seq_len means non-overlapping."""
    if stride <= 0:
        return seq_len
    return min(int(stride), seq_len)


def calibration_windows(
    tokenized_documents: Iterable[list[int]],
    *,
    seq_len: int,
    stride: int,
    num_sequences: int,
) -> Iterator[list[int]]:
    """Yield exact-length token windows from the selected corpus documents."""
    if seq_len <= 0 or num_sequences <= 0:
        raise ValueError("seq_len and num_sequences must be positive")
    step = resolve_stride(stride, seq_len)
    tokens: list[int] = []
    emitted = 0
    for document in tokenized_documents:
        tokens.extend(document)
        while len(tokens) >= seq_len:
            yield tokens[:seq_len]
            emitted += 1
            if emitted == num_sequences:
                return
            del tokens[:step]
    raise ValueError(
        f"corpus provided only {emitted} complete windows; requested {num_sequences}"
    )


def as_bshd_numpy(out: np.ndarray, n_kv: int, head_dim: int) -> np.ndarray:
    """Normalize hook output to (batch, seq, n_kv, head_dim)."""
    if out.ndim == 4:
        _batch, a, c, d = out.shape
        if c == n_kv and d == head_dim:
            return out
        if a == n_kv and d == head_dim:
            return np.transpose(out, (0, 2, 1, 3))
        raise ValueError(
            f"Unexpected 4D KV {tuple(out.shape)}; want n_kv={n_kv} head_dim={head_dim}"
        )
    if out.ndim != 3:
        raise ValueError(f"Unexpected KV rank {out.ndim} shape {tuple(out.shape)}")
    return out.reshape(out.shape[0], out.shape[1], n_kv, head_dim)


def parse_layer_subset(spec: str | None, n_layers: int) -> list[int]:
    """'0:8' or '0,2,5' or None for all layers in 0..n_layers-1."""
    if spec is None or spec.strip() in ("", "all"):
        return list(range(n_layers))
    spec = spec.strip()
    if ":" in spec:
        start_s, end_s = spec.split(":", 1)
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else n_layers
        return list(range(start, end))
    return [int(p) for p in spec.split(",") if p.strip()]


@dataclass
class ActivationRunMeta:
    corpus: str
    dataset_config: str
    source_model: str
    source_revision: str
    target_model: str
    target_revision: str
    seed: int
    seq_len: int
    stride: int
    num_sequences: int
    source_layers: list[int]
    target_layers: list[int]
    num_kv_heads: int
    head_dim: int
    precision: str
    rope_stripped_on_keys: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActivationRunMeta:
        return cls(**data)


def write_run_metadata(out_dir: Path, meta: ActivationRunMeta) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run.json"
    path.write_text(json.dumps(meta.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


def read_run_metadata(out_dir: Path) -> ActivationRunMeta:
    return ActivationRunMeta.from_dict(json.loads((out_dir / "run.json").read_text()))
