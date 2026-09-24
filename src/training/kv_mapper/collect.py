"""Calibration windows and activation-run metadata. No model weights in git."""

from __future__ import annotations

import json
import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Iterator

import numpy as np


def resolve_stride(stride: int, seq_len: int) -> int:
    """Token step between window starts; zero requests disjoint windows."""
    if stride < 0:
        raise ValueError("window stride must be nonnegative")
    return stride or seq_len


def calibration_windows(
    tokenized_documents: Iterable[list[int]],
    *,
    seq_len: int,
    window_stride: int,
    num_sequences: int,
) -> Iterator[list[int]]:
    """Yield exact-length token windows from the selected corpus documents."""
    if seq_len <= 0 or num_sequences <= 0:
        raise ValueError("seq_len and num_sequences must be positive")
    step = resolve_stride(window_stride, seq_len)
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
    dataset_revision: str
    source_model: str
    source_revision: str
    target_model: str
    target_revision: str
    seed: int
    seq_len: int
    window_stride: int
    fitting_token_step: int
    token_sha256: str
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
    payload = json.dumps(meta.to_dict(), indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != payload:
        raise ValueError("existing run metadata differs; choose another output directory")
    if not path.exists():
        temp = path.with_suffix(".tmp")
        temp.write_text(payload)
        os.replace(temp, path)
    return path


def read_run_metadata(out_dir: Path) -> ActivationRunMeta:
    return ActivationRunMeta.from_dict(json.loads((out_dir / "run.json").read_text()))


def token_fingerprint(windows: np.ndarray) -> str:
    """Stable digest of the exact token IDs and their two-dimensional shape."""
    canonical = np.ascontiguousarray(windows, dtype="<i8")
    return hashlib.sha256(np.asarray(canonical.shape, dtype="<i8").tobytes() + canonical.tobytes()).hexdigest()


def write_activation_chunk(path: Path, keys: list[np.ndarray], values: list[np.ndarray]) -> None:
    """Write one sampled sequence atomically, with a digest for safe resume."""
    if len(keys) != len(values) or not keys:
        raise ValueError("chunk needs matching, nonempty K/V layers")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as stream:
        np.savez(stream, keys=np.stack(keys), values=np.stack(values))
    digest = hashlib.sha256(temp.read_bytes()).hexdigest()
    os.replace(temp, path)
    path.with_suffix(".sha256").write_text(digest + "\n")


def validate_activation_chunk(path: Path, n_layers: int, n_rows: int, n_heads: int, head_dim: int) -> bool:
    if not path.exists():
        return False
    digest_path = path.with_suffix(".sha256")
    if not digest_path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest_path.read_text().strip():
        raise ValueError(f"corrupt or incomplete activation chunk: {path}")
    expected = (n_layers, n_rows, n_heads, head_dim)
    with np.load(path, allow_pickle=False) as chunk:
        if chunk["keys"].shape != expected or chunk["values"].shape != expected:
            raise ValueError(f"activation chunk {path} does not match expected shape {expected}")
    return True
