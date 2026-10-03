"""Immutable, tenant-scoped source KV snapshots for a same-host C2 probe.

The snapshot store is a bounded reference transport for two vLLM processes
sharing a filesystem. A source pod address and remote transport are separate
work; callers must never infer a hit from a cache ID alone.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file

from src.training.kv_mapper.mapper_id import require_weight_commit

_KV_CHANNELS = 2
_KV_TENSOR_RANK = 3


@dataclass(frozen=True)
class SourceSnapshot:
    namespace: str
    cache_id: str
    source_model: str
    source_revision: str
    token_ids: tuple[int, ...]
    layers: dict[int, tuple[torch.Tensor, torch.Tensor]]
    rope_theta: float
    expires_at: float


def _key(namespace: str, cache_id: str) -> str:
    if not namespace or not cache_id:
        raise ValueError("namespace and cache ID are required")
    digest = hashlib.sha256()
    for value in (namespace, cache_id):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _validate(snapshot: SourceSnapshot) -> None:
    if not snapshot.source_model:
        raise ValueError("source model and pinned revision are required")
    require_weight_commit(snapshot.source_revision)
    if not snapshot.token_ids or not snapshot.layers:
        raise ValueError("source snapshot must contain tokens and KV")
    if any(not isinstance(token, int) or token < 0 for token in snapshot.token_ids):
        raise ValueError("source snapshot token IDs must be nonnegative integers")
    if not math.isfinite(snapshot.expires_at):
        raise ValueError("source snapshot expiry must be finite")
    if not math.isfinite(snapshot.rope_theta) or snapshot.rope_theta <= 0:
        raise ValueError("source snapshot RoPE theta must be positive and finite")
    if set(snapshot.layers) != set(range(len(snapshot.layers))):
        raise ValueError("source layers must be contiguous from zero")
    shape: tuple[int, ...] | None = None
    dtype: torch.dtype | None = None
    for key, value in snapshot.layers.items():
        if len(value) != _KV_CHANNELS:
            raise ValueError(f"source layer {key} must contain K and V")
        for tensor in value:
            if tensor.ndim != _KV_TENSOR_RANK or tensor.shape[0] != len(
                snapshot.token_ids
            ):
                raise ValueError("source KV must be [tokens, heads, head_dim]")
            if shape is None:
                shape, dtype = tuple(tensor.shape), tensor.dtype
            if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise ValueError("source KV layers must have equal shape and dtype")


class LocalSnapshotStore:
    """Atomic local transport for a single trusted machine and tenant scope."""

    def __init__(self, root: Path):
        self.root = root

    def _path(self, namespace: str, cache_id: str) -> Path:
        return self.root / f"{_key(namespace, cache_id)}.safetensors"

    def prune_expired(self, now: float | None = None) -> int:
        """Remove expired snapshots from this trusted local directory."""
        cutoff = time.time() if now is None else now
        removed = 0
        for path in self.root.glob("*.safetensors"):
            try:
                with safe_open(path, framework="pt", device="cpu") as source:
                    expiry = float((source.metadata() or {})["expires_at"])
            except (OSError, KeyError, TypeError, ValueError, SafetensorError):
                continue
            if expiry <= cutoff:
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
                removed += 1
        return removed

    def publish(self, snapshot: SourceSnapshot) -> Path:
        _validate(snapshot)
        if snapshot.expires_at <= time.time():
            raise ValueError("cannot publish an expired source snapshot")
        path = self._path(snapshot.namespace, snapshot.cache_id)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.prune_expired()
        temp = self.root / f".{uuid.uuid4().hex}.safetensors"
        tensors = {"token_ids": torch.tensor(snapshot.token_ids, dtype=torch.int64)}
        for layer, (key, value) in snapshot.layers.items():
            tensors[f"layer.{layer}.k"] = key.detach().contiguous().cpu()
            tensors[f"layer.{layer}.v"] = value.detach().contiguous().cpu()
        metadata = {
            "namespace": snapshot.namespace,
            "cache_id": snapshot.cache_id,
            "source_model": snapshot.source_model,
            "source_revision": snapshot.source_revision,
            "expires_at": str(snapshot.expires_at),
            "rope_theta": str(snapshot.rope_theta),
            "layer_count": str(len(snapshot.layers)),
        }
        try:
            save_file(tensors, str(temp), metadata=metadata)
            os.chmod(temp, 0o600)
            # Publishing a second version under the same key could replace a
            # snapshot after the target has checked it. A new source prefix
            # must use a new cache ID instead.
            os.link(temp, path)
        finally:
            temp.unlink(missing_ok=True)
        return path

    def load(
        self,
        namespace: str,
        cache_id: str,
        *,
        source_model: str,
        source_revision: str,
        now: float | None = None,
    ) -> SourceSnapshot:
        path = self._path(namespace, cache_id)
        with safe_open(path, framework="pt", device="cpu") as source:
            metadata = source.metadata() or {}
            if (
                metadata.get("namespace") != namespace
                or metadata.get("cache_id") != cache_id
                or metadata.get("source_model") != source_model
                or metadata.get("source_revision") != source_revision
            ):
                raise ValueError("source snapshot identity mismatch")
            expires_at = float(metadata["expires_at"])
            if expires_at <= (time.time() if now is None else now):
                path.unlink(missing_ok=True)
                raise ValueError("source snapshot expired")
            layer_count = int(metadata["layer_count"])
            expected = {"token_ids"} | {
                f"layer.{layer}.{channel}"
                for layer in range(layer_count)
                for channel in ("k", "v")
            }
            if set(source.keys()) != expected:
                raise ValueError("source snapshot layer keys are incomplete")
            token_ids = source.get_tensor("token_ids")
            if token_ids.dtype != torch.int64 or token_ids.ndim != 1:
                raise ValueError("source snapshot token IDs are malformed")
            snapshot = SourceSnapshot(
                namespace=namespace,
                cache_id=cache_id,
                source_model=source_model,
                source_revision=source_revision,
                token_ids=tuple(int(token) for token in token_ids.tolist()),
                layers={
                    layer: (
                        source.get_tensor(f"layer.{layer}.k"),
                        source.get_tensor(f"layer.{layer}.v"),
                    )
                    for layer in range(layer_count)
                },
                rope_theta=float(metadata["rope_theta"]),
                expires_at=expires_at,
            )
        _validate(snapshot)
        return snapshot
