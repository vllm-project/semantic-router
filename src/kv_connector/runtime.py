"""Artifact loading and full-head mapper application for the vLLM connector."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from src.training.kv_mapper.artifact import (
    CompatibilitySpec,
    Manifest,
    read_artifact,
    verify_compatibility,
)

_KV_TENSOR_RANK = 3


@dataclass(frozen=True)
class MapperArtifact:
    path: Path
    manifest: Manifest
    _weights: dict[str, torch.Tensor] = field(repr=False, compare=False)
    _device_weights: dict[torch.device, dict[str, torch.Tensor]] = field(
        default_factory=dict, repr=False, compare=False
    )

    @classmethod
    def open(cls, path: Path, deployment: CompatibilitySpec) -> MapperArtifact:
        """Check the entire artifact before vLLM can advertise a cache hit."""
        # read_artifact verifies checksums and applies the shared validate_artifact contract.
        manifest, tensors = read_artifact(path)
        verify_compatibility(manifest, deployment)
        if not manifest.rope_stripped_on_keys:
            raise ValueError("connector requires pre-RoPE key mapping")
        if (
            manifest.compatibility.source_tp != 1
            or manifest.compatibility.target_tp != 1
        ):
            raise ValueError("full-head connector currently requires TP=1")

        return cls(
            path=path,
            manifest=manifest,
            _weights={name: torch.from_numpy(value) for name, value in tensors.items()},
        )

    def apply_layer(
        self,
        target_layer: int,
        channel: str,
        source_layers: dict[int, torch.Tensor],
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Map pre-RoPE source K or source V, each [tokens, heads, head_dim]."""
        if channel not in ("k", "v"):
            raise ValueError(f"unknown channel: {channel}")
        sources = self.manifest.source_layers_per_target[channel][str(target_layer)]
        parts = [source_layers[index] for index in sources]
        width = (
            self.manifest.compatibility.num_kv_heads
            * self.manifest.compatibility.head_dim
        )
        if not parts or any(
            part.ndim != _KV_TENSOR_RANK
            or part.shape[1:]
            != (
                self.manifest.compatibility.num_kv_heads,
                self.manifest.compatibility.head_dim,
            )
            for part in parts
        ):
            raise ValueError("source KV shape does not match mapper manifest")
        tokens = parts[0].shape[0]
        if any(part.shape[0] != tokens for part in parts):
            raise ValueError("source layers have different token counts")
        x = torch.cat([part.reshape(tokens, width) for part in parts], dim=-1)
        target_device = torch.device(device)
        if target_device.type == "cuda" and target_device.index is None:
            target_device = torch.device("cuda", torch.cuda.current_device())
        if target_device not in self._device_weights:
            self._device_weights[target_device] = {
                name: tensor.to(device=target_device)
                for name, tensor in self._weights.items()
            }
        weights = self._device_weights[target_device]
        w = weights[f"target.{target_layer}.{channel}.W"]
        b = weights[f"target.{target_layer}.{channel}.b"]
        # The fitting run solves in float64; accumulate in float32 at serving time.
        y = x.to(device=target_device, dtype=torch.float32) @ w
        y += b
        return y.to(dtype=dtype).reshape(
            tokens,
            self.manifest.compatibility.num_kv_heads,
            self.manifest.compatibility.head_dim,
        )
