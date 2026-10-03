"""Artifact loading and full-head mapper application for the vLLM connector."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from src.training.kv_mapper.artifact import (
    MANIFEST_FILE,
    WEIGHTS_FILE,
    CompatibilitySpec,
    Manifest,
    verify_checksums,
    verify_compatibility,
)

_KV_TENSOR_RANK = 3


@dataclass(frozen=True)
class MapperArtifact:
    path: Path
    manifest: Manifest

    @classmethod
    def open(cls, path: Path, deployment: CompatibilitySpec) -> MapperArtifact:
        """Check the entire artifact before vLLM can advertise a cache hit."""
        verify_checksums(path)
        manifest = Manifest.from_dict(json.loads((path / MANIFEST_FILE).read_text()))
        verify_compatibility(manifest, deployment)
        if manifest.schema_version != 1:
            raise ValueError(f"unsupported mapper schema: {manifest.schema_version}")
        if manifest.compatibility.variant != "full_head":
            raise ValueError("connector currently supports full_head artifacts only")
        if not manifest.rope_stripped_on_keys:
            raise ValueError("connector requires pre-RoPE key mapping")
        if (
            manifest.compatibility.source_tp != 1
            or manifest.compatibility.target_tp != 1
        ):
            raise ValueError("full-head connector currently requires TP=1")

        width = manifest.compatibility.num_kv_heads * manifest.compatibility.head_dim
        selections = manifest.source_layers_per_target
        if set(selections) != {"k", "v"}:
            raise ValueError("mapper requires K and V source lists")
        expected_layers = {int(layer) for layer in selections["k"]}
        if expected_layers != set(range(len(expected_layers))):
            raise ValueError("target layer indices must be contiguous from zero")
        if set(selections["v"]) != set(selections["k"]):
            raise ValueError("K and V target layers differ")
        with safe_open(path / WEIGHTS_FILE, framework="pt", device="cpu") as weights:
            names = set(weights.keys())
            expected_names: set[str] = set()
            for layer in sorted(expected_layers):
                for channel in ("k", "v"):
                    sources = selections[channel][str(layer)]
                    if len(sources) != manifest.topk or len(set(sources)) != len(
                        sources
                    ):
                        raise ValueError(f"invalid source layers for {layer}.{channel}")
                    if any(source < 0 for source in sources):
                        raise ValueError(f"negative source layer for {layer}.{channel}")
                    weight_name = f"target.{layer}.{channel}.W"
                    bias_name = f"target.{layer}.{channel}.b"
                    expected_names.update((weight_name, bias_name))
                    weight = weights.get_slice(weight_name)
                    shape = weight.get_shape()
                    if shape != [len(sources) * width, width]:
                        raise ValueError(f"invalid shape for {weight_name}: {shape}")
                    bias = weights.get_slice(bias_name)
                    if bias.get_shape() != [width]:
                        raise ValueError(f"invalid shape for {bias_name}")
                    if weight.get_dtype() != "F32" or bias.get_dtype() != "F32":
                        raise ValueError(
                            f"mapper tensors must be float32 for {layer}.{channel}"
                        )
            if names != expected_names:
                raise ValueError("mapper tensor keys do not match the manifest")
        return cls(path=path, manifest=manifest)

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
        with safe_open(
            self.path / WEIGHTS_FILE, framework="pt", device="cpu"
        ) as weights:
            w = weights.get_tensor(f"target.{target_layer}.{channel}.W")
            b = weights.get_tensor(f"target.{target_layer}.{channel}.b")
        # The fitting run solves in float64; accumulate in float32 at serving time.
        y = x.to(device=device, dtype=torch.float32) @ w.to(
            device=device, dtype=torch.float32
        )
        y += b.to(device=device, dtype=torch.float32)
        return y.to(dtype=dtype).reshape(
            tokens,
            self.manifest.compatibility.num_kv_heads,
            self.manifest.compatibility.head_dim,
        )
