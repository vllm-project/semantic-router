"""Vela 1.0 Omni readouts over the native towers: pooling, projections and the CLAP residual.

Each readout follows the published model's ``encode_text`` / ``encode_image``
/ ``encode_audio`` in its operation order, including where it normalizes
twice. A vector that is not finite or has no length is ``None`` (the
published model raises; the runtime answers ``invalid_model_output``).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .package import OmniPackage

NORM_FLOOR = 1e-12
# SiglipVisionConfig's width, for a vision config that omits it (Nano's).
SIGLIP_WIDTH = 768


def unit(vector: torch.Tensor) -> torch.Tensor | None:
    """``F.normalize`` of rows, or None when one is not finite or has no length."""
    if not bool(torch.isfinite(vector).all()) or bool(
        (vector.norm(dim=-1) <= NORM_FLOOR).any()
    ):
        return None
    return F.normalize(vector, dim=-1)


class ClapProjection(nn.Module):
    """CLAP's audio projection: ``linear2(relu(linear1(pooled)))``."""

    def __init__(self, hidden: int, width: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden, width)
        self.linear2 = nn.Linear(width, width)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        projected: torch.Tensor = self.linear2(F.relu(self.linear1(pooled)))
        return projected


class Residual(nn.Module):
    """The frozen CLAP residual: ``weight @ ((clap - mean) / scale)`` added to the speech projection."""

    mean: torch.Tensor
    scale: torch.Tensor

    def __init__(self, dimension: int, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dimension, width))
        self.register_buffer("mean", torch.empty(width))
        self.register_buffer("scale", torch.empty(width))

    def forward(self, clap: torch.Tensor) -> torch.Tensor:
        return F.linear((clap - self.mean) / self.scale, self.weight)


class OmniReadout(nn.Module):
    """Every readout of one variant, loaded from ``model.safetensors``."""

    def __init__(self, package: OmniPackage):
        super().__init__()
        contract, layout = package.contract, package.layout
        vision = package.vision_config.get("hidden_size", SIGLIP_WIDTH)
        speech = package.speech_config["d_model"]
        clap = package.clap_config
        self.variant = package.variant
        self.dimension = contract.dimension
        self.normalize_pooled_image = layout.normalize_pooled_image
        self.image_projection = nn.Linear(vision, contract.dimension)
        self.speech_projection = nn.Linear(speech, contract.dimension)
        self.clap_projection = ClapProjection(
            clap["patch_embeds_hidden_size"] * 2 ** (len(clap["depths"]) - 1),
            clap["projection_dim"],
        )
        self.residual = Residual(contract.dimension, clap["projection_dim"])
        self.load(package.weights, package)

    def load(self, weights: Path, package: OmniPackage) -> None:
        from safetensors import safe_open

        layout = package.layout
        names = {
            f"{layout.image_projection}{leaf}": f"image_projection.{leaf}"
            for leaf in ("weight", "bias")
        }
        names.update(
            {
                f"{layout.speech_projection}{leaf}": f"speech_projection.{leaf}"
                for leaf in ("weight", "bias")
            }
        )
        names.update(
            {
                f"{layout.clap_projection}{part}.{leaf}": f"clap_projection.{part}.{leaf}"
                for part in ("linear1", "linear2")
                for leaf in ("weight", "bias")
            }
        )
        names.update(
            {
                f"{layout.residual}{leaf}": f"residual.{leaf}"
                for leaf in ("weight", "mean", "scale")
            }
        )
        state = {}
        with safe_open(str(weights), framework="pt", device="cpu") as handle:
            for checkpoint, name in names.items():
                state[name] = handle.get_tensor(checkpoint).to(torch.float32, copy=True)
        self.load_state_dict(state, strict=True)
        scale = self.residual.scale
        if not bool(torch.isfinite(self.residual.mean).all()) or not bool(
            (torch.isfinite(scale) & (scale > 0)).all()
        ):
            raise ValueError("invalid frozen CLAP residual statistics")

    def text(self, hidden: torch.Tensor) -> torch.Tensor | None:
        """One text's embedding from its final hidden states ``[tokens, hidden]``.

        Nano reads its first token (CLS); Mini its last, normalized at full
        width and then over the published prefix.
        """
        if self.variant == "nano":
            first = unit(hidden[:1].float())
            return None if first is None else unit(first.float())
        full = unit(hidden[-1:].float())
        return None if full is None else unit(full[..., : self.dimension].float())

    def image(self, pooled: torch.Tensor) -> torch.Tensor | None:
        """One image's embedding from the vision tower's pooled vector ``[1, hidden]``."""
        if self.normalize_pooled_image:
            normalized = unit(pooled.float())
            if normalized is None:
                return None
            pooled = normalized
        projected = unit(self.image_projection(pooled))
        return None if projected is None else unit(projected.float())

    def clap(self, pooled: torch.Tensor) -> torch.Tensor | None:
        """The CLAP embedding of one input from its windows' pooled vectors ``[windows, hidden]``."""
        windows = unit(self.clap_projection(pooled).float())
        if windows is None or len(windows) == 1:
            return windows
        return unit(windows.mean(0, keepdim=True))

    def audio(self, speech: torch.Tensor, clap: torch.Tensor) -> torch.Tensor | None:
        """One input's embedding from Whisper's hidden states ``[1, frames, hidden]`` and its CLAP embedding."""
        affine = self.speech_projection(speech.mean(dim=1))
        fused = unit(affine + self.residual(clap))
        return None if fused is None else unit(fused.float())
