"""Native backbone architectures, keyed by the Transformers ``model_type`` of their config."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import torch
from torch import nn

from ....accel.kernels import KernelSet
from .bert import BertBackbone
from .clap import ClapAudioBackbone
from .forest import ForestShape
from .modernbert import Layout, ModernBertBackbone
from .qwen3 import Qwen3Backbone
from .qwen3_5 import Qwen3_5Backbone
from .siglip import SiglipVisionBackbone
from .tree import Tree
from .whisper import WhisperEncoderBackbone

ARCHITECTURES: dict[str, type[nn.Module]] = {
    BertBackbone.model_type: BertBackbone,
    ClapAudioBackbone.model_type: ClapAudioBackbone,
    ModernBertBackbone.model_type: ModernBertBackbone,
    Qwen3Backbone.model_type: Qwen3Backbone,
    Qwen3_5Backbone.model_type: Qwen3_5Backbone,
    SiglipVisionBackbone.model_type: SiglipVisionBackbone,
    WhisperEncoderBackbone.model_type: WhisperEncoderBackbone,
}


class Backbone(Protocol):
    """What every architecture holds: its config and the kernels the engine binds before a forward."""

    config: dict[str, Any]
    kernels: KernelSet | None


class EncoderBackbone(Protocol):
    """A bidirectional encoder: hidden states at layer exits of padded or packed rows (``modernbert.Layout``)."""

    @property
    def num_layers(self) -> int: ...

    def packed(
        self,
        lengths: Sequence[int],
        device: torch.device | str,
        width: int | None = None,
        uniform: bool = False,
    ) -> Layout: ...

    def padded(
        self,
        attention_mask: torch.Tensor | None,
        rows: int,
        width: int,
        device: torch.device | str,
    ) -> Layout: ...

    def masked(
        self, valid: torch.Tensor, rows: int, width: int, device: torch.device | str
    ) -> Layout: ...

    def encode(
        self,
        input_ids: torch.Tensor,
        layout: Layout,
        exits: Sequence[int] = (),
        normalize_exits: bool = False,
    ) -> dict[int, torch.Tensor]: ...


class TreeBackbone(Protocol):
    """A decoder that runs a shared-context request as one packed tree row (``tree.py``)."""

    def forward_tree(self, input_ids: torch.Tensor, tree: Tree) -> torch.Tensor: ...


class ForestBackbone(Protocol):
    """A decoder that runs prefixes and their blocks as padded rows (``forest.py``)."""

    def forward_forest(
        self,
        prefix_ids: torch.Tensor,
        prefix_mask: torch.Tensor,
        block_ids: torch.Tensor,
        block_mask: torch.Tensor,
        owner: torch.Tensor,
        shape: ForestShape,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class ComputedBuffers(Protocol):
    """A module whose buffers are computed, never loaded: ``computed_buffers`` rebuilds them after a meta build."""

    def computed_buffers(self) -> None: ...


def build(model_type: str, config: dict[str, Any]) -> nn.Module:
    if model_type not in ARCHITECTURES:
        raise ValueError(f"the native engine has no {model_type!r} backbone")
    return ARCHITECTURES[model_type](config)
