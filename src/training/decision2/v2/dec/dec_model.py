"""Decoder-track readout extensions around the shared Decision 2.0 candidate model.

Each residual readout is multiplied by a zero-initialized gate, so a freshly
attached extension reproduces the source model's logits exactly until the
optimizer moves the gate. Checkpoints that carry a residual declare
``head_variant = "dec-residual"``; the shared loader rejects that variant, so a
residual checkpoint cannot be silently evaluated without its extra readout.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from training.model.data import canonical, file_sha256
from training.model.decision_model import ARCHITECTURE, CandidateHead, DecisionModel
from training.model.infer import checkpoint_fingerprint
from training.model.lora import (
    LORA_FORMAT,
    select_target_modules,
    verify_adapter_config,
)
from training.model.source import verify_source

DEC_RESIDUAL_VERSION = "dec-readout-residual/1"
DEC_HEAD_VARIANT = "dec-residual"
RESIDUAL_FILE = "dec_residual.safetensors"
RESIDUALS = ("ordinal_score", "layer_mix")
# Adam moves a scalar by roughly its learning rate per step; the fixed scale
# lets a zero gate reach O(1) within a few hundred updates.
GATE_SCALE = 10.0
SCORE_TYPE_ID = 2


class OrdinalScoreResidual(nn.Module):
    """Unimodal level prior for native Score rows, read from the global query.

    The query predicts a location on the ordered level axis and a precision;
    the residual adds ``-precision * (level - location)^2`` in level units to
    every offered Score level. Choice and Noul rows receive exactly zero.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.location = nn.Linear(hidden_size, 1)
        self.precision = nn.Linear(hidden_size, 1)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        query: torch.Tensor,
        candidate_mask: torch.Tensor,
        score_level_indices: torch.Tensor,
        task_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        with torch.autocast(device_type=query.device.type, enabled=False):
            normalized = self.norm(query.float())
            location = torch.sigmoid(self.location(normalized))
            precision = F.softplus(self.precision(normalized))
            span = (candidate_mask.sum(-1, keepdim=True).float() - 1).clamp_min(1.0)
            position = score_level_indices.float() / span
            residual = -precision * ((position - location) * span).square()
            is_score = (task_type_ids == SCORE_TYPE_ID)[:, None].float()
            return GATE_SCALE * self.gate * residual * is_score


class LayerMixResidual(nn.Module):
    """Second candidate readout over a learned mixture of all decoder layers."""

    def __init__(self, hidden_size: int, layer_count: int, head_dim: int = 256):
        super().__init__()
        self.layer_count = layer_count
        self.mix = nn.Parameter(torch.zeros(layer_count))
        self.head = CandidateHead(hidden_size, head_dim)
        self.gate = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _rms(value: torch.Tensor) -> torch.Tensor:
        return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)

    def forward(self, candidates: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        if (
            candidates.shape[0] != self.layer_count
            or query.shape[0] != self.layer_count
        ):
            raise ValueError("Layer-mix readout received a different layer count")
        with torch.autocast(device_type=query.device.type, enabled=False):
            weights = torch.softmax(self.mix.float(), dim=0)
            mixed_candidates = torch.einsum(
                "l,lbwh->bwh", weights, self._rms(candidates.float())
            )
            mixed_query = torch.einsum("l,lbh->bh", weights, self._rms(query.float()))
            return GATE_SCALE * self.gate * self.head(mixed_candidates, mixed_query)


class DecModel(DecisionModel):
    """Shared candidate model plus optional zero-gated residual readouts."""

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        metadata: dict[str, Any],
        *,
        ordinal_score: OrdinalScoreResidual | None = None,
        layer_mix: LayerMixResidual | None = None,
    ):
        super().__init__(backbone, head, metadata)
        self.ordinal_score = ordinal_score
        self.layer_mix = layer_mix

    @classmethod
    def wrap(cls, model: DecisionModel, residuals: tuple[str, ...] = ()) -> DecModel:
        unknown = set(residuals) - set(RESIDUALS)
        if unknown or len(set(residuals)) != len(residuals):
            raise ValueError(f"Unknown or repeated residual readouts: {residuals}")
        if model.metadata.get("head_variant", "shared") != "shared":
            raise ValueError("Residual readouts extend only the shared candidate head")
        if model.metadata.get("architecture") != ARCHITECTURE:
            raise ValueError(
                "Residual readouts require the Qwen3.5 shared architecture"
            )
        metadata = dict(model.metadata)
        backbone = model.backbone
        if hasattr(backbone, "get_base_model"):
            backbone = backbone.get_base_model()
        config = backbone.config
        ordinal = layer_mix = None
        if residuals:
            layer_count = config.num_hidden_layers + 1
            metadata["head_variant"] = DEC_HEAD_VARIANT
            metadata["dec_residual"] = {
                "version": DEC_RESIDUAL_VERSION,
                "residuals": sorted(residuals),
                "gate_scale": GATE_SCALE,
                "layer_count": layer_count,
                "head_dim": metadata["head_dim"],
            }
            if "ordinal_score" in residuals:
                ordinal = OrdinalScoreResidual(config.hidden_size)
            if "layer_mix" in residuals:
                layer_mix = LayerMixResidual(
                    config.hidden_size, layer_count, metadata["head_dim"]
                )
        return cls(
            model.backbone,
            model.head,
            metadata,
            ordinal_score=ordinal,
            layer_mix=layer_mix,
        )

    def residual_parameters(self) -> list[nn.Parameter]:
        modules = [m for m in (self.ordinal_score, self.layer_mix) if m is not None]
        return [p for module in modules for p in module.parameters()]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        candidate_positions: torch.Tensor,
        candidate_mask: torch.Tensor,
        query_positions: torch.Tensor,
        task_type_ids: torch.Tensor | None = None,
        score_level_indices: torch.Tensor | None = None,
        **unused: Any,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=self.layer_mix is not None,
        )
        hidden = outputs.last_hidden_state
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        candidates = hidden[batch[:, None], candidate_positions]
        query = hidden[batch, query_positions]
        scores = self.head(candidates, query)
        if self.ordinal_score is not None:
            if task_type_ids is None or score_level_indices is None:
                raise ValueError("Ordinal Score residual needs type and level IDs")
            scores = scores + self.ordinal_score(
                query, candidate_mask, score_level_indices, task_type_ids
            )
        if self.layer_mix is not None:
            layers = outputs.hidden_states
            scores = scores + self.layer_mix(
                torch.stack([h[batch[:, None], candidate_positions] for h in layers]),
                torch.stack([h[batch, query_positions] for h in layers]),
            )
        return scores.masked_fill(~candidate_mask, -float("inf"))

    def residual_state(self) -> dict[str, torch.Tensor]:
        tensors: dict[str, torch.Tensor] = {}
        for name in RESIDUALS:
            module = getattr(self, name)
            if module is not None:
                tensors.update(
                    {
                        f"{name}.{key}": value.detach().float().cpu().contiguous()
                        for key, value in module.state_dict().items()
                    }
                )
        return tensors

    def save(self, path: str | Path, tokenizer: Any) -> None:
        from safetensors.torch import save_file

        super().save(path, tokenizer)
        tensors = self.residual_state()
        if tensors:
            save_file(tensors, str(Path(path) / RESIDUAL_FILE))


def load_dec_checkpoint(
    path: str | Path, source_path: str | Path | None, *, trainable_adapter: bool = False
) -> tuple[DecModel, Any]:
    """Load a decoder-track checkpoint, including any residual readout."""
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    path = Path(path)
    metadata = json.loads((path / "decision_config.json").read_text(encoding="utf-8"))
    residual = metadata.get("dec_residual")
    if residual is None:
        model, tokenizer = DecisionModel.from_checkpoint(
            path, source_path=source_path, trainable_adapter=trainable_adapter
        )
        return DecModel.wrap(model), tokenizer
    if (
        residual.get("version") != DEC_RESIDUAL_VERSION
        or metadata.get("head_variant") != DEC_HEAD_VARIANT
        or metadata.get("architecture") != ARCHITECTURE
        or residual.get("gate_scale") != GATE_SCALE
    ):
        raise ValueError("Unsupported decoder-track residual checkpoint")
    contract = metadata.get("lora")
    if metadata.get("checkpoint_format") != LORA_FORMAT or not isinstance(
        contract, dict
    ):
        raise ValueError("Residual checkpoints are LoRA continuations of a 1.0 source")
    if contract.get("source_kind") != "decision1" or source_path is None:
        raise ValueError("Residual checkpoint needs its Decision 1.0 --source-path")
    verify_source(Path(source_path), contract.get("source_fingerprint"))
    verify_adapter_config(path / "adapter", contract)
    source_model, _ = DecisionModel.from_decision1(source_path, metadata["head_dim"])
    if contract.get("target_modules") != select_target_modules(source_model.backbone):
        raise ValueError("Adapter target modules differ from the pinned source")
    modules = dict(source_model.backbone.named_modules())
    dimensions = {
        name: [modules[name].in_features, modules[name].out_features]
        for name in contract["target_modules"]
    }
    if contract.get("target_dimensions") != dimensions:
        raise ValueError("Adapter projection dimensions differ from the pinned source")
    from peft import PeftModel

    source_model.backbone = PeftModel.from_pretrained(
        source_model.backbone,
        path / "adapter",
        is_trainable=trainable_adapter,
        local_files_only=True,
    )
    source_model.head.load_state_dict(
        load_file(str(path / "decision_head.safetensors")), strict=True
    )
    source_model.metadata = {
        key: value
        for key, value in metadata.items()
        if key not in ("dec_residual", "head_variant")
    }
    model = DecModel.wrap(source_model, tuple(residual["residuals"]))
    model.metadata = metadata
    state = load_file(str(path / RESIDUAL_FILE))
    for name in RESIDUALS:
        module = getattr(model, name)
        subset = {
            key.removeprefix(f"{name}."): value
            for key, value in state.items()
            if key.startswith(f"{name}.")
        }
        if module is None:
            if subset:
                raise ValueError(f"Undeclared residual tensors for {name}")
            continue
        module.load_state_dict(subset, strict=True)
    return model, AutoTokenizer.from_pretrained(path, local_files_only=True)


def dec_fingerprint(path: Path, source_path: Path | None) -> dict[str, Any]:
    """Shared inference identity plus any residual readout weights."""
    identity = checkpoint_fingerprint(path, source_path)
    residual = path / RESIDUAL_FILE
    if not residual.is_file():
        return identity
    files = dict(identity["files_sha256"])
    prefix = "checkpoint/" if any(k.startswith("checkpoint/") for k in files) else ""
    files[f"{prefix}{RESIDUAL_FILE}"] = file_sha256(residual)
    return {
        "model_sha256": hashlib.sha256(canonical(files).encode("utf-8")).hexdigest(),
        "files_sha256": files,
    }
