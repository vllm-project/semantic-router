"""The native PyTorch engine: builds a backbone from a ``ModelSpec`` and runs it on one device."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

import torch
from torch import nn

from ...plugins.base import (
    Accelerator,
    DeviceInfo,
    Engine,
    EngineModel,
    EngineOptions,
    ForwardBatch,
    ForwardOutput,
    ModelSpec,
)
from ...scheduler.planner import padded
from . import fast, models
from .models.lora import attach
from .models.tree import Tree
from .weights import keep_linear_bf16, load_adapter, load_backbone


class NativeEngineModel(EngineModel):
    def __init__(
        self,
        backbone: nn.Module,
        accelerator: Accelerator,
        device_info: DeviceInfo,
        spec: ModelSpec,
        options: EngineOptions,
        residency: dict[str, int] | None,
    ):
        self.backbone = backbone
        self.accelerator = accelerator
        self.device_info = device_info
        self.device = accelerator.torch_device(device_info)
        self.spec = spec
        self.options = options
        self.residency = residency
        self.kernels = accelerator.kernels(device_info)
        self.kernels.allow_approximate = not options.exact_kernels_only
        backbone.kernels = self.kernels
        self.fast: dict[str, Any] = {}
        self.masks: fast.Masks | None = None
        self.graphs: fast.Graphs | None = None
        if self.device.type == "cuda":
            self._install_fast()

    def _install_fast(self) -> None:
        """The exact GPU fast path (``fast.py``): fused layers, lean LoRA, host masks and graphs."""
        options, backbone = self.options, self.backbone
        if options.fused_kernels:
            reason = fast.fused_unavailable(backbone, self.kernels)
            self.fast["fused_layers"] = 0 if reason else fast.install_fused(backbone)
            if reason:
                self.fast["fused_skipped"] = reason
        if self.spec.backbone.lora is not None and self.residency is not None:
            self.fast["lora"] = fast.install_lean_lora(backbone)
        if options.fused_kernels or options.graphs:
            self.masks = fast.Masks()
        if options.graphs:
            self.graphs = fast.Graphs(backbone, self.masks)

    def receipt(self) -> dict[str, Any]:
        """What runs this model: kernels, fast-path pieces and graph statistics."""
        out: dict[str, Any] = {"kernels": self.kernels.describe(), **self.fast}
        if self.graphs is not None:
            out["graphs"] = self.graphs.receipt()
        return out

    def autocast(self) -> AbstractContextManager[Any]:
        if self.device.type == "cpu":
            from contextlib import nullcontext

            return nullcontext()
        return self.accelerator.autocast(self.device_info, self.spec.dtype.autocast)

    supports_shared_context = True

    def _forward_tree(self, batch: ForwardBatch) -> ForwardOutput:
        """The batch as one shared-context tree: its first ``shared_prefix`` tokens computed once."""
        prefix, lengths = batch.shared_prefix, batch.lengths
        suffix = [length - prefix for length in lengths]
        width = padded(max(suffix))
        ids = batch.input_ids
        packed = torch.cat(
            [ids[0, :prefix]]
            + [ids[row, prefix:length] for row, length in enumerate(lengths)]
        )[None].to(self.device)
        tree = Tree(
            prefix,
            suffix,
            width,
            self.device,
            padded_exact=any(length != padded(max(lengths)) for length in lengths),
        )
        gather = (batch.gather - prefix).clamp(min=0).to(self.device)
        query = (batch.query - prefix).to(self.device)
        with torch.inference_mode(), self.autocast():
            hidden = self.backbone.forward_tree(packed, tree)
            rows_hidden = tree.rows(hidden[0, prefix:])
            rows = torch.arange(rows_hidden.shape[0], device=rows_hidden.device)
            gathered = rows_hidden[rows[:, None], gather]
            queried = rows_hidden[rows, query]
        return ForwardOutput(gathered=gathered, query=queried)

    def forward(self, batch: ForwardBatch) -> ForwardOutput:
        if batch.shared_prefix:
            return self._forward_tree(batch)
        input_ids = batch.input_ids.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)
        gather = batch.gather.to(self.device)
        query = batch.query.to(self.device)
        with torch.inference_mode(), self.autocast():
            if self.graphs is not None:
                hidden = self.graphs(input_ids, attention_mask, batch.lengths)
            elif self.masks is not None:
                padded = any(n != input_ids.shape[1] for n in batch.lengths)
                hidden = self.backbone(
                    input_ids,
                    attention_mask,
                    masks=self.masks.build(attention_mask, padded),
                )
            else:
                hidden = self.backbone(input_ids, attention_mask)
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            gathered = hidden[rows[:, None], gather]
            queried = hidden[rows, query]
        return ForwardOutput(gathered=gathered, query=queried)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.backbone.parameters())

    def memory_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.backbone.parameters())

    def close(self) -> None:
        self.backbone = None  # type: ignore[assignment]
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


class NativeEngine(Engine):
    name = "native"

    @classmethod
    def descriptor(cls) -> dict[str, Any]:
        return {
            "architectures": sorted(models.ARCHITECTURES),
            "outputs": ["gathered"],
            "shared_context": True,
            "lora": "peft-unmerged",
        }

    def supports(self, spec: ModelSpec, device: DeviceInfo) -> str | None:
        if spec.backbone.model_type not in models.ARCHITECTURES:
            return f"no native {spec.backbone.model_type!r} backbone"
        if (
            device.accelerator != "cpu"
            and not device.bf16
            and spec.dtype.autocast == "bfloat16"
        ):
            return f"{device.label} has no BF16 support"
        return None

    def load(
        self,
        spec: ModelSpec,
        accelerator: Accelerator,
        device: DeviceInfo,
        options: EngineOptions,
    ) -> NativeEngineModel:
        if options.threads:
            torch.set_num_threads(options.threads)
        backbone_spec = spec.backbone
        with torch.device("meta"):
            backbone = models.build(backbone_spec.model_type, backbone_spec.config)
        # Rotary buffers are computed, not loaded: build them on a real device.
        backbone.rotary_emb = type(backbone.rotary_emb)(backbone_spec.config)
        load_backbone(backbone, backbone_spec.weight_files, backbone_spec.weight_prefix)
        lora = backbone_spec.lora
        if lora is not None:
            if options.merge_lora:
                raise ValueError(
                    "merged LoRA changes answers; it belongs to the max_speed profile"
                )
            with torch.device("meta"):
                attach(
                    backbone,
                    list(lora.target_modules),
                    lora.rank,
                    lora.alpha / lora.rank,
                )
            load_adapter(backbone, lora.weight_files)
        leftovers = [
            name for name, parameter in backbone.named_parameters() if parameter.is_meta
        ]
        if leftovers:
            raise ValueError(f"backbone parameters were not loaded: {leftovers[:3]}")
        backbone = backbone.float()
        target = accelerator.torch_device(device)
        residency = None
        if target.type != "cpu" and spec.dtype.bf16_resident:
            residency = keep_linear_bf16(backbone)
        backbone = backbone.to(target).eval()
        return NativeEngineModel(
            backbone, accelerator, device, spec, options, residency
        )
