"""Explicit Qwen3.5 text-block LoRA target selection and adapter contract."""

from __future__ import annotations

import json
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any

LORA_FORMAT = "peft-lora/1"
FULL_ATTENTION = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)
LINEAR_ATTENTION = (
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_a",
    "linear_attn.out_proj",
)
MLP = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def select_target_modules(
    backbone: Any, is_linear: Callable[[Any], bool] | None = None
) -> list[str]:
    """Return exact modules for every hybrid decoder layer, refusing partial coverage."""
    if is_linear is None:
        from torch import nn

        is_linear = lambda module: isinstance(module, nn.Linear)
    modules = dict(backbone.named_modules())
    layer_types = getattr(getattr(backbone, "config", None), "layer_types", None)
    if not isinstance(layer_types, (list, tuple)) or not layer_types:
        raise ValueError("Qwen3.5 text backbone needs explicit layer_types")
    targets: list[str] = []
    for index, kind in enumerate(layer_types):
        if kind not in ("full_attention", "linear_attention"):
            raise ValueError(f"Unsupported Qwen3.5 layer type: {kind}")
        suffixes = (
            *MLP,
            *(FULL_ATTENTION if kind == "full_attention" else LINEAR_ATTENTION),
        )
        for suffix in suffixes:
            name = f"layers.{index}.{suffix}"
            if name not in modules or not is_linear(modules[name]):
                raise ValueError(f"Missing Qwen3.5 linear LoRA target: {name}")
            targets.append(name)
        opposite = LINEAR_ATTENTION if kind == "full_attention" else FULL_ATTENTION
        if any(f"layers.{index}.{suffix}" in modules for suffix in opposite):
            raise ValueError(
                f"Qwen3.5 layer {index} has incompatible attention branches"
            )
    layer_count = getattr(backbone.config, "num_hidden_layers", len(layer_types))
    if layer_count != len(layer_types):
        raise ValueError("Qwen3.5 layer_types and num_hidden_layers disagree")
    return targets


def lora_metadata(
    *,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: list[str],
    target_dimensions: dict[str, list[int]],
    source_kind: str,
    source_fingerprint: dict[str, Any],
    base_revision: str | None,
) -> dict[str, Any]:
    if rank < 1 or alpha < 1 or not 0 <= dropout < 1:
        raise ValueError("Invalid LoRA rank, alpha, or dropout")
    if source_kind not in ("base", "posttrained", "decision1", "decision2"):
        raise ValueError("Unsupported LoRA initialization source")
    if not target_modules or len(set(target_modules)) != len(target_modules):
        raise ValueError("LoRA target list must be nonempty and unique")
    if set(target_dimensions) != set(target_modules) or any(
        not isinstance(dims, list)
        or len(dims) != 2
        or any(type(value) is not int or value < 1 for value in dims)
        for dims in target_dimensions.values()
    ):
        raise ValueError("LoRA target dimensions must cover every source projection")
    if (
        not isinstance(source_fingerprint.get("files_sha256"), dict)
        or not source_fingerprint["files_sha256"]
    ):
        raise ValueError("LoRA needs source file hashes")
    return {
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "target_modules": target_modules,
        "target_dimensions": target_dimensions,
        "source_kind": source_kind,
        "source_fingerprint": source_fingerprint,
        "base_revision": base_revision,
    }


def _safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError("Adapter safetensors header is truncated")
        length = int.from_bytes(length_bytes, "little")
        if not 2 <= length <= 128 << 20:
            raise ValueError("Adapter safetensors header has an invalid length")
        header_bytes = stream.read(length)
        if len(header_bytes) != length:
            raise ValueError("Adapter safetensors header is truncated")
    header = json.loads(header_bytes)
    if not isinstance(header, dict):
        raise ValueError("Adapter safetensors header is invalid")
    return {name: value for name, value in header.items() if name != "__metadata__"}


def verify_adapter_config(
    adapter_path: Path, expected: dict[str, Any]
) -> dict[str, Any]:
    config = json.loads(
        (adapter_path / "adapter_config.json").read_text(encoding="utf-8")
    )
    if not isinstance(config, dict) or config.get("peft_type") != "LORA":
        raise ValueError("Checkpoint adapter is not PEFT LoRA")
    targets = expected["target_modules"]
    serialized_targets = config.get("target_modules")
    full_targets, suffix_targets = set(targets), {
        name.rsplit(".", 1)[-1] for name in targets
    }
    if (
        config.get("r") != expected["rank"]
        or config.get("lora_alpha") != expected["alpha"]
        or config.get("lora_dropout") != expected["dropout"]
        or config.get("bias") != "none"
        or not isinstance(serialized_targets, list)
        or set(serialized_targets) not in (full_targets, suffix_targets)
    ):
        raise ValueError(
            "Checkpoint adapter config differs from Decision 2.0 LoRA metadata"
        )
    weights = adapter_path / "adapter_model.safetensors"
    if not weights.is_file():
        raise ValueError("Checkpoint adapter has no safetensors weights")
    header = _safetensors_header(weights)
    dimensions = expected.get("target_dimensions")
    if not isinstance(dimensions, dict) or set(dimensions) != full_targets:
        raise ValueError("Checkpoint has no complete source projection dimensions")
    expected_shapes = {}
    for name, (in_features, out_features) in dimensions.items():
        stem = f"base_model.model.{name}"
        expected_shapes[f"{stem}.lora_A.weight"] = [expected["rank"], in_features]
        expected_shapes[f"{stem}.lora_B.weight"] = [out_features, expected["rank"]]
    if set(header) != set(expected_shapes):
        raise ValueError(
            "Checkpoint adapter tensor coverage differs from the exact source projections"
        )
    if any(
        not isinstance(header[name], dict)
        or header[name].get("shape") != shape
        or header[name].get("dtype") != "F32"
        for name, shape in expected_shapes.items()
    ):
        raise ValueError(
            "Checkpoint adapter tensor shapes/dtypes differ from source projections"
        )
    return config


def attach_lora(
    model: Any,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    source_kind: str,
    source_fingerprint: dict[str, Any],
) -> None:
    from peft import LoraConfig, get_peft_model

    targets = select_target_modules(model.backbone)
    modules = dict(model.backbone.named_modules())
    dimensions = {
        name: [modules[name].in_features, modules[name].out_features]
        for name in targets
    }
    contract = lora_metadata(
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=targets,
        target_dimensions=dimensions,
        source_kind=source_kind,
        source_fingerprint=source_fingerprint,
        base_revision=model.metadata.get("base_revision"),
    )
    contract["peft_version"] = version("peft")
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=targets,
        bias="none",
        task_type=None,
    )
    model.backbone = get_peft_model(model.backbone, config)
    model.head.requires_grad_(True)
    model.metadata["checkpoint_format"] = LORA_FORMAT
    model.metadata["lora"] = contract
    if not any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise RuntimeError(
            "PEFT did not expose any trainable backbone adapter parameters"
        )


def adapter_parameters(model: Any) -> list[Any]:
    parameters = [
        parameter
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("LoRA model has no trainable adapter parameters")
    return parameters
