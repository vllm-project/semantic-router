"""Explicit adapter or full-encoder optimization and checkpoint ownership."""

# ruff: noqa: PLC0415

import hashlib
import json
import math
from pathlib import Path

from .model import load_model, save_adapter
from .task_head import task_head_scope, tensor_digest


def configure_trainable_tail(model, last_layers):
    """Restrict a complete ModernBERT checkpoint to its tail and entire task head."""
    if (
        type(model).__name__ != "ModernBertForSequenceClassification"
        or hasattr(model, "peft_config")
        or hasattr(model, "get_base_model")
    ):
        raise ValueError("Tail training requires a complete ModernBERT classifier")
    layers = model.model.layers
    if type(last_layers) is not int or not 1 <= last_layers <= len(layers):
        raise ValueError("Trainable tail must contain one through all encoder layers")
    start = len(layers) - last_layers
    modules = [*layers[start:], model.head, model.classifier]
    selected = {
        id(parameter) for module in modules for parameter in module.parameters()
    }
    named = dict(model.named_parameters())
    if not selected or not selected <= {id(parameter) for parameter in named.values()}:
        raise ValueError("Trainable modules do not belong to the classifier")
    if any(
        id(parameter) in selected for parameter in model.model.final_norm.parameters()
    ):
        raise ValueError("Encoder final normalization must remain frozen")
    model.requires_grad_(False)
    for module in modules:
        module.requires_grad_(True)
    tensors = {
        name: {"shape": list(parameter.shape), "numel": parameter.numel()}
        for name, parameter in named.items()
        if parameter.requires_grad
    }
    return {
        "architecture": type(model).__name__,
        "layer_indices": list(range(start, len(layers))),
        "trainable_tensors": tensors,
        "trainable_parameters": sum(item["numel"] for item in tensors.values()),
        "encoder_final_norm_trainable": False,
    }


def frozen_parameter_receipt(model):
    """Hash frozen parameters once so an optional restricted update is auditable."""
    return {
        name: {
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "sha256": tensor_digest(parameter),
        }
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }


def assert_frozen_parameters_preserved(expected, model):
    if frozen_parameter_receipt(model) != expected:
        raise ValueError("Frozen parameter scope or tensor values changed")


def validate_method(method, adapter, fresh_head):
    if method not in {"lora", "full"}:
        raise ValueError("Training method must be lora or full")
    if method == "lora" and (adapter is None or fresh_head):
        raise ValueError("LoRA continuation requires its initialized adapter")
    if method == "full" and adapter is not None:
        raise ValueError(
            "Full training starts from a complete checkpoint, not an adapter"
        )


def load_trainable_model(base, contract, adapter, method, fresh_head=False):
    validate_method(method, adapter, fresh_head)
    model, tokenizer, labels, ids = load_model(
        base, contract, adapter, trainable=method == "lora"
    )
    if method == "full":
        if type(model).__name__ != "ModernBertForSequenceClassification":
            raise ValueError("Full sequence training requires a ModernBERT classifier")
        model.requires_grad_(True)
        if fresh_head:
            from torch import nn

            head_parameter = next(model.head.parameters())
            model.head = type(model.head)(model.config).to(
                device=head_parameter.device, dtype=head_parameter.dtype
            )
            classifier = model.classifier
            model.classifier = nn.Linear(
                classifier.in_features,
                classifier.out_features,
                bias=classifier.bias is not None,
                device=classifier.weight.device,
                dtype=classifier.weight.dtype,
            )
            # Fresh modules let HF apply its native initialization without
            # mutating the initialization guards on loaded checkpoint tensors.
            model.head.apply(model._init_weights)
            # ModernBERT initializes the final classifier on its owning task
            # module, not when visiting the bare Linear child.
            model._init_weights(model)
    return model, tokenizer, labels, ids


def optimizer_groups(model, learning_rate, head_learning_rate=None):
    rates = [learning_rate]
    if head_learning_rate is not None:
        rates.append(head_learning_rate)
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("Learning rates must be finite and positive")
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    head_ids = {
        id(parameter)
        for module in (base.head, base.classifier)
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    encoder, head = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (head if id(parameter) in head_ids else encoder).append(parameter)
    groups = []
    for name, parameters, rate in (
        ("encoder", encoder, learning_rate),
        ("task_head", head, head_learning_rate or learning_rate),
    ):
        if parameters:
            groups.append(
                {"name": name, "params": parameters, "lr": rate, "initial_lr": rate}
            )
    if not groups:
        raise ValueError("No trainable parameters")
    return groups


def initial_artifact_receipt(base, adapter):
    """Hash the actual local initialization, including every weight shard."""
    root = Path(base)
    if not root.is_dir():
        raise ValueError("Training requires a downloaded local base checkpoint")
    names = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    for path in root.iterdir():
        if (
            path.suffix == ".safetensors"
            or path.name
            in {
                "model.safetensors.index.json",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
            }
            or path.name.startswith("pytorch_model-")
        ):
            names.add(path.name)
    files = {}
    for name in sorted(names):
        path = root / name
        if path.is_file():
            with path.open("rb") as stream:
                files[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {"base_files": files}
    if adapter is not None:
        for filename in ("adapter_model.safetensors", "adapter_config.json"):
            path = Path(adapter) / filename
            if filename == "adapter_config.json" and not path.is_file():
                result[filename] = None
                continue
            with path.open("rb") as stream:
                result[filename] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def verify_full_checkpoint(model, checkpoint):
    """Require every tensor of a complete sequence checkpoint to be preserved."""
    if type(model).__name__ != "ModernBertForSequenceClassification":
        raise ValueError("Full checkpoints must own the actual encoder weights")
    from safetensors.torch import load_file

    saved = load_file(str(Path(checkpoint) / "model.safetensors"), device="cpu")
    state = model.state_dict()
    if saved.keys() != state.keys():
        raise ValueError("Full checkpoint omitted or introduced model tensors")
    import torch

    for name, value in state.items():
        if saved[name].dtype != value.dtype or not torch.equal(
            saved[name], value.detach().cpu()
        ):
            raise ValueError(f"Full checkpoint changed tensor: {name}")


def save_training_checkpoint(model, tokenizer, destination, method, base_id, revision):
    if method == "lora":
        save_adapter(model, tokenizer, destination, base_id, revision)
        return
    if method != "full" or hasattr(model, "peft_config"):
        raise ValueError("Full checkpoints must own the actual encoder weights")
    destination.mkdir(parents=True, exist_ok=True)
    expected = task_head_scope(model)
    model.save_pretrained(destination, safe_serialization=True)
    verify_full_checkpoint(model, destination)
    tokenizer.save_pretrained(destination)
    (destination / "task-head.json").write_text(json.dumps(expected, indent=2) + "\n")
    (destination / "training-origin.json").write_text(
        json.dumps(
            {"method": method, "base_model": base_id, "base_revision": revision},
            indent=2,
        )
        + "\n"
    )
