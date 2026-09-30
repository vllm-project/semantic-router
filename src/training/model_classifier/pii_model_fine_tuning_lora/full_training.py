# Optional tensor dependencies stay lazy for the data-only PII tooling.
# ruff: noqa: PLC0415
"""Explicit full ModernBERT token-classifier training and native storage."""

import hashlib
import json
from pathlib import Path

TOKEN_ARCHITECTURE = "ModernBertForTokenClassification"
HEAD_PREFIXES = ("head.", "classifier.")


def validate_method(method, adapter, fresh_head):
    if method == "lora":
        if not adapter or fresh_head:
            raise ValueError("LoRA continuation requires an adapter and keeps its head")
    elif method == "full":
        if adapter:
            raise ValueError("Full training cannot load an adapter")
    else:
        raise ValueError(f"Unknown training method: {method}")


def load_token_model(
    base, label_to_id, id_to_label, *, fresh_head=False, adapter=False
):
    """Reject random encoder repair; fresh mode resets the entire token head."""
    import torch
    from transformers import AutoConfig, AutoModelForTokenClassification

    config = AutoConfig.from_pretrained(base)
    if config.model_type != "modernbert":
        raise ValueError("PII full checkpoints require ModernBERT")
    if not fresh_head and not adapter:
        if config.architectures != [TOKEN_ARCHITECTURE]:
            raise ValueError(
                "A native PII checkpoint must declare the token architecture"
            )
        if config.label2id != label_to_id or config.id2label != id_to_label:
            raise ValueError(
                "Native PII label order differs from the supplied contract"
            )
    config.num_labels = len(label_to_id)
    config.label2id = label_to_id
    config.id2label = id_to_label
    # TF5 removed this constructor keyword; TF4 still exposes a config option.
    if hasattr(config, "reference_compile"):
        config.reference_compile = False
    model, loading = AutoModelForTokenClassification.from_pretrained(
        base,
        config=config,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
        output_loading_info=True,
        ignore_mismatched_sizes=fresh_head,
    )
    if type(model).__name__ != TOKEN_ARCHITECTURE:
        raise ValueError("PII must use the native token classifier")
    missing = set(loading.get("missing_keys", []))
    mismatched = {
        item[0] if isinstance(item, (list, tuple)) else item
        for item in loading.get("mismatched_keys", [])
    }
    allowed = {name for name in model.state_dict() if name.startswith(HEAD_PREFIXES)}
    unexpected = set(loading.get("unexpected_keys", []))
    if unexpected and not (fresh_head or adapter):
        raise ValueError("Unexpected tensors in the native PII checkpoint")
    if fresh_head and any(not name.startswith("decoder.") for name in unexpected):
        raise ValueError("Unexpected non-MLM tensors in the initialization artifact")
    if loading.get("error_msgs") or (missing | mismatched) - allowed:
        raise ValueError(
            "Missing or incompatible encoder tensors in PII initialization"
        )
    if (missing or mismatched) and not (fresh_head or adapter):
        raise ValueError("Native PII checkpoint is missing its complete token head")
    if fresh_head:
        # ModernBERT initializes dense/norm on their owners, and classifier on
        # the owning task module. New modules also avoid TF5's per-parameter
        # initialized guards without changing private flags or encoder values.
        model.head = type(model.head)(model.config)
        model.classifier = torch.nn.Linear(
            model.classifier.in_features,
            model.classifier.out_features,
            bias=model.classifier.bias is not None,
        )
        model.head.apply(model._init_weights)
        model._init_weights(model)
    if not all(
        bool(torch.isfinite(value).all()) for value in model.state_dict().values()
    ):
        raise ValueError("Non-finite PII checkpoint tensors")
    return model


def optimizer_groups(model, encoder_lr, head_lr):
    if type(model).__name__ != TOKEN_ARCHITECTURE:
        raise ValueError("Full PII optimization requires a native token classifier")
    if encoder_lr <= 0 or head_lr <= 0:
        raise ValueError("Learning rates must be positive")
    groups = {"encoder": [], "task_head": []}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(True)
        groups["task_head" if name.startswith(HEAD_PREFIXES) else "encoder"].append(
            parameter
        )
    if not all(groups.values()):
        raise ValueError("Both encoder and complete token head must be trainable")
    return [
        {"name": name, "params": parameters, "lr": lr, "initial_lr": lr}
        for (name, parameters), lr in zip(
            groups.items(), (encoder_lr, head_lr), strict=True
        )
    ]


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def initial_artifact_receipt(base):
    path = Path(base)
    if not path.is_dir():
        raise ValueError("Full training requires a local frozen base directory")
    names = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    }
    files = {
        item.name: file_sha256(item)
        for item in sorted(path.iterdir())
        if item.is_file()
        and (
            item.name in names
            or item.suffix == ".safetensors"
            or item.name.endswith(".index.json")
        )
    }
    if "config.json" not in files or not any(
        name.endswith(".safetensors") for name in files
    ):
        raise ValueError("Full training needs config and safetensors base weights")
    return {"directory": str(path.resolve()), "files": files}


def tensor_receipt(model):
    """Hash actual values, including the complete prediction head and encoder."""
    import torch

    result = {}
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        result[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(
                tensor.view(-1).view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        }
    return result


def verify_full_checkpoint(model, directory):
    """The saved state must contain every tensor with identical dtype and values."""
    import torch
    from safetensors.torch import load_file

    if type(model).__name__ != TOKEN_ARCHITECTURE:
        raise ValueError("Expected a complete native token classifier")
    directory = Path(directory)
    index = directory / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        shards = sorted(set(weight_map.values()))
        if any(Path(name).name != name for name in shards):
            raise ValueError("Invalid checkpoint shard path")
    else:
        shards = ["model.safetensors"]
        weight_map = None
    actual = {}
    for name in shards:
        values = load_file(str(directory / name))
        if set(actual) & set(values):
            raise ValueError("Duplicate checkpoint tensor")
        if weight_map is not None and any(
            weight_map.get(key) != name for key in values
        ):
            raise ValueError("Checkpoint shard index differs from actual tensors")
        actual.update(values)
    expected = model.state_dict()
    if set(actual) != set(expected) or (
        weight_map is not None and set(weight_map) != set(expected)
    ):
        raise ValueError("Full checkpoint tensor keys differ from the complete model")
    for name, value in expected.items():
        stored = actual[name]
        if (
            stored.shape != value.shape
            or stored.dtype != value.dtype
            or not bool(torch.isfinite(stored).all())
            or not torch.equal(stored, value.detach().cpu())
        ):
            raise ValueError(f"Full checkpoint tensor mismatch: {name}")
    return {
        "files": {name: file_sha256(directory / name) for name in shards},
        "tensors": tensor_receipt(model),
    }


def save_full_checkpoint(model, tokenizer, directory, origin):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory)
    receipt = verify_full_checkpoint(model, directory)
    receipt["config_sha256"] = file_sha256(directory / "config.json")
    receipt["tokenizer_files"] = {
        path.name: file_sha256(path)
        for path in sorted(directory.iterdir())
        if path.name
        in {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
    }
    receipt["method"] = "full"
    receipt["task"] = "token-classification"
    receipt["label2id"] = model.config.label2id
    receipt["origin"] = origin
    (directory / "full-checkpoint.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt
