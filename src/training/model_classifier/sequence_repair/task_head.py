"""Inspect the effective task head and verify its adapter/merged storage."""

# ruff: noqa: PLC0415

import hashlib


def _modules(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if type(base).__name__ not in {
        "ModernBertForSequenceClassification",
        "ModernBertForTokenClassification",
    }:
        raise ValueError(
            "Task-head auditing requires a supported ModernBERT task model"
        )
    for name in ("head", "classifier"):
        module = getattr(base, name)
        owned = hasattr(module, "modules_to_save")
        if owned:
            active = module.active_adapter
            if isinstance(active, list):
                if len(active) != 1:
                    raise ValueError("Task-head auditing requires one active adapter")
                active = active[0]
            module = module.modules_to_save[active]
        yield name, module, owned


def tensor_digest(tensor):
    import torch

    value = tensor.detach().cpu().contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Task head contains non-finite tensors")
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def task_head_scope(model):
    """Describe actual effective tensors, including deliberately inherited heads."""
    tensors = {}
    for name, module, owned in _modules(model):
        trainable = dict(module.named_parameters())
        for key, tensor in module.state_dict().items():
            parameter = trainable.get(key)
            tensors[f"{name}.{key}"] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": tensor.numel(),
                "trainable": parameter is not None and parameter.requires_grad,
                "adapter_owned": owned,
                "sha256": tensor_digest(tensor),
            }
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return {
        "architecture": type(base).__name__,
        "modules_to_save": (
            model.peft_config["default"].modules_to_save
            if hasattr(model, "get_base_model")
            else None
        ),
        "trainable_task_head_parameters": sum(
            item["numel"] for item in tensors.values() if item["trainable"]
        ),
        "tensors": tensors,
    }


def verify_saved_task_head(model, adapter_path):
    """Reject an adapter that omits or changes an effective trainable task tensor."""
    from safetensors.torch import load_file

    saved = load_file(str(adapter_path), device="cpu")
    scope = task_head_scope(model)
    for name, item in scope["tensors"].items():
        key = f"base_model.model.{name}"
        value = saved.get(key)
        item["saved_key"] = key if value is not None else None
        if value is None:
            if item["trainable"] or item["adapter_owned"]:
                raise ValueError(f"Adapter omitted effective task tensor: {name}")
            continue
        if (
            list(value.shape) != item["shape"]
            or str(value.dtype) != item["dtype"]
            or tensor_digest(value) != item["sha256"]
        ):
            raise ValueError(f"Adapter changed effective task tensor: {name}")
    return scope


def assert_task_head_preserved(expected, model):
    """Compare tensor identity across PEFT merge and serialized model reload."""
    observed = task_head_scope(model)
    identity = ("shape", "dtype", "sha256")
    expected_tensors = expected["tensors"]
    if expected_tensors.keys() != observed["tensors"].keys():
        raise ValueError("Task-head tensor names changed")
    for name, item in expected_tensors.items():
        if any(item[key] != observed["tensors"][name][key] for key in identity):
            raise ValueError(f"Task-head tensor changed: {name}")
    return observed
