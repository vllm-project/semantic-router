"""Load safetensors checkpoints into native modules by exact parameter name.

Every tensor is upcast to FP32, as the released runtime loads with
``dtype=torch.float32``. Persistent buffers (BatchNorm statistics) load like
parameters. A missing or unexpected backbone tensor is an error; tensors
outside ``prefix`` (for example a vision tower) are ignored, and so are
unknown tensors under it unless ``strict`` asks otherwise. A module's
``ignored_tensors`` names checkpoint tensors it recomputes or never reads.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path

import torch
from torch import nn

from .models.lora import adapter_key


def iter_tensors(
    files: Iterable[Path], prefix: str = ""
) -> Iterator[tuple[str, torch.Tensor]]:
    from safetensors import safe_open

    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            names = handle.keys()
            for name in names:
                if prefix and not name.startswith(prefix):
                    continue
                yield name[len(prefix) :], handle.get_tensor(name)


def set_parameter(
    root: nn.Module, name: str, tensor: torch.Tensor, dtype: torch.dtype = torch.float32
) -> None:
    """Replace a (possibly meta) parameter with a loaded tensor in the process's own memory, FP32 by default.

    ``get_tensor`` returns a view of the checkpoint's file mapping, which an
    FP32 tensor would keep: GEMMs then read file-backed pages, a short CPU
    forward 8-11% slower than on a private copy. ``dtype`` is the dtype the
    engine holds the parameter in; reading a checkpoint straight into it gives
    the values an FP32 read and a cast would.
    """
    owner_name, _, leaf = name.rpartition(".")
    owner = root.get_submodule(owner_name) if owner_name else root
    owner._parameters[leaf] = nn.Parameter(
        tensor.to(dtype, copy=True).contiguous(), requires_grad=False
    )


def set_buffer(root: nn.Module, name: str, tensor: torch.Tensor) -> None:
    """Replace a (possibly meta) persistent buffer with a loaded tensor, FP32 when floating."""
    owner_name, _, leaf = name.rpartition(".")
    owner = root.get_submodule(owner_name) if owner_name else root
    dtype = torch.float32 if tensor.is_floating_point() else tensor.dtype
    owner._buffers[leaf] = tensor.to(dtype, copy=True).contiguous()


def persistent_buffers(module: nn.Module) -> dict[str, torch.Tensor]:
    """The buffers ``module`` saves in its state dict, by full name."""
    return {
        f"{prefix}.{leaf}" if prefix else leaf: buffer
        for prefix, owner in module.named_modules()
        for leaf, buffer in owner._buffers.items()
        if buffer is not None and leaf not in owner._non_persistent_buffers_set
    }


def _renamed(name: str, renames: Mapping[str, str]) -> str:
    for old, new in renames.items():
        if name.startswith(old):
            return new + name[len(old) :]
    return name


def load_backbone(
    module: nn.Module,
    files: Iterable[Path],
    prefix: str = "",
    renames: Mapping[str, str] | None = None,
    *,
    strict: bool = False,
    dtype: torch.dtype = torch.float32,
) -> None:
    """Load every parameter (in ``dtype``) and persistent buffer of ``module``; ``renames`` maps checkpoint name prefixes to module ones."""
    parameters = dict(module.named_parameters())
    buffers = persistent_buffers(module)
    expected = set(parameters) | set(buffers)
    ignored = tuple(getattr(module, "ignored_tensors", ()))
    seen: set[str] = set()
    for checkpoint_name, tensor in iter_tensors(files, prefix):
        name = _renamed(checkpoint_name, renames or {})
        target = parameters.get(name, buffers.get(name))
        if target is None:
            if name.rpartition(".")[2] in ignored or (prefix and not strict):
                continue
            raise ValueError(f"checkpoint tensor {name!r} has no backbone parameter")
        if tuple(tensor.shape) != tuple(target.shape):
            raise ValueError(
                f"{name}: checkpoint shape {tuple(tensor.shape)} != {tuple(target.shape)}"
            )
        if name in parameters:
            set_parameter(module, name, tensor, dtype)
        else:
            set_buffer(module, name, tensor)
        seen.add(name)
    missing = sorted(expected - seen)
    if missing:
        raise ValueError(
            f"checkpoint misses {len(missing)} backbone tensors, e.g. {missing[:3]}"
        )


def load_adapter(module: nn.Module, files: Iterable[Path]) -> int:
    """Load PEFT LoRA factors into attached ``LoRALinear`` layers; returns the tensor count."""
    modules = dict(module.named_modules())
    loaded = 0
    expected = {
        f"{name}.{factor}"
        for name, layer in modules.items()
        if hasattr(layer, "lora_A")
        for factor in ("lora_A", "lora_B")
    }
    seen: set[str] = set()
    for name, tensor in iter_tensors(files):
        target_name, factor = adapter_key(name)
        layer = modules.get(target_name)
        if layer is None or not hasattr(layer, factor):
            raise ValueError(f"adapter tensor {name!r} has no LoRA layer")
        linear = getattr(layer, factor)
        if tuple(tensor.shape) != tuple(linear.weight.shape):
            raise ValueError(
                f"{name}: adapter shape {tuple(tensor.shape)} != {tuple(linear.weight.shape)}"
            )
        set_parameter(module, f"{target_name}.{factor}.weight", tensor)
        seen.add(f"{target_name}.{factor}")
        loaded += 1
    if seen != expected:
        raise ValueError(f"adapter covers {len(seen)} of {len(expected)} LoRA factors")
    return loaded


def cast_parameters(module: nn.Module, dtype: torch.dtype) -> None:
    """Hold every parameter in ``dtype``; buffers such as rotary frequencies keep theirs, as Transformers loads."""
    for parameter in module.parameters():
        parameter.data = parameter.data.to(dtype)


def lay_out_linears(
    module: nn.Module, make: Callable[[nn.Linear], nn.Module]
) -> nn.Module:
    """Every ``nn.Linear`` inside ``module`` replaced in place by ``make(linear)``."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, make(child))
        else:
            lay_out_linears(child, make)
    return module


def keep_linear_bf16(module: nn.Module) -> dict[str, int]:
    """Hold BF16-exact Linear weights in BF16 (what BF16 autocast multiplies with anyway).

    Weights BF16 cannot hold exactly (for example FP32-trained LoRA factors) and
    weights shared with other module kinds stay FP32.
    """
    shared = {
        id(parameter)
        for layer in module.modules()
        if not isinstance(layer, nn.Linear)
        for parameter in layer.parameters(recurse=False)
    }
    counts = {"linear_bf16": 0, "linear_fp32": 0}
    for layer in module.modules():
        if not isinstance(layer, nn.Linear):
            continue
        for parameter in (layer.weight, layer.bias):
            if (
                parameter is None
                or parameter.dtype != torch.float32
                or id(parameter) in shared
            ):
                continue
            rounded = parameter.data.to(torch.bfloat16)
            if torch.equal(rounded.float(), parameter.data):
                parameter.data = rounded
        counts[
            "linear_bf16" if layer.weight.dtype == torch.bfloat16 else "linear_fp32"
        ] += 1
    return counts
