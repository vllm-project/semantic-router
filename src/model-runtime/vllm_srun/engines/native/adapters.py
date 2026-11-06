"""LoRA adapter networks over a loaded backbone (``BackboneSpec.adapters``).

An adapter network is the backbone's architecture whose layer stack carries
unmerged LoRA factors on every target Linear and, where the adapter trained
them, its own copies of whole layers. Every module outside the layer stack
(embedding, rotary tables, final norm) is the backbone's own module, and every
layer parameter the adapter files do not hold is the backbone's own
``Parameter`` object: the network adds the adapter's tensors only, as the
released runtimes build it (frozen weights shared by reference).
"""

from __future__ import annotations

import torch
from torch import nn

from ...plugins.base import AdapterSpec, BackboneSpec
from . import models
from .models.lora import attach
from .weights import iter_tensors, set_parameter

LAYERS = "layers"
BASE = ".base."
BASE_LAYER = ".base_layer."
FACTORS = ("lora_A", "lora_B")


def module_name(checkpoint_name: str) -> str:
    """An adapter tensor's parameter name in the network: ``X.base.weight`` -> ``X.base_layer.weight``, ``X.lora_A`` -> ``X.lora_A.weight``."""
    if checkpoint_name.rpartition(".")[2] in FACTORS:
        return f"{checkpoint_name}.weight"
    return checkpoint_name.replace(BASE, BASE_LAYER)


def targets(network: nn.Module, leaves: tuple[str, ...]) -> list[str]:
    """Every Linear of the layer stack whose leaf name is one of ``leaves``."""
    return [
        name
        for name, module in network.named_modules()
        if name.startswith(f"{LAYERS}.")
        and isinstance(module, nn.Linear)
        and name.rpartition(".")[2] in leaves
    ]


def load_adapter_network(
    backbone: nn.Module,
    spec: BackboneSpec,
    name: str,
    dtype: torch.dtype = torch.float32,
    private: bool = True,
) -> nn.Module:
    """The adapter ``name`` of ``spec`` over ``backbone`` (loaded): shared modules, LoRA, the adapter's tensors."""
    adapter: AdapterSpec = spec.adapters[name]
    with torch.device("meta"):
        network = models.build(spec.model_type, spec.config)
    for child, _ in list(network.named_children()):
        if child != LAYERS:
            setattr(network, child, getattr(backbone, child))
    wrapped = targets(network, adapter.targets)
    if not wrapped:
        raise ValueError(f"adapter {name!r} targets no Linear layer of the backbone")
    with torch.device("meta"):
        attach(network, wrapped, adapter.rank, adapter.alpha / adapter.rank)
    parameters = dict(network.named_parameters())
    loaded: set[str] = set()
    for checkpoint_name, tensor in iter_tensors(
        adapter.weight_files, adapter.weight_prefix
    ):
        target = module_name(checkpoint_name)
        if not target.startswith(f"{LAYERS}.") or target not in parameters:
            raise ValueError(
                f"adapter {name!r} tensor {checkpoint_name!r} has no parameter"
            )
        if tuple(tensor.shape) != tuple(parameters[target].shape):
            raise ValueError(
                f"adapter {name!r} {checkpoint_name}: shape {tuple(tensor.shape)}"
                f" != {tuple(parameters[target].shape)}"
            )
        set_parameter(network, target, tensor, dtype, private)
        loaded.add(target)
    shared = dict(backbone.named_parameters())
    for target, parameter in parameters.items():
        if target in loaded or not parameter.is_meta:
            continue
        own = target.replace(BASE_LAYER, ".")
        if target.rpartition(".")[0].rpartition(".")[2] in FACTORS or own not in shared:
            raise ValueError(f"adapter {name!r} does not hold {target!r}")
        owner_name, _, leaf = target.rpartition(".")
        network.get_submodule(owner_name)._parameters[leaf] = shared[own]
    return network
