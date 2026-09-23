"""Optional, instance-local ROCm convolution acceleration for Decision Eos.

The optimization is deliberately a narrow opt-in seam.  It never changes a
Transformers module global, never affects another resident model, and falls
back to the unchanged library convolution whenever its exact release contract
cannot be proved.  The optimized kernel itself is deliberately supplied by a
vLLM-SR-owned optional module; no model-repository Python is imported.
"""

from __future__ import annotations

import hashlib
import inspect
import textwrap
import types
from dataclasses import dataclass
from typing import Any, Callable


_EOS_FORWARD_SHA256 = "a781d4df0c45a955a3b8f1d9433e5749ea8cc2148b636566a64745ee271dd31b"
_EOS_TORCH_GIT = "6bbd26020da1c6dc198625dfcdd968b1e4e6b1c5"
_EOS_TRANSFORMERS = "5.17.0"
_EOS_GFX = "gfx942"
_EOS_EXPECTED_LAYERS = 18


@dataclass(frozen=True, slots=True)
class EosRocmConvReceipt:
    """A truthful record of whether one Eos instance received the optimization."""

    installed: bool
    reason: str
    layers: int = 0
    forward_sha256: str | None = None


class _ConvController:
    def __init__(
        self, reference: Callable[..., Any], kernel: Callable[..., Any], torch: Any
    ) -> None:
        self._reference = reference
        self._kernel = kernel
        self._torch = torch

    def __call__(self, x, weight, bias=None, activation=None, **kwargs):
        # This geometry is the only one covered by the owned kernel contract.
        supported = (
            not self._torch.is_grad_enabled()
            and x.device.type == "cuda"
            and x.ndim == 3
            and x.dtype == self._torch.bfloat16
            and weight.dtype == self._torch.bfloat16
            and bias is None
            and activation in {"silu", "swish"}
            and tuple(weight.shape) == (x.shape[1], 4)
            and weight.is_contiguous()
            and x.shape[0] * x.shape[2] >= 2048
        )
        if supported:
            return self._kernel(x, weight, bias, activation=activation, **kwargs)
        return self._reference(x, weight, bias, activation=activation, **kwargs)


def install_eos_rocm_conv(
    model: Any,
    *,
    torch: Any,
    transformers: Any,
    modeling_qwen35: Any,
    kernel: Callable[..., Any] | None,
) -> EosRocmConvReceipt:
    """Install the optional fast path on one fully resident Eos model only.

    A missing optional kernel, a non-qualified stack, hooks/offload, or source
    drift all return a non-installed receipt.  None are errors because they
    retain the exact upstream path.
    """

    if kernel is None:
        return EosRocmConvReceipt(False, "optional_kernel_unavailable")
    if (
        not getattr(torch.version, "hip", None)
        or getattr(torch.version, "git_version", None) != _EOS_TORCH_GIT
        or getattr(transformers, "__version__", None) != _EOS_TRANSFORMERS
    ):
        return EosRocmConvReceipt(False, "unqualified_runtime")
    try:
        parameter = next(model.parameters())
    except (AttributeError, StopIteration):
        return EosRocmConvReceipt(False, "model_has_no_parameters")
    if parameter.device.type != "cuda":
        return EosRocmConvReceipt(False, "non_gpu_model")
    architecture = getattr(
        torch.cuda.get_device_properties(parameter.device), "gcnArchName", ""
    ).split(":")[0]
    if architecture != _EOS_GFX:
        return EosRocmConvReceipt(False, "unqualified_gpu")
    forward = inspect.unwrap(modeling_qwen35.Qwen3_5GatedDeltaNet.forward)
    try:
        source = inspect.getsource(forward)
    except (OSError, TypeError):
        return EosRocmConvReceipt(False, "unavailable_forward_source")
    digest = hashlib.sha256(textwrap.dedent(source).encode()).hexdigest()
    if digest != _EOS_FORWARD_SHA256 or forward.__closure__ is not None:
        return EosRocmConvReceipt(False, "unqualified_forward", forward_sha256=digest)
    layers = [
        value
        for value in model.modules()
        if type(value) is modeling_qwen35.Qwen3_5GatedDeltaNet
    ]
    if len(layers) != _EOS_EXPECTED_LAYERS or any(
        hasattr(layer, "_hf_hook")
        or hasattr(layer.conv1d, "_hf_hook")
        or "forward" in layer.__dict__
        for layer in layers
    ):
        return EosRocmConvReceipt(False, "modified_or_offloaded_model")

    reference = modeling_qwen35.causal_conv1d_fn
    controller = _ConvController(reference, kernel, torch)
    namespace = dict(forward.__globals__)
    namespace["causal_conv1d_fn"] = controller
    local_forward = types.FunctionType(
        forward.__code__,
        namespace,
        forward.__name__,
        forward.__defaults__,
        forward.__closure__,
    )
    local_forward.__kwdefaults__ = forward.__kwdefaults__
    # Preserve the library's known acceleration wrapper for this instance
    # without rebinding a class or a module-global function.
    local_forward = modeling_qwen35.force_accelerate_hooks("conv1d")(local_forward)
    for layer in layers:
        layer.forward = types.MethodType(local_forward, layer)
    if modeling_qwen35.causal_conv1d_fn is not reference:
        # Defensive fail-closed check: do not permit a global mutation.
        return EosRocmConvReceipt(False, "unexpected_global_mutation")
    return EosRocmConvReceipt(
        True,
        "installed",
        layers=len(layers),
        forward_sha256=digest,
    )
