"""Quantized checkpoints the native engine loads: compressed-tensors int4 (W4A16), FP8 (W8A8) and NVFP4 weights.

``parse`` reads a checkpoint's ``config.json``. A compressed-tensors
``quantization_config`` with one group of weights is accepted in two schemes:

- ``int4``: the ``pack-quantized`` format, 4-bit integer weights in groups of
  input columns, symmetric (GPTQ) or asymmetric with zero points (AWQ), and
  BF16 activations.
- ``float8``: the ``float-quantized`` format, FP8 (E4M3) weights with one scale
  per output channel, and activations quantized to FP8 per token as each call
  runs (``FP8_DYNAMIC``).
- ``nvfp4``: the ``nvfp4-pack-quantized`` format, FP4 (E2M1) weights packed two
  per byte with an FP8 (E4M3) scale per 16 input columns under one FP32 global
  scale per layer, and activations quantized the same way as each call runs,
  their 16-value block scales computed from the values under a calibrated
  global scale (``NVFP4``, ``dynamic: local``). It needs FP4 matrix units
  (Blackwell), so ``requires`` names the ``nvfp4`` capability on every
  accelerator and placement refuses other devices before any weight loads.

Anything else is refused rather than loaded as BF16. ``engines/native/quantized.py``
holds the layers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .errors import PackageError

PACK = 8  # 4-bit values per int32 word
INT4_BITS = 4
FLOAT8_BITS = 8
FP4_BITS = 4
FP4_GROUP = 16  # values per NVFP4 block scale
FP4_PER_BYTE = 2  # NVFP4 ``weight_packed`` holds two FP4 values per uint8
INT4 = "int4"
FLOAT8 = "float8"
FLOAT8_DEVICES = (
    "FP8 weights need an NVIDIA GPU of compute capability 8.9 or later "
    "or an AMD Instinct MI300 or MI350 series GPU"
)
NVFP4 = "nvfp4"
NVFP4_FORMAT = "nvfp4-pack-quantized"
FP8_SCALE_DTYPE = "torch.float8_e4m3fn"
# The device capability an NVFP4 checkpoint needs (``Accelerator.capabilities``).
NVFP4_CAPABILITY = "nvfp4"
# Checkpoint tensors that describe a quantized weight rather than hold parameters.
METADATA = (
    "weight_scale",
    "weight_shape",
    "weight_zero_point",
    "weight_global_scale",
    "input_global_scale",
)


@dataclass(frozen=True)
class QuantizedWeights:
    """Which linear layers a checkpoint holds quantized (``targets`` minus ``ignore``, full checkpoint names)."""

    scheme: str
    targets: tuple[str, ...]
    ignore: tuple[str, ...]
    group_size: int = 0
    symmetric: bool = True

    def holds(self, name: str) -> bool:
        return _matches(name, self.targets) and not _matches(name, self.ignore)


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    for pattern in patterns:
        if pattern.startswith("re:"):
            if re.fullmatch(pattern[3:], name):
                return True
        elif pattern in ("Linear", name):
            return True
    return False


def _is_nvfp4(args: dict[str, Any], dynamic: bool | str) -> bool:
    """Whether ``args`` (a ``weights`` or ``input_activations`` group) is compressed-tensors' NVFP4 scheme."""
    return (
        args.get("num_bits") == FP4_BITS
        and args.get("type") == "float"
        and args.get("strategy") == "tensor_group"
        and args.get("group_size") == FP4_GROUP
        and args.get("symmetric") is True
        and args.get("dynamic") == dynamic
        and args.get("scale_dtype", FP8_SCALE_DTYPE) == FP8_SCALE_DTYPE
        and args.get("block_structure") is None
        and args.get("actorder") is None
    )


def parse(config: dict[str, Any]) -> QuantizedWeights | None:
    """The quantized weights ``config.json`` declares, None without a ``quantization_config``."""
    quantization = config.get("quantization_config")
    if quantization is None:
        return None
    groups = quantization.get("config_groups") or {}
    form = quantization.get("format")
    if (
        quantization.get("quant_method", "compressed-tensors") != "compressed-tensors"
        or form not in ("pack-quantized", "float-quantized", NVFP4_FORMAT)
        or len(groups) != 1
    ):
        raise PackageError(
            "only compressed-tensors int4 (pack-quantized), FP8 (float-quantized) and "
            f"NVFP4 ({NVFP4_FORMAT}) weights are supported, not {form!r}"
        )
    (group,) = groups.values()
    weights = group.get("weights") or {}
    common = {
        "targets": tuple(group.get("targets") or ()),
        "ignore": tuple(quantization.get("ignore") or ()),
    }
    if group.get("output_activations") is not None:
        raise PackageError("quantized output activations are not supported")
    if form == NVFP4_FORMAT:
        inputs = group.get("input_activations") or {}
        if not (
            _is_nvfp4(weights, dynamic=False) and _is_nvfp4(inputs, dynamic="local")
        ):
            raise PackageError(
                "only NVFP4 is supported among FP4 schemes: FP4 (E2M1) weights and "
                "activations in groups of 16 with FP8 (E4M3) block scales under a tensor "
                "global scale, weights static and activations dynamic per block"
            )
        return QuantizedWeights(scheme=NVFP4, group_size=FP4_GROUP, **common)
    if form == "float-quantized":
        inputs = group.get("input_activations") or {}
        if (
            weights.get("num_bits") != FLOAT8_BITS
            or weights.get("type") != "float"
            or weights.get("strategy") != "channel"
            or weights.get("symmetric") is not True
            or weights.get("dynamic")
            or inputs.get("num_bits") != FLOAT8_BITS
            or inputs.get("type") != "float"
            or inputs.get("strategy") != "token"
            or inputs.get("dynamic") is not True
        ):
            raise PackageError(
                "only FP8 weights per output channel with FP8 activations per token "
                "(FP8_DYNAMIC) are supported"
            )
        return QuantizedWeights(scheme=FLOAT8, **common)
    size = weights.get("group_size")
    if (
        weights.get("num_bits") != INT4_BITS
        or weights.get("type") != "int"
        or not isinstance(weights.get("symmetric"), bool)
        or weights.get("strategy") != "group"
        or weights.get("dynamic")
        or weights.get("actorder") not in (None, "weight")
        or not isinstance(size, int)
        or size <= 0
        or size % PACK
        or group.get("input_activations") is not None
    ):
        raise PackageError(
            "only 4-bit integer weights in groups, with BF16 activations, are supported"
        )
    return QuantizedWeights(
        scheme=INT4, group_size=size, symmetric=weights["symmetric"], **common
    )
