"""Quantized checkpoints the native engine loads: compressed-tensors int4 (W4A16) and FP8 (W8A8) weights.

``parse`` reads a checkpoint's ``config.json``. A compressed-tensors
``quantization_config`` with one group of weights is accepted in two schemes:

- ``int4``: the ``pack-quantized`` format, 4-bit integer weights in groups of
  input columns, symmetric (GPTQ) or asymmetric with zero points (AWQ), and
  BF16 activations.
- ``float8``: the ``float-quantized`` format, FP8 (E4M3) weights with one scale
  per output channel, and activations quantized to FP8 per token as each call
  runs (``FP8_DYNAMIC``).

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
INT4 = "int4"
FLOAT8 = "float8"
# Checkpoint tensors that describe a quantized weight rather than hold parameters.
METADATA = ("weight_scale", "weight_shape", "weight_zero_point")


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


def parse(config: dict[str, Any]) -> QuantizedWeights | None:
    """The quantized weights ``config.json`` declares, None without a ``quantization_config``."""
    quantization = config.get("quantization_config")
    if quantization is None:
        return None
    groups = quantization.get("config_groups") or {}
    form = quantization.get("format")
    if (
        quantization.get("quant_method", "compressed-tensors") != "compressed-tensors"
        or form not in ("pack-quantized", "float-quantized")
        or len(groups) != 1
    ):
        raise PackageError(
            "only compressed-tensors int4 (pack-quantized) and FP8 (float-quantized) "
            f"weights are supported, not {form!r}"
        )
    (group,) = groups.values()
    weights = group.get("weights") or {}
    common = {
        "targets": tuple(group.get("targets") or ()),
        "ignore": tuple(quantization.get("ignore") or ()),
    }
    if group.get("output_activations") is not None:
        raise PackageError("quantized output activations are not supported")
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
