"""Validated, backend-specific GPU visibility for Decision containers."""

from __future__ import annotations

import re

GPU_DEVICE_INDEX = re.compile(r"(?:0|[1-9][0-9]{0,3})")
ROCM_VISIBLE_DEVICES_ENV = "ROCR_VISIBLE_DEVICES"


class DecisionGPUDeviceError(ValueError):
    """A GPU selector is malformed or unsupported by the resolved backend."""


def validate_gpu_device(device: object) -> str:
    """Accept one canonical decimal device index, not a list or range."""

    if not isinstance(device, str) or GPU_DEVICE_INDEX.fullmatch(device) is None:
        raise DecisionGPUDeviceError(
            "--gpu-device must be one GPU index from 0 to 9999, without "
            "whitespace, lists, or ranges."
        )
    return device


def gpu_visibility_environment(backend: str, device: str) -> dict[str, str]:
    """Translate one selector to runtime visibility, failing closed elsewhere."""

    validated = validate_gpu_device(device)
    if backend == "rocm":
        return {ROCM_VISIBLE_DEVICES_ENV: validated}
    # CUDA needs its own container-level device-selection contract and hardware
    # validation before a public selector can be enabled for that backend.
    raise DecisionGPUDeviceError(
        "--gpu-device is currently supported only for the ROCm Decision backend."
    )
