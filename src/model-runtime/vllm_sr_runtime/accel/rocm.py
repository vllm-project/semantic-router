"""ROCm accelerator (MI300X / MI325X, gfx942). Validated by the parity records."""

from __future__ import annotations

from typing import Any

from .gpu import GPUAccelerator


class ROCmAccelerator(GPUAccelerator):
    name = "rocm"
    validated = True
    platform_attribute = "hip"

    def arch(self, properties: Any) -> str | None:
        name = getattr(properties, "gcnArchName", None)
        return name.split(":", 1)[0] if isinstance(name, str) else None
