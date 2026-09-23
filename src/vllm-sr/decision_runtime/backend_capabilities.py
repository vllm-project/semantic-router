"""Runtime-owned hardware capabilities for the two Decision model families.

The model repository describes weights and optional tuning data. It does not
authorize an execution backend. Keep that decision in the installed runtime so
a new weight revision can use an existing, structurally compatible executor.
"""

from __future__ import annotations

from dataclasses import dataclass

from cli.model_catalog_types import CatalogProviderModel

from .runtime_profile import RuntimeProfile, UnsupportedRuntimeBackendError


@dataclass(frozen=True, slots=True)
class RuntimeBackendCapability:
    backend: str
    backbone_dtype: str
    target: str | None


def require_runtime_backend(
    catalog: CatalogProviderModel,
    profile: RuntimeProfile,
    backend: str,
    *,
    target: str | None = None,
) -> RuntimeBackendCapability:
    """Check family, model size, and installed hardware implementation.

    CPU is an execution path for sub-1B Decision models. Product qualification
    still requires model-backed CPU receipts; this method does not assert that
    a particular revision meets quality or latency targets.
    """

    if backend == "rocm" and profile.family in {"vela", "qwen3.5"}:
        if target is not None and target != "gfx942":
            raise UnsupportedRuntimeBackendError(
                f"Decision ROCm target {target!r} is unsupported"
            )
        dtype = "float32" if profile.family == "vela" else "bfloat16"
        return RuntimeBackendCapability(backend, dtype, "gfx942")

    if backend == "cpu" and target is None and profile.family in {"vela", "qwen3.5"}:
        try:
            size = float(catalog.parameter_size.removesuffix("B"))
        except (TypeError, ValueError):
            size = float("inf")
        if catalog.parameter_size.endswith("B") and 0 < size < 1:
            return RuntimeBackendCapability(backend, "float32", None)

    raise UnsupportedRuntimeBackendError(
        f"Decision {profile.family} has no installed {backend!r} executor for "
        f"{catalog.parameter_size} on {target or 'this host'}"
    )
