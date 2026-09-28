"""Qwen3.5 gated-delta runtime identity for decoder-track GPU jobs.

Transformers binds each gated-delta / causal-conv1d function once, at import,
to the installed package or to its reference PyTorch implementation. A launcher
path that hides ``/opt/decision-fla`` silently selects the reference path
(the 27B faults), and FLA without a persisted Triton autotune cache trains but
does not reproduce across processes. ``require_runtime`` fails a GPU job before
any work unless every binding is the image's kernel and the autotune cache is
shared and persisted.
"""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version
from typing import Any

BINDINGS = {
    "torch_chunk_gated_delta_rule": "fla.",
    "torch_recurrent_gated_delta_rule": "fla.",
    "causal_conv1d_fn": "causal_conv1d.",
    "causal_conv1d_update": "causal_conv1d.",
}


def _binding(function: Any, depth: int = 0) -> tuple[str, bool] | None:
    """Follow decorator closures to the cell named ``implementation``."""
    code = getattr(function, "__code__", None)
    cells = getattr(function, "__closure__", None) or ()
    if code is not None and "implementation" in code.co_freevars:
        values = {}
        for name, cell in zip(code.co_freevars, cells):
            try:
                values[name] = cell.cell_contents
            except ValueError:
                continue
        implementation = values.get("implementation")
        if implementation is not None:
            name = f"{implementation.__module__}.{implementation.__qualname__}"
            return name, bool(values.get("is_new_implementation"))
    if depth >= 4:
        return None
    for cell in cells:
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if callable(value):
            found = _binding(value, depth + 1)
            if found is not None:
                return found
    inner = getattr(function, "__wrapped__", None)
    return _binding(inner, depth + 1) if inner is not None else None


def kernel_bindings(module: Any) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in BINDINGS:
        found = _binding(getattr(module, name, None))
        result[name] = found[0] if found and found[1] else None
    return result


def _package(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def runtime_identity() -> dict[str, Any]:
    from transformers.models.qwen3_5 import modeling_qwen3_5

    bindings = kernel_bindings(modeling_qwen3_5)
    try:
        import fla

        fla_path = os.path.dirname(fla.__file__)
    except ImportError:
        fla_path = None
    return {
        "kernel_bindings": bindings,
        "fla_path": fla_path,
        "flash_linear_attention": _package("flash-linear-attention"),
        "causal_conv1d": _package("causal-conv1d") or _package("causal_conv1d"),
        "triton": _package("triton"),
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        "triton_cache_autotuning": os.environ.get("TRITON_CACHE_AUTOTUNING"),
    }


def violations(identity: dict[str, Any]) -> list[str]:
    problems = [
        f"{name} is not bound to {prefix}* (got {identity['kernel_bindings'][name]})"
        for name, prefix in BINDINGS.items()
        if not (identity["kernel_bindings"][name] or "").startswith(prefix)
    ]
    if identity["triton_cache_autotuning"] != "1":
        problems.append("TRITON_CACHE_AUTOTUNING must be 1")
    if not identity["triton_cache_dir"]:
        problems.append("TRITON_CACHE_DIR must name the shared persisted cache")
    return problems


def require_runtime() -> dict[str, Any]:
    identity = runtime_identity()
    problems = violations(identity)
    if problems:
        raise RuntimeError("Decoder runtime check failed: " + "; ".join(problems))
    return identity
