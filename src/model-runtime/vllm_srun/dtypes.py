"""``--dtype``: the precision a backbone computes in.

``auto`` runs the precision the model was published in: its family's
``DtypePolicy``, with the kernel variants of its released runtime and the
faster copies and kernels its records consent to. Any other value that names
the published precision on the device class is ``auto`` too. Otherwise every
backbone and tower weight is held and computed in that dtype, with the
reference kernels: an explicit choice of numerics, which the golden
references (recorded at the published precision) no longer check.
"""

from __future__ import annotations

from dataclasses import replace

from .plugins.base import DtypePolicy, ModelSpec


def published(policy: DtypePolicy, gpu: bool) -> str:
    """The precision ``policy`` computes in on a GPU or a CPU."""
    held = policy.gpu_weights if gpu else policy.cpu_weights
    if held:
        return held
    if gpu and policy.autocast:
        return policy.autocast
    return "float32"


def resolve(dtype: str, policy: DtypePolicy, gpu: bool) -> str:
    """``dtype``, or ``auto`` when it names the published precision."""
    return "auto" if dtype in ("auto", published(policy, gpu)) else dtype


def apply(spec: ModelSpec, dtype: str, gpu: bool) -> ModelSpec:
    """``spec`` as it runs at ``dtype`` on a GPU or a CPU."""
    if resolve(dtype, spec.dtype, gpu) == "auto":
        return spec
    held = None if dtype == "float32" else dtype
    return replace(
        spec,
        dtype=DtypePolicy(
            weights="float32",
            autocast=None,
            head="float32",
            bf16_resident=False,
            gpu_weights=held,
            cpu_weights=held,
        ),
        kernel_variants={},
    )
