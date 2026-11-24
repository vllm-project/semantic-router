"""``--dtype``: the precision a backbone computes in.

``auto`` runs its family's ``DtypePolicy``: the precision the model was
published in (Decision 3.0 computes in FP32 on CPUs instead, where its BF16
answers depend on the CPU's kernels, except on a CPU class with its own BF16
references: ``on_host``), with the kernel variants of its released runtime and
the faster copies and kernels its records consent to. Any other
value that names that precision on the device class is ``auto`` too.
Otherwise every backbone and tower weight is held and computed in that dtype,
with the reference kernels: an explicit choice of numerics, which only golden
references recorded at that dtype check (``readiness.qualified``: Decision
3.0's BF16 on CPUs).
"""

from __future__ import annotations

from dataclasses import replace

from .plugins.base import Accelerator, DeviceInfo, DtypePolicy, ModelSpec


def on_host(
    spec: ModelSpec, dtype: str, accelerator: Accelerator, device: DeviceInfo
) -> ModelSpec:
    """``spec`` as ``--dtype dtype`` runs it on ``device``.

    On a CPU that computes BF16 natively and whose golden reference key
    (``Accelerator.golden_reference``) has a BF16 reference for every golden
    request (``DtypePolicy.cpu_bf16_references``), ``auto`` holds BF16 as
    ``--dtype bfloat16`` does, with no reduced copy or approximate CPU kernel:
    those consents rest on records at the policy's own precision. Elsewhere,
    and at any explicit ``--dtype``, ``spec`` is unchanged.
    """
    policy = spec.dtype
    if (
        dtype != "auto"
        or not device.bf16
        or policy.cpu_weights == "bfloat16"
        or not policy.cpu_bf16_references
        or accelerator.torch_device(device).type != "cpu"
        or accelerator.golden_reference(device) not in policy.cpu_bf16_references
    ):
        return spec
    return replace(
        spec,
        dtype=replace(
            policy,
            cpu_weights="bfloat16",
            reduced_cpu=None,
            approximate_cpu_kernels=False,
        ),
    )


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
