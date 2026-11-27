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


def approximate_sources(
    spec: ModelSpec, accelerator: Accelerator, device: DeviceInfo
) -> tuple[str, ...]:
    """Where the approximate CPU kernels ``spec`` consents to come from on ``device``, at ``auto``."""
    kernels = accelerator.kernels(device)
    kernels.consent(spec.dtype.consents_on(accelerator.name, True))
    kernels.use_variants(spec.kernel_variants)
    return kernels.approximate_sources()


def with_kernels(key: str, sources: tuple[str, ...]) -> str:
    """A CPU key that also names the approximate kernels its BF16 answers were computed with."""
    return ":".join((key, *sources))


def referenced(policy: DtypePolicy, key: str) -> bool:
    """Whether ``key`` has a BF16 reference for every golden request (one seam, for the recorder)."""
    return key in policy.cpu_bf16_references


def on_host(
    spec: ModelSpec, dtype: str, accelerator: Accelerator, device: DeviceInfo
) -> ModelSpec:
    """``spec`` as ``--dtype dtype`` runs it on ``device``.

    On a CPU that computes BF16 natively and whose golden reference key
    (``Accelerator.golden_reference``) has a BF16 reference for every golden
    request (``DtypePolicy.cpu_bf16_references``), ``auto`` holds BF16 as
    ``--dtype bfloat16`` does, without a reduced copy. A BF16 reference holds
    one GEMM summation order, so the approximate CPU kernels the family
    consents to stay only where the key that also names them
    (``with_kernels``: ``cpu:…:amd:zentorch``) has references; with the
    plain key's references the model runs exact kernels. Elsewhere, and at any
    explicit ``--dtype``, ``spec`` is unchanged.
    """
    policy = spec.dtype
    if (
        dtype != "auto"
        or not device.bf16
        or policy.cpu_weights == "bfloat16"
        or accelerator.torch_device(device).type != "cpu"
    ):
        return spec
    key = accelerator.golden_reference(device)
    if key is None:
        return spec
    sources = approximate_sources(spec, accelerator, device)
    if sources and referenced(policy, with_kernels(key, sources)):
        consents = policy.approximate_cpu_kernels
    elif referenced(policy, key):
        consents = False
    else:
        return spec
    return replace(
        spec,
        dtype=replace(
            policy,
            cpu_weights="bfloat16",
            reduced_cpu=None,
            approximate_cpu_kernels=consents,
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
