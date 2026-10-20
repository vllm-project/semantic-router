"""Shared logic for CUDA and ROCm accelerators (both use the torch.cuda device API).

The released runtime runs the backbone under BF16 autocast and takes the
gated-delta and causal-convolution kernels from FLA and causal-conv1d when
they are installed, else the torch references. The same order applies here.
"""

from __future__ import annotations

import importlib
import inspect
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch

from ..engines.native.quantized import float8_format
from ..plugins.base import Accelerator, DeviceInfo
from .kernels import Kernel, KernelSet, reference_kernels, varlen_attention_ref

_DEVICE_LOCKS: dict[int, Any] = {}
_LOCKS = threading.Lock()

# Errors after which the process's device context stays unusable.
DEVICE_FAULT_MARKERS = (
    "CUDA error",
    "HIP error",
    "device-side assert",
    "illegal memory access",
)


def device_lock(index: int) -> Any:
    """The lock that serializes this process's device work on one GPU."""
    with _LOCKS:
        lock = _DEVICE_LOCKS.get(index)
        if lock is None:
            lock = _DEVICE_LOCKS[index] = threading.RLock()
        return lock


def _current_device(target: torch.device) -> AbstractContextManager[Any]:
    if target.type != "cuda" or not torch.cuda.is_available():
        return nullcontext()
    return torch.cuda.device(target)


def _optional(module: str, attribute: str) -> Any:
    try:
        return getattr(importlib.import_module(module), attribute)
    except Exception:
        return None


def nvfp4_unavailable(index: int | None) -> str | None:
    """Why device ``index`` cannot multiply NVFP4 weights (``quantization.NVFP4``); None when it can.

    NVFP4 needs NVIDIA FP4 tensor cores (Blackwell: compute capability 10.x or
    12.x), a PyTorch with FP4 tensors and the block-scaled ``scaled_mm``, and
    Triton, which quantizes the activations.
    """
    if torch.version.hip is not None:
        return "NVFP4 weights need NVIDIA FP4 tensor cores, which ROCm GPUs do not have"
    scaling = _optional("torch.nn.functional", "ScalingType")
    if not hasattr(torch, "float4_e2m1fn_x2") or not hasattr(scaling, "BlockWise1x16"):
        return (
            f"PyTorch {torch.__version__} has no FP4 tensors with block-scaled "
            "torch.nn.functional.scaled_mm"
        )
    if not torch.cuda.is_available():
        return "no CUDA device"
    major, minor = torch.cuda.get_device_capability(index or 0)
    if major not in (10, 12):
        return (
            f"compute capability {major}.{minor} has no FP4 tensor cores "
            "(NVFP4 needs a Blackwell GPU, compute capability 10.x or 12.x)"
        )
    if _optional("triton", "__version__") is None:
        return "NVFP4 activations are quantized with Triton, which is not installed"
    return None


class GPUAccelerator(Accelerator):
    """Base for accelerators that drive devices through ``torch.cuda``."""

    platform_attribute: str = ""

    def available(self) -> bool:
        return (
            bool(getattr(torch.version, self.platform_attribute, None))
            and torch.cuda.is_available()
        )

    def devices(self) -> list[DeviceInfo]:
        if not self.available():
            return []
        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            try:
                free, total = torch.cuda.mem_get_info(index)
            except RuntimeError:
                free, total = None, properties.total_memory
            devices.append(
                DeviceInfo(
                    accelerator=self.name,
                    index=index,
                    name=properties.name,
                    total_memory=total,
                    free_memory=free,
                    bf16=torch.cuda.is_bf16_supported(),
                    arch=self.arch(properties),
                )
            )
        return devices

    def arch(self, properties: Any) -> str | None:
        return None

    def torch_device(self, device: DeviceInfo) -> torch.device:
        return torch.device("cuda", device.index or 0)

    def kernels(self, device: DeviceInfo) -> KernelSet:
        kernels = reference_kernels(device.label)
        conv = _optional("causal_conv1d", "causal_conv1d_fn")
        if conv is not None:
            kernels.register(
                Kernel("causal_conv1d", _wrap_conv(conv), "causal-conv1d", exact=True)
            )
        delta = _optional("fla.ops.gated_delta_rule", "chunk_gated_delta_rule")
        if delta is not None:
            kernels.register(
                Kernel("chunk_gated_delta_rule", _wrap_delta(delta), "fla", exact=True)
            )
        static = _static_delta()
        if static is not None:
            kernels.register(
                Kernel("chunk_gated_delta_rule_static", static, "fla", exact=True)
            )
        varlen = _optional("torch.nn.attention.varlen", "varlen_attn")
        if varlen is not None:
            kernels.register(
                Kernel(
                    "varlen_attention",
                    _wrap_varlen(varlen),
                    "torch-varlen-attention",
                    exact=False,
                )
            )
        return kernels

    def capabilities(self, device: DeviceInfo) -> dict[str, bool]:
        return {
            "bf16_autocast": device.bf16,
            "graphs": True,
            "triton": _optional("triton", "__version__") is not None,
            "fla": _optional("fla.ops.gated_delta_rule", "chunk_gated_delta_rule")
            is not None,
            "causal_conv1d": _optional("causal_conv1d", "causal_conv1d_fn") is not None,
            "float8": float8_format(self.torch_device(device)) is not None,
            "nvfp4": nvfp4_unavailable(device.index) is None,
        }

    def lacks(self, device: DeviceInfo, capability: str) -> str:
        if capability == "nvfp4":
            reason = nvfp4_unavailable(device.index) or "no FP4 tensor cores"
            return f"{device.label}: {reason}; serve NVFP4 packages on a Blackwell GPU"
        return super().lacks(device, capability)

    def autocast(
        self, device: DeviceInfo, dtype: str | None
    ) -> AbstractContextManager[Any]:
        if dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=getattr(torch, dtype))

    def synchronize(self, device: DeviceInfo) -> None:
        torch.cuda.synchronize(device.index or 0)

    def execute(self, device: DeviceInfo, work: Callable[[], Any]) -> Any:
        """Run device work on its device, holding the device's lock, so the process's models never launch on it at once.

        A model captures a HIP / CUDA graph the second time it sees a shape,
        and the capture fails when another thread launches work on the device,
        whatever the capture mode; every capture also synchronizes the device.
        Triton kernels (FLA's among them) launch on the thread's current
        device, so the work makes the model's GPU current: otherwise a model on
        any GPU but the first faults reading its own memory from GPU 0.
        """
        with device_lock(device.index or 0), _current_device(self.torch_device(device)):
            return work()

    def device_fault(self, error: BaseException) -> bool:
        """Device errors poison the context; running out of memory fails only the batch."""
        if isinstance(error, torch.OutOfMemoryError):
            return False
        if isinstance(error, getattr(torch, "AcceleratorError", ())):
            return True
        return isinstance(error, RuntimeError) and any(
            marker in str(error) for marker in DEVICE_FAULT_MARKERS
        )


# causal-conv1d computes a row's offset in 32 bits (``index_t`` is ``uint32_t``): in a
# call over more elements, a row past the range reads and writes other rows' memory and
# its own output is never written. Calls stay within the signed range.
CONV_ELEMENTS = 1 << 31


def _wrap_conv(fn: Any) -> Any:
    """causal-conv1d's convolution; a call over more than ``CONV_ELEMENTS`` elements runs
    in slices of whole rows."""

    def conv(hidden_states, weight, bias=None, activation=None):
        batch, channels, length = hidden_states.shape
        row = max(hidden_states.stride(0), channels * length, 1)
        rows = max(1, CONV_ELEMENTS // row)
        if batch <= rows:
            return fn(hidden_states, weight, bias=bias, activation=activation)
        out = torch.empty_like(hidden_states)
        for start in range(0, batch, rows):
            part = hidden_states[start : start + rows]
            out[start : start + rows] = fn(part, weight, bias, activation=activation)
        return out

    return conv


# FLA's chunk length, which a static layout's chunk index counts in.
CHUNK = 64


def _static_delta() -> Any:
    """FLA's chunked gated delta rule over packed sequences with a caller-given chunk index, for graphs.

    The public entry point builds the chunk index from ``cu_seqlens`` on the
    host, which a captured graph would keep from its capture. This one takes it
    as a device tensor the caller rewrites before each replay and derives the
    rest from ``cu_seqlens`` on the device. FLA's tensor cache matches tensors
    by identity, so its chunk offsets are reused only among the calls of one
    forward, whose layout views are its own. Otherwise it is the public path:
    L2-normalized query and key, scale ``1 / sqrt(K)``, the output cast to the
    query's dtype. None where this FLA lacks the entry point.
    """
    try:
        from fla.modules.l2norm import l2norm_fwd
        from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd
    except Exception:
        return None
    if "chunk_indices" not in inspect.signature(chunk_gated_delta_rule_fwd).parameters:
        return None

    def delta(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        output_final_state,
        cu_seqlens,
        chunk_indices,
    ):
        q, _ = l2norm_fwd(query.contiguous())
        k, _ = l2norm_fwd(key.contiguous())
        _, out, _, final, _, _ = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=value.contiguous(),
            g=g.contiguous(),
            beta=beta.contiguous(),
            scale=k.shape[-1] ** -0.5,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        return out.to(q.dtype), final

    delta.chunk_indexed = True  # type: ignore[attr-defined]
    delta.regions = _static_regions(l2norm_fwd)  # type: ignore[attr-defined]
    return delta


def _static_regions(l2norm_fwd: Any) -> Any:
    """``chunk_gated_delta_rule_static`` over a static layout's two regions, the second continuing the first.

    FLA's gate cumulative sum and intra-chunk solve treat every chunk alone, so
    they run once over both regions' chunks (the layout's combined offsets and
    chunk index); the recurrence runs per region, the second from the first's
    final states of each segment's ``owner``, and the output per region. Every
    chunk computes what the region's own call computes. None where this FLA's
    forward has other stages.
    """
    try:
        from fla.ops.gated_delta_rule import chunk as fla_chunk
    except Exception:
        return None
    stages = (
        "chunk_local_cumsum",
        "chunk_gated_delta_rule_fwd_intra",
        "chunk_gated_delta_rule_fwd_h",
        "chunk_fwd_o",
    )
    if not hasattr(fla_chunk, "RCP_LN2") or not all(
        "chunk_indices"
        in inspect.signature(getattr(fla_chunk, name, lambda: None)).parameters
        for name in stages
    ):
        return None
    cumsum, intra, recurrence, output = (getattr(fla_chunk, name) for name in stages)
    source = inspect.getsource(fla_chunk.chunk_gated_delta_rule_fwd)
    if any(f"{name}(" not in source for name in stages):
        return None

    def regions(query, key, value, g, beta, split, combined, first, second, owner):
        q, _ = l2norm_fwd(query.contiguous())
        k, _ = l2norm_fwd(key.contiguous())
        v = value.contiguous()
        g = cumsum(
            g.contiguous(), chunk_size=CHUNK, scale=fla_chunk.RCP_LN2,
            cu_seqlens=combined[0], chunk_indices=combined[1],
        )  # fmt: skip
        w, u, _ = intra(
            k=k, v=v, g=g, beta=beta.contiguous(),
            cu_seqlens=combined[0], chunk_size=CHUNK, chunk_indices=combined[1],
        )  # fmt: skip
        outputs, initial = [], None
        for rows, (offsets, chunks), last in (
            (slice(0, split), first, False),
            (slice(split, None), second, True),
        ):
            h, v_new, final = recurrence(
                k=k[:, rows], w=w[:, rows], u=u[:, rows], g=g[:, rows],
                initial_state=initial, output_final_state=not last, chunk_size=CHUNK,
                cu_seqlens=offsets, chunk_indices=chunks,
            )  # fmt: skip
            outputs.append(output(
                q=q[:, rows], k=k[:, rows], v=v_new, h=h, g=g[:, rows], scale=k.shape[-1] ** -0.5,
                cu_seqlens=offsets, chunk_size=CHUNK, chunk_indices=chunks,
            ))  # fmt: skip
            if not last:
                initial = final[owner].contiguous()
        return torch.cat(outputs, dim=1).to(q.dtype)

    return regions


def _wrap_varlen(fn: Any) -> Any:
    """PyTorch's variable-length flash attention over packed rows, non-causal, keys with fewer heads in-kernel."""
    accepted = set(inspect.signature(fn).parameters)

    def attend(query, key, value, packed, *, scale, groups):
        if query.dtype not in (torch.float16, torch.bfloat16):
            # flash attention takes 16-bit inputs only (FP32 at --dtype float32)
            return varlen_attention_ref(
                query, key, value, packed, scale=scale, groups=groups
            )
        # Torch 2.10 uses the default scale and infers GQA from the head counts.
        if "scale" not in accepted and scale != query.shape[-1] ** -0.5:
            return varlen_attention_ref(
                query, key, value, packed, scale=scale, groups=groups
            )
        extra = {}
        if "scale" in accepted:
            extra["scale"] = scale
        if "enable_gqa" in accepted:
            extra["enable_gqa"] = groups > 1
        cu = packed.cu_seqlens_int32
        return fn(
            query,
            key,
            value,
            cu,
            cu,
            packed.max_length,
            packed.max_length,
            **extra,
        )

    return attend


def _wrap_delta(fn: Any) -> Any:
    """FLA's chunked gated delta rule, called as Transformers' fallback wrapper calls it.

    ``cu_seqlens`` (variable-length sequences packed in one row, shared-context
    tree mode) passes through, with ``cu_seqlens_cpu`` where FLA takes it.
    """
    accepted = set(inspect.signature(fn).parameters)

    def delta(
        query,
        key,
        value,
        g,
        beta,
        chunk_size=64,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None,
        cu_seqlens_cpu=None,
    ):
        extra = {}
        if cu_seqlens is not None:
            extra["cu_seqlens"] = cu_seqlens
            if cu_seqlens_cpu is not None and "cu_seqlens_cpu" in accepted:
                extra["cu_seqlens_cpu"] = cu_seqlens_cpu
        return fn(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            **extra,
        )

    return delta
