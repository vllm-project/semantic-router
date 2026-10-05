"""ONNX Runtime execution providers per accelerator, and the session options the engine uses.

The CPU provider is validated. GPU providers run only when the installed
onnxruntime build has them, never fall back to the CPU for a node they cannot
run, and stay unvalidated until a record says otherwise.

CPU sessions share one intra-op pool (created before the first session) unless
a model on another engine serves the process's CPU too. Each session's own
pool spins after a run, so graphs that run one after another (Omni's CLAP
windows, then its audio graph) otherwise share the cores with the previous
session's spinning threads: on 16 cores CLAP + audio took 124 ms with
per-session pools and 43 ms with the shared one. An idle pool spins about
40 ms, which would double a native forward that follows an ONNX Runtime run on
the same cores, and the shared pool's spin can't be bounded. So beside another
engine's CPU models each session gets its own pool whose idle threads spin for
``SHARED_PROCESS_SPIN_US`` and then sleep: long enough to carry a run's
parallel sections, short enough to leave the cores to the next model. Pools
that stop spinning the moment a run returns cost Omni 13-24 % of its 4-caller
throughput, and pools that never spin take Nano text from 5.1 to 7.4 ms. Pools
are sized to the configured threads, else the CPUs the process may run on
(ONNX Runtime's own default counts the host's CPUs, not the cpuset).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ...plugins.base import DeviceInfo

# Preference order per accelerator; the first installed provider runs the model.
PROVIDERS: dict[str, tuple[str, ...]] = {
    "cpu": ("CPUExecutionProvider",),
    "cuda": ("CUDAExecutionProvider",),
    "rocm": ("MIGraphXExecutionProvider", "ROCMExecutionProvider"),
    "xpu": ("OpenVINOExecutionProvider",),
}
VALIDATED = frozenset({"CPUExecutionProvider"})
# Providers that also run on a CPU device when an operator asks for them by name.
CPU_ALTERNATIVES = frozenset({"OpenVINOExecutionProvider"})


@dataclass(frozen=True)
class ProviderChoice:
    """The execution provider that runs a model's graphs on one device."""

    name: str
    options: dict[str, str] = field(default_factory=dict)

    @property
    def validated(self) -> bool:
        return self.name in VALIDATED

    @property
    def gpu(self) -> bool:
        return (
            self.name != "CPUExecutionProvider"
            and self.options.get("device_type") != "CPU"
        )


def choose(
    device: DeviceInfo, available: Sequence[str], requested: str | None = None
) -> ProviderChoice | str:
    """The provider for ``device``, or the reason no installed provider can run it.

    ``requested`` names one provider explicitly (for example the OpenVINO
    provider on an Intel CPU); otherwise the accelerator's preference order
    applies.
    """
    candidates = PROVIDERS.get(device.accelerator)
    if candidates is None:
        return f"onnxruntime has no execution provider for {device.accelerator}"
    if requested is not None:
        allowed = candidates + (
            tuple(CPU_ALTERNATIVES) if device.accelerator == "cpu" else ()
        )
        if requested not in allowed:
            return f"{requested} cannot run on {device.label}"
        candidates = (requested,)
    for name in candidates:
        if name in available:
            return ProviderChoice(name, _options(name, device))
    return (
        f"the installed onnxruntime has none of {', '.join(candidates)} "
        f"(installed: {', '.join(available) or 'none'})"
    )


def _options(name: str, device: DeviceInfo) -> dict[str, str]:
    if name == "CPUExecutionProvider":
        return {}
    if name == "OpenVINOExecutionProvider":
        if device.accelerator == "cpu":
            return {"device_type": "CPU"}
        index = "" if device.index is None else f".{device.index}"
        return {"device_type": f"GPU{index}"}
    return {"device_id": str(device.index or 0)}


_SHARED_POOL: dict[str, int] = {}
# How long an idle thread of a session's own pool spins before it sleeps, beside
# another engine's CPU models (``session.intra_op.spin_duration_us``, ONNX
# Runtime 1.26 and later).
SHARED_PROCESS_SPIN_US = 1000


def cpu_threads(threads: int | None) -> int:
    """The intra-op pool size: as configured, else the CPUs this process may run on."""
    if threads:
        return threads
    affinity = getattr(os, "sched_getaffinity", None)
    return len(affinity(0)) if affinity else os.cpu_count() or 1


def shared_pool(threads: int) -> int | None:
    """The size of the process's shared CPU pool, created on first use; None if ORT started without it."""
    import onnxruntime as ort

    if "size" not in _SHARED_POOL:
        try:
            ort.set_global_thread_pool_sizes(threads, 1)
        except RuntimeError:
            _SHARED_POOL["size"] = 0
        else:
            _SHARED_POOL["size"] = threads
    return _SHARED_POOL["size"] or None


def shared_pool_size() -> int | None:
    """The shared CPU pool's size if it exists; never creates it."""
    return _SHARED_POOL.get("size") or None


def session_options(
    choice: ProviderChoice, threads: int | None, shared_cpu: bool = False
) -> Any:
    """Sequential execution with every graph optimization; GPU sessions never fall back to the CPU.

    CPU sessions run on the shared pool, unless ``shared_cpu`` (another
    engine's models serve the process's CPU too): then each gets its own pool
    of ``cpu_threads(threads)`` whose idle threads spin for
    ``SHARED_PROCESS_SPIN_US``. Once the shared pool exists every CPU session
    must use it.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.log_severity_level = 3
    size = cpu_threads(threads)
    pooled = not choice.gpu and (
        (not shared_cpu and shared_pool(size)) or _SHARED_POOL.get("size")
    )
    if pooled:
        options.use_per_session_threads = False
    else:
        options.intra_op_num_threads = size
        options.inter_op_num_threads = 1
        if not choice.gpu and shared_cpu:
            options.add_session_config_entry(
                "session.intra_op.spin_duration_us", str(SHARED_PROCESS_SPIN_US)
            )
    if choice.gpu:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return options
