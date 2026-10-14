"""ONNX Runtime execution providers per accelerator, and the session options the engine uses.

The CPU provider is validated. GPU providers run only when the installed
onnxruntime build has them, never fall back to the CPU for a node they cannot
run, and stay unvalidated until a record says otherwise.

Every CPU session has its own intra-op pool, sized to the configured threads
(else the CPUs the process may run on; ONNX Runtime's own default counts the
host's CPUs, not the cpuset) and capped by the spec's ``graph_threads``. A
process-wide shared pool can't be sized per graph, and graphs differ: on 16
cores Omni Nano's bare text graph serves four callers twice as fast on 12
threads as on 16 (15 workers plus four callers oversubscribe the cores),
while its image graph takes 109 ms on 16 threads and 163 ms on 8.

An idle pool's threads spin before they sleep. Unbounded (about 40 ms), they
share the cores with whatever runs next: Omni's audio graph after its CLAP
windows took 124 ms instead of 43, and a native forward after an ONNX Runtime
run doubled. Pools that stop spinning the moment a run returns cost Omni
13-24 % of its 4-caller throughput, and pools that never spin take Nano text
from 5.1 to 7.4 ms. So idle threads spin ``SPIN_US``, or the spec's
``graph_spin_us`` for a graph whose threads would sleep inside a run (Omni
Nano's text graph on 12 threads answered 0.3-0.4 ms faster with a 10 ms
spin), and at most ``NEIGHBOR_SPIN_US`` beside another engine's CPU models (a
2 ms spin added 1.8 ms to a native forward that followed; 1 ms added none).
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


# How long an idle intra-op thread spins before it sleeps (ONNX Runtime 1.26
# and later): alone or beside other ONNX Runtime models, and beside another
# engine's CPU models.
SPIN_ENTRY = "session.intra_op.spin_duration_us"
SPIN_US = 2000
NEIGHBOR_SPIN_US = 1000


def cpu_threads(threads: int | None, cap: int | None = None) -> int:
    """The intra-op pool size: as configured, else the CPUs this process may run on; at most ``cap``."""
    if threads:
        size = threads
    else:
        affinity = getattr(os, "sched_getaffinity", None)
        size = len(affinity(0)) if affinity else os.cpu_count() or 1
    return min(size, cap) if cap else size


def session_options(
    choice: ProviderChoice,
    threads: int,
    cpu_neighbors: bool = False,
    spin_us: int | None = None,
) -> Any:
    """Sequential execution with every graph optimization on a pool of ``threads``.

    Idle CPU threads spin ``spin_us`` (``SPIN_US`` when it is ``None``; 0
    never spins), at most ``NEIGHBOR_SPIN_US`` when another engine's models
    serve the process's CPU too. GPU sessions never fall back to the CPU.
    """
    import onnxruntime as ort

    if spin_us is not None and spin_us < 0:
        # ONNX Runtime reads a negative duration as its own unbounded default.
        raise ValueError(
            f"a graph's idle spin must be 0 or more microseconds, not {spin_us}"
        )
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.log_severity_level = 3
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    if not choice.gpu:
        spin = SPIN_US if spin_us is None else spin_us
        if cpu_neighbors:
            spin = min(spin, NEIGHBOR_SPIN_US)
        options.add_session_config_entry(SPIN_ENTRY, str(spin))
    if choice.gpu:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return options
