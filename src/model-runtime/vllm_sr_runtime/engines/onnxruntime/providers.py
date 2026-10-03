"""ONNX Runtime execution providers per accelerator, and the session options the engine uses.

The CPU provider is validated. GPU providers run only when the installed
onnxruntime build has them, never fall back to the CPU for a node they cannot
run, and stay unvalidated until a record says otherwise.
"""

from __future__ import annotations

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


def session_options(
    choice: ProviderChoice, threads: int | None, spinning: bool | None = None
) -> Any:
    """Sequential execution with every graph optimization; GPU sessions never fall back to the CPU."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    if threads:
        options.intra_op_num_threads = threads
    if spinning is not None:
        options.add_session_config_entry(
            "session.intra_op.allow_spinning", "1" if spinning else "0"
        )
    if choice.gpu:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return options
