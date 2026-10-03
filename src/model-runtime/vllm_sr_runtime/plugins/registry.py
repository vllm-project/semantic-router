"""Entry-point discovery for families, engines, accelerators and profiles."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from typing import Any

GROUPS = {
    "families": "vllm_sr_runtime.families",
    "engines": "vllm_sr_runtime.engines",
    "accelerators": "vllm_sr_runtime.accelerators",
    "profiles": "vllm_sr_runtime.profiles",
}

# Built-in plugins, used when the distribution metadata is absent (for
# example when the package runs from a source tree without installation).
BUILTIN = {
    "families": {
        "decision1": "vllm_sr_runtime.families.decision1.family:Decision1Family",
        "decision2": "vllm_sr_runtime.families.decision2.family:Decision2Family",
        "multimodal_embedding": "vllm_sr_runtime.families.multimodal_embedding.family:MultimodalEmbeddingFamily",
        "task_heads": "vllm_sr_runtime.families.task_heads.family:TaskHeadsFamily",
        "vela2": "vllm_sr_runtime.families.vela2.family:Vela2Family",
    },
    "engines": {
        "native": "vllm_sr_runtime.engines.native.engine:NativeEngine",
        "onnxruntime": "vllm_sr_runtime.engines.onnxruntime.engine:OnnxRuntimeEngine",
    },
    "accelerators": {
        "cpu": "vllm_sr_runtime.accel.cpu:CPUAccelerator",
        "cuda": "vllm_sr_runtime.accel.cuda:CUDAAccelerator",
        "mps": "vllm_sr_runtime.accel.mps:MPSAccelerator",
        "rocm": "vllm_sr_runtime.accel.rocm:ROCmAccelerator",
        "xpu": "vllm_sr_runtime.accel.xpu:XPUAccelerator",
    },
    "profiles": {
        "exact": "vllm_sr_runtime.profiles.exact:ExactProfile",
        "shared_context": "vllm_sr_runtime.profiles.shared_context:SharedContextProfile",
        "batching": "vllm_sr_runtime.profiles.batching:BatchingProfile",
        "max_speed": "vllm_sr_runtime.profiles.max_speed:MaxSpeedProfile",
    },
}


@dataclass(frozen=True)
class PluginEntry:
    group: str
    name: str
    target: str
    distribution: str | None
    version: str | None

    def load(self) -> Any:
        module_name, _, attribute = self.target.partition(":")
        module = __import__(module_name, fromlist=[attribute])
        return getattr(module, attribute)

    def describe(self) -> dict[str, Any]:
        return {
            "group": GROUPS[self.group],
            "name": self.name,
            "distribution": self.distribution,
            "version": self.version,
            "capabilities": self.capabilities(),
        }

    def capabilities(self) -> dict[str, Any]:
        """The plugin class's capability descriptor; a plugin that fails to import reports why."""
        try:
            descriptor = getattr(self.load(), "descriptor", None)
            return dict(descriptor()) if callable(descriptor) else {}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}


class PluginConflictError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def discover() -> dict[str, dict[str, PluginEntry]]:
    """All plugins by kind; a name claimed by two distributions is an error."""
    found: dict[str, dict[str, PluginEntry]] = {kind: {} for kind in GROUPS}
    for kind, group in GROUPS.items():
        for entry in metadata.entry_points(group=group):
            distribution = getattr(entry, "dist", None)
            plugin = PluginEntry(
                group=kind,
                name=entry.name,
                target=entry.value,
                distribution=distribution.name if distribution else None,
                version=distribution.version if distribution else None,
            )
            existing = found[kind].get(entry.name)
            if existing and existing.target != plugin.target:
                raise PluginConflictError(
                    f"{group}: {entry.name!r} is registered by "
                    f"{existing.distribution} and {plugin.distribution}"
                )
            found[kind][entry.name] = plugin
        for name, target in BUILTIN[kind].items():
            found[kind].setdefault(
                name,
                PluginEntry(kind, name, target, "vllm-sr-runtime", None),
            )
    return found


def plugin(kind: str, name: str) -> PluginEntry:
    entries = discover()[kind]
    if name not in entries:
        raise KeyError(
            f"no {kind[:-1]} plugin named {name!r}; available: {', '.join(sorted(entries))}"
        )
    return entries[name]


def instantiate(kind: str, name: str) -> Any:
    return plugin(kind, name).load()()


def names(kind: str) -> list[str]:
    return sorted(discover()[kind])
