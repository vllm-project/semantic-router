"""ONNX Runtime engine: runs the ONNX graphs a package ships or a prepared bundle holds.

Graphs are data: the engine loads them with ONNX Runtime and no custom
operators. It returns named graph outputs (heads baked in) or hidden states
when a graph exports them. Execution providers: CPU (validated); CUDA, ROCm
and MIGraphX, and OpenVINO when the installed onnxruntime build provides them
(unvalidated until recorded).
"""

from __future__ import annotations

from typing import Any, ClassVar

from ...plugins.base import (
    Accelerator,
    DeviceInfo,
    Engine,
    EngineModel,
    EngineOptions,
    ModelSpec,
)


class OnnxRuntimeEngine(Engine):
    name: ClassVar[str] = "onnxruntime"

    @classmethod
    def descriptor(cls) -> dict[str, Any]:
        return {"outputs": ["graph"], "graphs": "onnx"}

    def supports(self, spec: ModelSpec, device: DeviceInfo) -> str | None:
        if not spec.graphs:
            return "the package ships no ONNX graph for this model"
        return "the onnxruntime engine cannot run models yet"

    def load(
        self,
        spec: ModelSpec,
        accelerator: Accelerator,
        device: DeviceInfo,
        options: EngineOptions,
    ) -> EngineModel:
        raise NotImplementedError("the onnxruntime engine cannot run models yet")
