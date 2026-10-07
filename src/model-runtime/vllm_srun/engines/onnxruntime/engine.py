"""ONNX Runtime engine: runs the ONNX graphs a package ships or a prepared bundle holds.

Graphs are data: the engine loads them with ONNX Runtime and no custom
operators, and returns named graph outputs (heads baked in, or hidden states
when a graph exports them). Each ``ModelSpec.graphs`` entry becomes one
session; an ``EncoderBatch`` names the graph it runs. The standard token
inputs (``input_ids``, ``attention_mask``, ``position_ids``,
``token_type_ids``) are filled from the batch, any other declared input comes
from ``EncoderBatch.graph_inputs``.

Execution providers (``providers.py``): CPU (validated); CUDA, MIGraphX or
ROCm, and OpenVINO when the installed onnxruntime build provides them
(unvalidated until recorded).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch

from ...plugins.base import (
    Accelerator,
    DeviceInfo,
    EncoderBatch,
    EncoderOutput,
    Engine,
    EngineModel,
    EngineOptions,
    ForwardBatch,
    ForwardOutput,
    ModelSpec,
)
from . import graphs as graph_files
from . import providers

if TYPE_CHECKING:
    from numpy.typing import NDArray

INSTALL_HINT = (
    "install the runtime's onnx extra from a repository checkout: "
    "pip install './src/model-runtime[onnx]'"
)
NUMPY_TYPES = {
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(bool)": np.bool_,
}


@dataclass(frozen=True)
class GraphInput:
    name: str
    shape: tuple[int | str | None, ...]
    dtype: Any


class GraphSession:
    """One graph loaded in an ONNX Runtime session, with its declared inputs and outputs.

    External initializers come from the model's shared weight-file maps, so
    the graph loads from bytes and ONNX Runtime never resolves file links.
    """

    def __init__(
        self,
        name: str,
        path: Path,
        options: Any,
        provider: providers.ProviderChoice,
        weights: graph_files.WeightFiles,
    ):
        import onnxruntime

        self.name = name
        self.path = path
        self.spin_us = (
            None
            if provider.gpu
            else int(options.get_session_config_entry(providers.SPIN_ENTRY))
        )
        self.facts = graph_files.read_graph(path)
        if self.facts.externals:
            # ONNX Runtime keeps the arrays referenced, not copied; they live as long as the session.
            self.initializers = [
                onnxruntime.OrtValue.ortvalue_from_numpy(weights.tensor(tensor))
                for tensor in self.facts.externals
            ]
            options.add_external_initializers(
                [tensor.name for tensor in self.facts.externals], self.initializers
            )
            source: Any = path.read_bytes()
        else:
            source = str(path)
        self.session = onnxruntime.InferenceSession(
            source, options, providers=[(provider.name, provider.options)]
        )
        self.inputs = tuple(
            GraphInput(
                item.name, tuple(item.shape), NUMPY_TYPES.get(item.type, np.float32)
            )
            for item in self.session.get_inputs()
        )
        self.outputs = tuple(item.name for item in self.session.get_outputs())

    def feeds(self, batch: EncoderBatch) -> dict[str, NDArray[Any]]:
        input_ids = batch.input_ids.detach().cpu().numpy()
        rows, width = input_ids.shape
        feeds: dict[str, NDArray[Any]] = {}
        for declared in self.inputs:
            if declared.name in batch.graph_inputs:
                value = batch.graph_inputs[declared.name].detach().cpu().numpy()
            elif declared.name == "input_ids":
                value = input_ids
            elif declared.name == "attention_mask":
                assert batch.attention_mask is not None
                value = batch.attention_mask.detach().cpu().numpy()
            elif declared.name == "position_ids":
                positions = np.arange(width, dtype=np.int64)[None]
                value = (
                    positions
                    if declared.shape[:1] == (1,)
                    else np.repeat(positions, rows, 0)
                )
            elif declared.name == "token_type_ids":
                value = np.zeros_like(input_ids)
            else:
                raise ValueError(
                    f"graph {self.name!r} needs the input {declared.name!r}"
                )
            feeds[declared.name] = np.ascontiguousarray(value, dtype=declared.dtype)
        return feeds

    def run(self, batch: EncoderBatch) -> dict[str, torch.Tensor]:
        names = list(batch.outputs) or list(self.outputs)
        unknown = sorted(set(names) - set(self.outputs))
        if unknown:
            raise ValueError(f"graph {self.name!r} has no outputs {unknown}")
        values = self.session.run(names, self.feeds(batch))
        return {
            name: torch.from_numpy(value)
            for name, value in zip(names, values, strict=True)
        }


class OnnxRuntimeModel(EngineModel):
    """A model's graphs loaded on one device; ``encode`` runs the graph a batch names."""

    hidden_states: ClassVar[bool] = False

    def __init__(
        self,
        graphs: dict[str, GraphSession],
        choice: providers.ProviderChoice,
        device_info: DeviceInfo,
        threads: dict[str, int],
    ):
        self.graphs = graphs
        self.choice = choice
        self.device_info = device_info
        self.device = torch.device("cpu")
        self.threads = threads

    def forward(self, batch: ForwardBatch) -> ForwardOutput:
        raise NotImplementedError("the onnxruntime engine runs encoder graphs only")

    def encode(self, batch: EncoderBatch) -> EncoderOutput:
        graph = self.graphs.get(batch.graph)
        if graph is None:
            raise ValueError(
                f"no graph {batch.graph!r} is loaded; loaded: {sorted(self.graphs)}"
            )
        return EncoderOutput(outputs=graph.run(batch))

    def parameter_count(self) -> int:
        return graph_files.parameters(graph.facts for graph in self.graphs.values())

    def memory_bytes(self) -> int:
        return 4 * self.parameter_count()

    def receipt(self) -> dict[str, Any]:
        """What runs this model: the provider, its options, the loaded graphs and their CPU pools."""
        return {
            "provider": self.choice.name,
            "provider_options": dict(self.choice.options),
            "validated": self.choice.validated,
            "threads": self.threads,
            "spin_us": {
                name: graph.spin_us
                for name, graph in self.graphs.items()
                if graph.spin_us is not None
            },
            "graphs": {name: graph.path.name for name, graph in self.graphs.items()},
        }

    def close(self) -> None:
        self.graphs = {}


class OnnxRuntimeEngine(Engine):
    name: ClassVar[str] = "onnxruntime"

    @classmethod
    def descriptor(cls) -> dict[str, Any]:
        try:
            import onnxruntime

            installed = onnxruntime.get_available_providers()
        except ImportError:
            installed = []
        return {
            **super().descriptor(),
            "architectures": ["onnx-graph"],
            "outputs": ["graph_outputs"],
            "devices": sorted(providers.PROVIDERS),
            "providers": {
                name: list(order) for name, order in providers.PROVIDERS.items()
            },
            "validated": sorted(providers.VALIDATED),
            "installed": installed,
        }

    def supports(self, spec: ModelSpec, device: DeviceInfo) -> str | None:
        if not spec.graphs:
            return "the package ships no ONNX graph for this model"
        try:
            import onnxruntime
        except ImportError:
            return f"onnxruntime is not installed ({INSTALL_HINT})"
        choice = providers.choose(device, onnxruntime.get_available_providers())
        return choice if isinstance(choice, str) else None

    def load(
        self,
        spec: ModelSpec,
        accelerator: Accelerator,
        device: DeviceInfo,
        options: EngineOptions,
    ) -> OnnxRuntimeModel:
        import onnxruntime

        choice = providers.choose(
            device,
            onnxruntime.get_available_providers(),
            options.extra.get("onnxruntime_provider"),
        )
        if isinstance(choice, str):
            raise RuntimeError(choice)
        neighbors = bool(options.cpu_neighbors - {self.name})
        threads = {
            name: providers.cpu_threads(
                options.threads, None if choice.gpu else spec.graph_threads.get(name)
            )
            for name in spec.graphs
        }
        weights = graph_files.WeightFiles()
        graphs = {}
        for name, path in spec.graphs.items():
            session = providers.session_options(
                choice, threads[name], neighbors, spec.graph_spin_us.get(name)
            )
            graphs[name] = GraphSession(name, Path(path), session, choice, weights)
        return OnnxRuntimeModel(graphs, choice, device, threads)
