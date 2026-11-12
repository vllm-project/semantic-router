"""Facts about an ONNX graph file read from its protobuf without ONNX or ONNX Runtime.

Only the model header is parsed: the custom metadata (``metadata_props``) and
every initializer's name, type, shape and storage, so loading never needs the
``onnx`` package. Graphs exported from one checkpoint share an external weight
file; initializers are keyed by storage, so a model's parameters count each
stored tensor once, and the engine maps each weight file once and hands its
tensors to every session (Hugging Face snapshots are links into a blob store,
which ONNX Runtime's own external-data loader refuses to follow).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ...errors import PackageError

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ModelProto, GraphProto, TensorProto and StringStringEntryProto field numbers (onnx.proto).
_MODEL_GRAPH = 7
_MODEL_METADATA = 14
_GRAPH_INITIALIZER = 5
_TENSOR_DIMS = 1
_TENSOR_DATA_TYPE = 2
_TENSOR_NAME = 8
_TENSOR_EXTERNAL_DATA = 13
_TENSOR_DATA_LOCATION = 14
_ENTRY_KEY = 1
_ENTRY_VALUE = 2
_EXTERNAL = 1

_VARINT, _FIXED64, _BYTES, _FIXED32 = 0, 1, 2, 5
_MAX_GRAPH_BYTES = 2 << 30
_MAX_VARINT_SHIFT = 63
# TensorProto.DataType values NumPy can hold.
DTYPES = {
    1: np.float32,
    2: np.uint8,
    3: np.int8,
    5: np.int16,
    6: np.int32,
    7: np.int64,
    9: np.bool_,
    10: np.float16,
    11: np.float64,
}


@dataclass(frozen=True)
class ExternalTensor:
    """An initializer stored in a weight file next to the graph."""

    name: str
    file: Path
    offset: int
    length: int
    dtype: type
    shape: tuple[int, ...]


@dataclass(frozen=True)
class GraphFacts:
    """A graph's custom metadata, its initializers' element counts by storage key, and its external tensors."""

    metadata: dict[str, str]
    initializers: dict[tuple[str, ...], int]
    externals: tuple[ExternalTensor, ...] = field(default=())


def parameters(graphs: Iterable[GraphFacts]) -> int:
    """Elements of every distinct stored initializer across ``graphs``."""
    merged: dict[tuple[str, ...], int] = {}
    for facts in graphs:
        merged.update(facts.initializers)
    return sum(merged.values())


def _varint(data: memoryview, offset: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        if offset >= len(data):
            raise PackageError("truncated ONNX protobuf")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift > _MAX_VARINT_SHIFT:
            raise PackageError("invalid ONNX protobuf varint")


def _fields(data: memoryview) -> Iterator[tuple[int, int, int | memoryview]]:
    """Yield ``(field number, wire type, value)``; length-delimited values as slices."""
    offset = 0
    while offset < len(data):
        key, offset = _varint(data, offset)
        number, wire = key >> 3, key & 7
        value: int | memoryview
        if wire == _VARINT:
            value, offset = _varint(data, offset)
        elif wire in (_FIXED64, _FIXED32):
            size = 8 if wire == _FIXED64 else 4
            value, offset = data[offset : offset + size], offset + size
        elif wire == _BYTES:
            size, offset = _varint(data, offset)
            if offset + size > len(data):
                raise PackageError("truncated ONNX protobuf")
            value, offset = data[offset : offset + size], offset + size
        else:
            raise PackageError(f"unsupported ONNX protobuf wire type {wire}")
        yield number, wire, value


def _entry(data: memoryview) -> tuple[str, str]:
    entry = {number: bytes(value).decode() for number, _, value in _fields(data)}
    return entry.get(_ENTRY_KEY, ""), entry.get(_ENTRY_VALUE, "")


def _external(
    graph: Path, name: str, data_type: int, dims: list[int], info: dict[str, str]
) -> ExternalTensor:
    """An external initializer whose bytes lie inside a file next to the graph.

    Without ``length`` the tensor runs to the end of the file (ONNX), as in
    exports that write one file per tensor.
    """
    location = info.get("location", "")
    relative = PurePosixPath(location)
    if (
        not location
        or relative.is_absolute()
        or ".." in relative.parts
        or "\\" in location
    ):
        raise PackageError(
            f"{graph.name}: external data of {name!r} must stay next to the graph"
        )
    if data_type not in DTYPES:
        raise PackageError(
            f"{graph.name}: unsupported external tensor type {data_type} for {name!r}"
        )
    dtype = DTYPES[data_type]
    path = graph.parent.joinpath(*relative.parts)
    offset = int(info.get("offset", 0))
    if "length" in info:
        length = int(info["length"])
    elif path.is_file():
        length = path.stat().st_size - offset
    else:
        raise PackageError(f"{graph.name}: external data file {location!r} is missing")
    if length != math.prod(dims) * np.dtype(dtype).itemsize:
        raise PackageError(
            f"{graph.name}: external data of {name!r} has the wrong length"
        )
    return ExternalTensor(name, path, offset, length, dtype, tuple(dims))


def _initializer(
    tensor: memoryview, graph: Path
) -> tuple[tuple[str, ...], int, ExternalTensor | None]:
    dims: list[int] = []
    name, external, location, data_type = "", {}, 0, 0
    for number, wire, value in _fields(tensor):
        if number == _TENSOR_DIMS:
            if wire == _VARINT:
                dims.append(cast(int, value))
            else:
                packed = cast(memoryview, value)
                offset = 0
                while offset < len(packed):
                    dim, offset = _varint(packed, offset)
                    dims.append(dim)
        elif number == _TENSOR_DATA_TYPE and wire == _VARINT:
            data_type = cast(int, value)
        elif number == _TENSOR_NAME and wire == _BYTES:
            name = bytes(value).decode()
        elif number == _TENSOR_EXTERNAL_DATA and wire == _BYTES:
            key, item = _entry(cast(memoryview, value))
            external[key] = item
        elif number == _TENSOR_DATA_LOCATION and wire == _VARINT:
            location = cast(int, value)
    if location != _EXTERNAL:
        return ("embedded", str(graph.resolve()), name), math.prod(dims), None
    tensor_info = _external(graph, name, data_type, dims, external)
    storage = ("external", str(tensor_info.file.resolve()), str(tensor_info.offset))
    return storage, math.prod(dims), tensor_info


def read_graph(path: Path) -> GraphFacts:
    """Metadata, initializer storage and external tensors of the ONNX graph at ``path``."""
    path = Path(path)
    if path.stat().st_size > _MAX_GRAPH_BYTES:
        raise PackageError(f"{path.name}: ONNX graph exceeds 2 GiB")
    data = memoryview(path.read_bytes())
    metadata: dict[str, str] = {}
    initializers: dict[tuple[str, ...], int] = {}
    externals: list[ExternalTensor] = []
    for number, wire, value in _fields(data):
        if wire != _BYTES:
            continue
        if number == _MODEL_METADATA:
            key, item = _entry(cast(memoryview, value))
            metadata[key] = item
        elif number == _MODEL_GRAPH:
            for graph_field, graph_wire, initializer in _fields(
                cast(memoryview, value)
            ):
                if graph_field == _GRAPH_INITIALIZER and graph_wire == _BYTES:
                    storage, elements, tensor = _initializer(
                        cast(memoryview, initializer), path
                    )
                    initializers[storage] = elements
                    if tensor is not None:
                        externals.append(tensor)
    return GraphFacts(
        metadata=metadata, initializers=initializers, externals=tuple(externals)
    )


class WeightFiles:
    """Read-only maps of a model's external weight files, each mapped once and shared by its graphs."""

    def __init__(self) -> None:
        self._maps: dict[Path, np.memmap[Any, np.dtype[np.uint8]]] = {}

    def tensor(self, external: ExternalTensor) -> NDArray[Any]:
        path = external.file.resolve()
        mapped = self._maps.get(path)
        if mapped is None:
            if not path.is_file():
                raise PackageError(f"missing external weight file {external.file.name}")
            mapped = self._maps[path] = np.memmap(path, dtype=np.uint8, mode="r")
        end = external.offset + external.length
        if external.offset < 0 or end > mapped.shape[0]:
            raise PackageError(
                f"external data of {external.name!r} lies outside {external.file.name}"
            )
        raw = mapped[external.offset : end]
        if external.offset % np.dtype(external.dtype).itemsize:
            raw = raw.copy()
        return raw.view(external.dtype).reshape(external.shape)
