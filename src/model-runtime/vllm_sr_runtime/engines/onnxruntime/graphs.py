"""Facts about an ONNX graph file read from its protobuf without ONNX or ONNX Runtime.

Only the model header is parsed: the custom metadata (``metadata_props``) and
every initializer's shape and storage, so loading never needs the ``onnx``
package and external weight files are never read. Graphs exported from one
checkpoint share an external weight file; their initializers are keyed by
storage, so a model's parameters count each stored tensor once.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from ...errors import PackageError

# ModelProto, GraphProto, TensorProto and StringStringEntryProto field numbers (onnx.proto).
_MODEL_GRAPH = 7
_MODEL_METADATA = 14
_GRAPH_INITIALIZER = 5
_TENSOR_DIMS = 1
_TENSOR_NAME = 8
_TENSOR_EXTERNAL_DATA = 13
_TENSOR_DATA_LOCATION = 14
_ENTRY_KEY = 1
_ENTRY_VALUE = 2
_EXTERNAL = 1

_VARINT, _FIXED64, _BYTES, _FIXED32 = 0, 1, 2, 5
_MAX_GRAPH_BYTES = 2 << 30
_MAX_VARINT_SHIFT = 63


@dataclass(frozen=True)
class GraphFacts:
    """A graph's custom metadata and its initializers' element counts by storage key."""

    metadata: dict[str, str]
    initializers: dict[tuple[str, ...], int]


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


def _initializer(tensor: memoryview, graph: Path) -> tuple[tuple[str, ...], int]:
    dims: list[int] = []
    name, external, location = "", {}, 0
    for number, wire, value in _fields(tensor):
        if number == _TENSOR_DIMS:
            if wire == _VARINT:
                dims.append(value)
            else:
                offset = 0
                while offset < len(value):
                    dim, offset = _varint(value, offset)
                    dims.append(dim)
        elif number == _TENSOR_NAME and wire == _BYTES:
            name = bytes(value).decode()
        elif number == _TENSOR_EXTERNAL_DATA and wire == _BYTES:
            key, item = _entry(value)
            external[key] = item
        elif number == _TENSOR_DATA_LOCATION and wire == _VARINT:
            location = value
    if location == _EXTERNAL:
        stored = str((graph.parent / external.get("location", "")).resolve())
        key = ("external", stored, external.get("offset", "0"))
    else:
        key = ("embedded", str(graph.resolve()), name)
    return key, math.prod(dims)


def read_graph(path: Path) -> GraphFacts:
    """Metadata and initializer storage of the ONNX graph at ``path``."""
    path = Path(path)
    if path.stat().st_size > _MAX_GRAPH_BYTES:
        raise PackageError(f"{path.name}: ONNX graph exceeds 2 GiB")
    data = memoryview(path.read_bytes())
    metadata: dict[str, str] = {}
    initializers: dict[tuple[str, ...], int] = {}
    for number, wire, value in _fields(data):
        if wire != _BYTES:
            continue
        if number == _MODEL_METADATA:
            key, item = _entry(value)
            metadata[key] = item
        elif number == _MODEL_GRAPH:
            for graph_field, graph_wire, item in _fields(value):
                if graph_field == _GRAPH_INITIALIZER and graph_wire == _BYTES:
                    key, elements = _initializer(item, path)
                    initializers[key] = elements
    return GraphFacts(metadata=metadata, initializers=initializers)
