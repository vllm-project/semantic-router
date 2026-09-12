"""Pack exact ONNX tensor storage into one confined, shared external-data blob.

Models retain every graph operation and tensor dtype/shape/value. The destination
must be new. Input models and their external files are never modified. Only
existing external numeric storage is consolidated. Every originally inline tensor
is preserved verbatim: ORT shape inference requires some constants to stay inline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path, PureWindowsPath

import onnx
from onnx import TensorProto, numpy_helper

CHUNK_BYTES = 8 * 1024 * 1024
ALIGNMENT = 64
BLOB_NAME = "weights.data"
ITEM_BYTES = {
    TensorProto.FLOAT: 4,
    TensorProto.FLOAT16: 2,
    TensorProto.DOUBLE: 8,
    TensorProto.BFLOAT16: 2,
    TensorProto.INT8: 1,
    TensorProto.UINT8: 1,
    TensorProto.INT16: 2,
    TensorProto.UINT16: 2,
    TensorProto.INT32: 4,
    TensorProto.UINT32: 4,
    TensorProto.INT64: 8,
    TensorProto.UINT64: 8,
    TensorProto.BOOL: 1,
    TensorProto.COMPLEX64: 8,
    TensorProto.COMPLEX128: 16,
}
STORAGE_FIELDS = (
    "raw_data",
    "float_data",
    "int32_data",
    "string_data",
    "int64_data",
    "double_data",
    "uint64_data",
    "external_data",
    "data_location",
)


def tensors(message, prefix="model"):
    """Visit all tensor values, including node attributes and nested graphs."""
    if isinstance(message, TensorProto):
        yield prefix, message
        return
    for field, value in message.ListFields():
        if field.message_type is None:
            continue
        if field.is_repeated:
            for index, child in enumerate(value):
                yield from tensors(child, f"{prefix}.{field.name}[{index}]")
        else:
            yield from tensors(value, f"{prefix}.{field.name}")


def _snapshot(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _safe_file(directory, location):
    directory = directory.resolve()
    path, windows = Path(location), PureWindowsPath(location)
    resolved = (directory / path).resolve()
    if (
        not location
        or path.is_absolute()
        or windows.drive
        or ".." in path.parts
        or ".." in windows.parts
        or not resolved.is_relative_to(directory)
    ):
        raise ValueError("External data must remain inside its graph directory")
    if not resolved.is_file():
        raise ValueError("External tensor file is missing")
    return resolved


def _storage(tensor, graph_path, snapshots):
    """Return raw bytes or a bounded file range, without loading large tensors."""
    if tensor.HasField("segment"):
        raise ValueError("Segmented tensor storage is not supported")
    if tensor.data_type == TensorProto.STRING:
        if tensor.external_data or tensor.data_location == TensorProto.EXTERNAL:
            raise ValueError("External string tensor storage is unsupported")
        return None
    if tensor.data_type not in ITEM_BYTES:
        raise ValueError(f"Unsupported exact-storage tensor dtype {tensor.data_type}")
    expected = math.prod(tensor.dims) * ITEM_BYTES[tensor.data_type]
    if any(dim < 0 for dim in tensor.dims):
        raise ValueError("Tensor dimensions must be nonnegative")
    if tensor.data_location == TensorProto.EXTERNAL:
        entries = {entry.key: entry.value for entry in tensor.external_data}
        if len(entries) != len(tensor.external_data) or set(entries) - {
            "location",
            "offset",
            "length",
        }:
            raise ValueError("Unknown or duplicate external tensor metadata")
        source = _safe_file(graph_path.parent, entries.get("location", ""))
        if any(
            tensor.HasField(name) if name == "raw_data" else len(getattr(tensor, name))
            for name in STORAGE_FIELDS[:7]
        ):
            raise ValueError("Tensor contains both external and inline storage")
        try:
            offset = int(entries.get("offset", "0"))
            length = int(entries.get("length", str(expected)))
        except ValueError as error:
            raise ValueError("Invalid external tensor offset/length") from error
        if offset < 0 or length != expected or offset + length > source.stat().st_size:
            raise ValueError("External tensor range does not match dtype/shape")
        snapshots.setdefault(source, _snapshot(source))
        if expected == 0:
            raise ValueError("External zero-length tensor is ambiguous; keep it inline")
        return source, offset, length
    if tensor.external_data:
        raise ValueError("External entries without EXTERNAL data_location")
    if expected == 0:
        return None
    raw = (
        tensor.raw_data
        if tensor.HasField("raw_data")
        else numpy_helper.to_array(tensor).tobytes(order="C")
    )
    if len(raw) != expected:
        raise ValueError("Inline tensor bytes do not match dtype/shape")
    return raw


def _chunks(storage):
    if isinstance(storage, bytes):
        for offset in range(0, len(storage), CHUNK_BYTES):
            yield storage[offset : offset + CHUNK_BYTES]
        return
    path, offset, remaining = storage
    with path.open("rb") as handle:
        handle.seek(offset)
        while remaining:
            block = handle.read(min(CHUNK_BYTES, remaining))
            if not block:
                raise ValueError("External tensor was truncated while packing")
            remaining -= len(block)
            yield block


def _digest(storage):
    result = hashlib.sha256()
    for chunk in _chunks(storage):
        result.update(chunk)
    return result.hexdigest()


def _clear_storage(tensor):
    for name in STORAGE_FIELDS:
        tensor.ClearField(name)


def _structure(model):
    """Serialize all non-storage fields as an exact graph-identity check."""
    copy = onnx.ModelProto()
    copy.CopyFrom(model)
    for _, tensor in tensors(copy):
        _clear_storage(tensor)
    return copy.SerializeToString(deterministic=True)


def _flat_name(name):
    return (
        isinstance(name, str)
        and name.endswith(".onnx")
        and Path(name).name == name
        and not PureWindowsPath(name).drive
        and "\\" not in name
        and name not in {".", ".."}
    )


def pack_models(sources: dict[str, Path], destination: Path) -> dict:
    """Create a new flat graph directory, verify every range, and return its receipt."""
    if (
        not sources
        or any(not _flat_name(name) for name in sources)
        or len({name.casefold() for name in sources}) != len(sources)
    ):
        raise ValueError("Each output graph needs a unique flat .onnx filename")
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Destination must not exist")
    sources = {name: Path(path).absolute() for name, path in sources.items()}
    if any(not path.is_file() or path.is_symlink() for path in sources.values()):
        raise ValueError("Input graphs must be ordinary existing files")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()  # exclusive reservation: never replace a caller's directory
    reservation = _snapshot(destination)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-packing-", dir=destination.parent)
    )
    snapshots, unique, cached, records = {}, {}, {}, []
    tensor_bytes = 0
    try:
        with (staging / BLOB_NAME).open("xb") as blob:
            for output_name, source in sorted(sources.items()):
                snapshots[source] = _snapshot(source)
                model = onnx.load(source, load_external_data=False)
                structure = _structure(model)
                tensor_records, inline_records = [], []
                for tensor_path, tensor in tensors(model):
                    if (
                        tensor.data_location != TensorProto.EXTERNAL
                        and not tensor.external_data
                    ):
                        inline_records.append(
                            {
                                "path": tensor_path,
                                "name": tensor.name,
                                "tensor_proto_sha256": hashlib.sha256(
                                    tensor.SerializeToString()
                                ).hexdigest(),
                            }
                        )
                        continue
                    storage = _storage(tensor, source, snapshots)
                    if storage is None:
                        continue
                    length = len(storage) if isinstance(storage, bytes) else storage[2]
                    cache_key = storage if not isinstance(storage, bytes) else None
                    digest = cached.get(cache_key) if cache_key is not None else None
                    if digest is None:
                        digest = _digest(storage)
                        if cache_key is not None:
                            cached[cache_key] = digest
                    identity = tensor.data_type, tuple(tensor.dims), length, digest
                    if identity not in unique:
                        padding = (-blob.tell()) % ALIGNMENT
                        blob.write(b"\0" * padding)
                        offset = blob.tell()
                        copied = hashlib.sha256()
                        for block in _chunks(storage):
                            blob.write(block)
                            copied.update(block)
                        if copied.hexdigest() != digest:
                            raise ValueError("Source tensor changed while copying")
                        unique[identity] = offset
                    offset = unique[identity]
                    tensor_records.append(
                        {
                            "path": tensor_path,
                            "name": tensor.name,
                            "dtype": tensor.data_type,
                            "shape": list(tensor.dims),
                            "offset": offset,
                            "length": length,
                            "sha256": digest,
                        }
                    )
                    tensor_bytes += length
                    _clear_storage(tensor)
                    tensor.data_location = TensorProto.EXTERNAL
                    for key, value in [
                        ("location", BLOB_NAME),
                        ("offset", str(offset)),
                        ("length", str(length)),
                    ]:
                        entry = tensor.external_data.add()
                        entry.key, entry.value = key, value
                if _structure(model) != structure:
                    raise AssertionError("Packing changed non-storage graph fields")
                output = staging / output_name
                output.write_bytes(model.SerializeToString())
                records.append(
                    {
                        "graph": output_name,
                        "source_graph_sha256": _digest(
                            (source, 0, source.stat().st_size)
                        ),
                        "non_storage_structure_sha256": hashlib.sha256(
                            structure
                        ).hexdigest(),
                        "tensors": tensor_records,
                        "inline_tensors": inline_records,
                    }
                )
            blob.flush()
            os.fsync(blob.fileno())
        checked = {}
        for record in records:
            graph_path = staging / record["graph"]
            packed = onnx.load(graph_path, load_external_data=False)
            expected = {item["path"]: item for item in record["tensors"]}
            inline_expected = {item["path"]: item for item in record["inline_tensors"]}
            for tensor_path, tensor in tensors(packed):
                if tensor_path in inline_expected:
                    if (
                        hashlib.sha256(tensor.SerializeToString()).hexdigest()
                        != inline_expected[tensor_path]["tensor_proto_sha256"]
                    ):
                        raise AssertionError("Originally inline tensor changed")
                    continue
                if tensor_path not in expected:
                    continue
                item = expected[tensor_path]
                storage = _storage(tensor, graph_path, {})
                if (
                    not isinstance(storage, tuple)
                    or storage[0] != (staging / BLOB_NAME).resolve()
                ):
                    raise AssertionError("Packed tensor escaped the shared blob")
                key = storage[1:]
                if key not in checked:
                    checked[key] = _digest(storage)
                if (
                    checked[key] != item["sha256"]
                    or storage[1:] != (item["offset"], item["length"])
                    or tensor.data_type != item["dtype"]
                    or list(tensor.dims) != item["shape"]
                ):
                    raise AssertionError("Packed tensor round-trip differs")
            onnx.checker.check_model(str(graph_path))
            record["graph_sha256"] = _digest((graph_path, 0, graph_path.stat().st_size))
        if any(_snapshot(path) != before for path, before in snapshots.items()):
            raise ValueError("Source files changed during packing")
        blob_path = staging / BLOB_NAME
        receipt = {
            "format": "shared-onnx-weights-v2",
            "graph_count": len(records),
            "tensor_occurrences": sum(
                len(x["tensors"]) + len(x["inline_tensors"]) for x in records
            ),
            "external_tensor_occurrences": sum(len(x["tensors"]) for x in records),
            "preserved_inline_tensor_occurrences": sum(
                len(x["inline_tensors"]) for x in records
            ),
            "unique_tensors": len(unique),
            "tensor_bytes_before_deduplication": tensor_bytes,
            "unique_tensor_bytes": sum(key[2] for key in unique),
            "shared_blob_bytes": blob_path.stat().st_size,
            "shared_blob_sha256": _digest((blob_path, 0, blob_path.stat().st_size)),
            "alignment": ALIGNMENT,
            "storage_location": BLOB_NAME,
            "all_tensor_ranges_round_trip_verified": True,
            "non_storage_graph_fields_unchanged": True,
            "numeric_engine_qualification": False,
            "graphs": records,
        }
        (staging / "packing-manifest.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        if _snapshot(destination) != reservation or any(destination.iterdir()):
            raise ValueError("Reserved destination changed while packing")
        os.replace(staging, destination)
        return receipt
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if (
            destination.exists()
            and _snapshot(destination) == reservation
            and not any(destination.iterdir())
        ):
            destination.rmdir()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-manifest",
        type=Path,
        required=True,
        help="JSON object mapping flat output .onnx names to input paths (relative to this manifest)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mapping = json.loads(args.input_manifest.read_text())
    if not isinstance(mapping, dict) or any(
        not isinstance(v, str) for v in mapping.values()
    ):
        raise ValueError("Input manifest must map output filenames to source paths")
    receipt = pack_models(
        {name: args.input_manifest.parent / path for name, path in mapping.items()},
        args.output,
    )
    print(json.dumps({key: value for key, value in receipt.items() if key != "graphs"}))


if __name__ == "__main__":
    main()
