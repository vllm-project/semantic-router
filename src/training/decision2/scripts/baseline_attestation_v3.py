"""Build and verify private, byte-complete JevArena v3 baseline attestations.

This is a CPU-only inventory tool. It counts tensors stored in an
inference-only safetensors package; it does not execute model code or inspect
evaluation labels. A model whose native loader omits or adds weights relative
to that package needs a separate runtime count before it can be attested.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import stat
import struct
import subprocess
from pathlib import Path
from typing import Any

SCHEMA = "decision2-baseline-package-attestation/1"
SHA = re.compile(r"[0-9a-f]{64}\Z")
MAX_HEADER = 64 << 20
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
RECEIPT_FIELDS = {
    "schema_version",
    "model_id",
    "revision",
    "native_model_sha256",
    "adapter_sha256",
    "calibration_sha256",
    "calibration_path",
    "package_path",
    "files",
    "weight_files",
    "parameter_count",
    "loaded_parameter_count",
    "size_b",
}
KEV_RECEIPT_FIELDS = {
    "base_package_path",
    "base_files",
    "base_weight_files",
    "base_parameter_count",
    "base_text_parameter_count",
    "adapter_parameter_count",
    "head_parameter_count",
    "native_loaded_count_receipt_path",
    "native_loaded_count_receipt_sha256",
}
KEV_BASE_TEXT_PARAMETERS = 4_205_751_296
KEV_BASE_ALL_PARAMETERS = 4_659_865_088
KEV_HEAD_PARAMETERS = 1_311_232
KEV_LOADED_PARAMETERS = KEV_BASE_TEXT_PARAMETERS + KEV_HEAD_PARAMETERS
ATTESTATION_FIELDS = {
    "key",
    "model_id",
    "revision",
    "size_b",
    "native_model_sha256",
    "adapter_sha256",
    "calibration_sha256",
    "receipt_path",
    "receipt_sha256",
}


class _QuietArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.exit(2, "Baseline attestation failed; inspect private inputs locally.\n")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON key")
        value[key] = item
    return value


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(_read_regular(path), object_pairs_hook=_no_duplicates)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _absolute_regular(path: Path) -> None:
    if not path.is_absolute():
        raise ValueError("Path must be absolute")
    for part in (path, *path.parents):
        if stat.S_ISLNK(part.lstat().st_mode):
            raise ValueError("Linked path component is forbidden")
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("Expected a regular file")


def _read_regular(path: Path) -> bytes:
    _absolute_regular(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("Expected a regular file")
        with os.fdopen(fd, "rb", closefd=False) as source:
            return source.read()
    finally:
        os.close(fd)


def _sha_file(path: Path) -> str:
    _absolute_regular(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("Expected a regular file")
        digest = hashlib.sha256()
        with os.fdopen(fd, "rb", closefd=False) as source:
            for chunk in iter(lambda: source.read(8 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(fd)


def _tree(root: Path) -> dict[str, str]:
    if not root.is_absolute() or stat.S_ISLNK(root.lstat().st_mode):
        raise ValueError("Package/runtime root must be an unlinked absolute directory")
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError("Package/runtime root must be a directory")
    files: dict[str, str] = {}

    def walk(directory: Path) -> None:
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
        for entry in entries:
            mode = entry.stat(follow_symlinks=False).st_mode
            if stat.S_ISLNK(mode):
                raise ValueError("Package/runtime contains a symlink")
            child = Path(entry.path)
            if stat.S_ISDIR(mode):
                walk(child)
            elif stat.S_ISREG(mode):
                relative = child.relative_to(root).as_posix()
                files[relative] = _sha_file(child)
            else:
                raise ValueError("Package/runtime contains a special file")

    walk(root)
    if not files:
        raise ValueError("Package/runtime is empty")
    return files


def _tensor_count(
    path: Path, names: set[str], *, selected_prefix: str | None = None
) -> int:
    _absolute_regular(path)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        size = os.fstat(fd).st_size
        with os.fdopen(fd, "rb", closefd=False) as source:
            prefix = source.read(8)
            if len(prefix) != 8:
                raise ValueError("Invalid safetensors header")
            header_size = struct.unpack("<Q", prefix)[0]
            if not 0 < header_size <= MAX_HEADER or 8 + header_size > size:
                raise ValueError("Invalid safetensors header length")
            header = json.loads(
                source.read(header_size), object_pairs_hook=_no_duplicates
            )
        if not isinstance(header, dict):
            raise ValueError("Invalid safetensors header")
        total = 0
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if name in names:
                raise ValueError("Duplicate tensor in weight shards")
            if not isinstance(tensor, dict) or set(tensor) != {
                "dtype",
                "shape",
                "data_offsets",
            }:
                raise ValueError("Invalid safetensors tensor record")
            shape, offsets = tensor["shape"], tensor["data_offsets"]
            if (
                not isinstance(shape, list)
                or any(type(dim) is not int or dim < 0 for dim in shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(offset) is not int for offset in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= size - 8 - header_size
                or not isinstance(tensor["dtype"], str)
                or tensor["dtype"] not in DTYPE_BYTES
            ):
                raise ValueError("Invalid safetensors tensor shape or offsets")
            count = 1
            for dim in shape:
                count *= dim
            if offsets[1] - offsets[0] != count * DTYPE_BYTES[tensor["dtype"]]:
                raise ValueError("Safetensors data span differs from tensor shape")
            if selected_prefix is None or name.startswith(selected_prefix):
                total += count
            names.add(name)
        return total
    finally:
        os.close(fd)


def _weights(
    package: Path, files: dict[str, str], *, selected_prefix: str | None = None
) -> tuple[list[str], int]:
    weight_files = sorted(name for name in files if name.endswith(".safetensors"))
    if not weight_files:
        raise ValueError("Inference package has no safetensors weights")
    names: set[str] = set()
    count = sum(
        _tensor_count(package / name, names, selected_prefix=selected_prefix)
        for name in weight_files
    )
    if count <= 0:
        raise ValueError("Inference package has no model parameters")
    return weight_files, count


def _sha(value: Any) -> bool:
    return isinstance(value, str) and SHA.fullmatch(value) is not None


def _private_input(path: Path) -> None:
    _absolute_regular(path)
    info = path.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("Native loader count receipt is not private")


def _hf_local_revision(package: Path, revision: str) -> None:
    """Require every local-dir metadata record to name the pinned revision."""
    metadata = package / ".cache/huggingface/download"
    if not metadata.is_dir() or stat.S_ISLNK(metadata.lstat().st_mode):
        raise ValueError("Pinned Hugging Face local-dir metadata is missing")
    files = sorted(metadata.rglob("*.metadata"))
    if not files:
        raise ValueError("Pinned Hugging Face local-dir metadata is missing")
    for path in files:
        lines = _read_regular(path).decode("utf-8").splitlines()
        if not lines or lines[0].strip() != revision:
            raise ValueError("Hugging Face local-dir revision differs")


def _kev_head_info(path: Path, base_id: str, base_revision: str) -> tuple[int, float]:
    # The published head is a small torch-serialized metadata dictionary.
    # weights_only avoids executing package pickle code during this CPU audit.
    import torch

    head = torch.load(
        io.BytesIO(_read_regular(path)), map_location="cpu", weights_only=True
    )
    if not isinstance(head, dict) or not isinstance(head.get("head"), dict):
        raise ValueError("Kev head metadata is malformed")
    if head.get("base") != base_id or head.get("base_revision") != base_revision:
        raise ValueError("Kev head base differs from pinned base")
    temperature = head.get("temperature")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("Kev head calibration temperature is invalid")
    tensors = head["head"]
    if not tensors or any(
        not isinstance(value, torch.Tensor) for value in tensors.values()
    ):
        raise ValueError("Kev head has missing or non-tensor weights")
    return sum(value.numel() for value in tensors.values()), float(temperature)


def _kev_receipt(
    model: Any,
    source_root: Path,
    model_root: Path,
    external_root: Path,
    calibration_path: Path | None,
    loaded_parameter_count: int,
    native_loaded_count_receipt_path: Path | None,
) -> dict[str, Any]:
    from inference.kev import KEV_BASE_ID, KEV_BASE_REVISION, KEV_SOURCE_REVISION

    if native_loaded_count_receipt_path is None:
        raise ValueError("Kev needs the native loader count receipt")
    _private_input(native_loaded_count_receipt_path)
    count_record = _json_object(native_loaded_count_receipt_path)
    package = model_root / model.model_dir
    base_package = model_root / "Qwen3.5-4B-Base"
    runtime = external_root / model.source_dir
    files, base_files, runtime_files = (
        _tree(package),
        _tree(base_package),
        _tree(runtime),
    )
    _hf_local_revision(package, model.revision)
    _hf_local_revision(base_package, KEV_BASE_REVISION)
    weight_files, adapter_count = _weights(package, files)
    base_weight_files, base_all_count = _weights(base_package, base_files)
    _, base_text_count = _weights(
        base_package, base_files, selected_prefix="model.language_model."
    )
    head_count, head_temperature = _kev_head_info(
        package / "head.pt", KEV_BASE_ID, KEV_BASE_REVISION
    )
    if (
        base_all_count != KEV_BASE_ALL_PARAMETERS
        or base_text_count != KEV_BASE_TEXT_PARAMETERS
        or head_count != KEV_HEAD_PARAMETERS
        or type(loaded_parameter_count) is not int
        or loaded_parameter_count != KEV_LOADED_PARAMETERS
        or base_text_count + head_count != loaded_parameter_count
    ):
        raise ValueError("Kev native loaded count differs from pinned components")
    if adapter_count <= 0:
        raise ValueError("Kev adapter weights are empty")
    count_runtime = count_record.get("runtime")
    if (
        count_record.get("schema_version")
        != "decision2-native-loaded-parameter-count/1"
        or count_record.get("model_id") != model.model_id
        or count_record.get("revision") != model.revision
        or count_record.get("count_method")
        != "sum(p.numel() for p in inference.kev.load_native(...)[0].parameters())"
        or count_record.get("parameter_count") != loaded_parameter_count
        or count_record.get("backbone_parameter_count") != base_text_count
        or count_record.get("head_parameter_count") != head_count
        or count_record.get("native_dtype") != "float32"
        or not isinstance(count_runtime, dict)
        or count_runtime.get("base_revision") != KEV_BASE_REVISION
        or count_runtime.get("source_revision") != KEV_SOURCE_REVISION
        or count_runtime.get("calibration_temperature") != head_temperature
        or count_runtime.get("inference_path")
        != "Checkpoint.load/DecisionModel.probs/api.to_answers"
    ):
        raise ValueError("Kev native loader count evidence differs")
    provenance = _json_object(package / "provenance.json")
    config = provenance.get("config")
    measured = provenance.get("measured_checkpoint")
    source_hashes = provenance.get("source_hashes")
    if (
        provenance.get("git_commit") != KEV_SOURCE_REVISION
        or not isinstance(config, dict)
        or config.get("base") != KEV_BASE_ID
        or config.get("base_revision") != KEV_BASE_REVISION
        or not isinstance(measured, dict)
        or measured.get("adapter_sha256") != files.get("adapter_model.safetensors")
        or not isinstance(source_hashes, dict)
        or not source_hashes
        or any(runtime_files.get(key) != value for key, value in source_hashes.items())
        or count_record.get("source_files_verified") != len(source_hashes)
    ):
        raise ValueError("Kev package provenance differs from pinned source/base")
    source_revision = subprocess.check_output(
        ["git", "-C", str(runtime), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if source_revision != KEV_SOURCE_REVISION:
        raise ValueError("Kev source checkout revision differs")
    if calibration_path is not None and calibration_path != package / "head.pt":
        raise ValueError("Kev calibration must be the packaged head")
    count_sha = _sha_file(native_loaded_count_receipt_path)
    composite = {
        "model_id": model.model_id,
        "revision": model.revision,
        "base_id": KEV_BASE_ID,
        "base_revision": KEV_BASE_REVISION,
        "source_revision": KEV_SOURCE_REVISION,
        "files": files,
        "base_files": base_files,
        "runtime_files": runtime_files,
        "native_loaded_count_receipt_sha256": count_sha,
    }
    return {
        "schema_version": SCHEMA,
        "model_id": model.model_id,
        "revision": model.revision,
        "native_model_sha256": hashlib.sha256(_json_bytes(composite)).hexdigest(),
        "adapter_sha256": _sha_file(source_root / "inference/kev.py"),
        "calibration_sha256": files["head.pt"],
        "calibration_path": str(package / "head.pt"),
        "package_path": str(package),
        "files": files,
        "weight_files": weight_files,
        "parameter_count": loaded_parameter_count,
        "loaded_parameter_count": loaded_parameter_count,
        "size_b": loaded_parameter_count / 1_000_000_000,
        "runtime_path": str(runtime),
        "runtime_files": runtime_files,
        "base_package_path": str(base_package),
        "base_files": base_files,
        "base_weight_files": base_weight_files,
        "base_parameter_count": base_all_count,
        "base_text_parameter_count": base_text_count,
        "adapter_parameter_count": adapter_count,
        "head_parameter_count": head_count,
        "native_loaded_count_receipt_path": str(native_loaded_count_receipt_path),
        "native_loaded_count_receipt_sha256": count_sha,
    }


def _receipt(
    model: Any,
    source_root: Path,
    model_root: Path,
    external_root: Path,
    calibration_path: Path | None,
    loaded_parameter_count: int,
    native_loaded_count_receipt_path: Path | None = None,
) -> dict[str, Any]:
    if model.key == "jev":
        raise ValueError("Hosted models have no local package attestation")
    if model.key == "kev":
        return _kev_receipt(
            model,
            source_root,
            model_root,
            external_root,
            calibration_path,
            loaded_parameter_count,
            native_loaded_count_receipt_path,
        )
    if native_loaded_count_receipt_path is not None:
        raise ValueError("Native loader receipt is only supported for Kev")
    package = model_root / model.model_dir
    files = _tree(package)
    weight_files, parameter_count = _weights(package, files)
    if (
        type(loaded_parameter_count) is not int
        or loaded_parameter_count != parameter_count
    ):
        raise ValueError("Native loader count differs from packaged tensor count")
    adapter = source_root / (model.module.replace(".", "/") + ".py")
    calibration_sha = _sha_file(calibration_path) if calibration_path else None
    receipt = {
        "schema_version": SCHEMA,
        "model_id": model.model_id,
        "revision": model.revision,
        "native_model_sha256": hashlib.sha256(_json_bytes(files)).hexdigest(),
        "adapter_sha256": _sha_file(adapter),
        "calibration_sha256": calibration_sha,
        "calibration_path": str(calibration_path) if calibration_path else None,
        "package_path": str(package),
        "files": files,
        "weight_files": weight_files,
        "parameter_count": parameter_count,
        "loaded_parameter_count": loaded_parameter_count,
        "size_b": parameter_count / 1_000_000_000,
    }
    if model.source_dir:
        runtime = external_root / model.source_dir
        receipt["runtime_path"] = str(runtime)
        receipt["runtime_files"] = _tree(runtime)
    return receipt


def verify_attestation(
    item: dict[str, Any],
    model: Any,
    *,
    source_root: Path,
    model_root: Path,
    external_root: Path,
) -> dict[str, Any]:
    """Recompute full byte inventory and parameter count from pinned inputs."""
    if not isinstance(item, dict) or set(item) != ATTESTATION_FIELDS:
        raise ValueError("Baseline attestation has missing or unknown fields")
    if (item["key"], item["model_id"], item["revision"]) != (
        model.key,
        model.model_id,
        model.revision,
    ):
        raise ValueError("Baseline identity differs from pinned catalog")
    if any(
        not _sha(item[field])
        for field in ("native_model_sha256", "adapter_sha256", "receipt_sha256")
    ):
        raise ValueError("Baseline attestation has an invalid digest")
    if item["calibration_sha256"] is not None and not _sha(item["calibration_sha256"]):
        raise ValueError("Baseline calibration digest is invalid")
    if (
        not isinstance(item["receipt_path"], str)
        or not Path(item["receipt_path"]).is_absolute()
        or type(item["size_b"]) is not float
    ):
        raise ValueError("Baseline receipt path or loaded size is invalid")
    path = Path(item["receipt_path"])
    _private_input(path)
    if _sha_file(path) != item["receipt_sha256"]:
        raise ValueError("Baseline receipt changed")
    receipt = _json_object(path)
    expected_fields = (
        RECEIPT_FIELDS
        | ({"runtime_path", "runtime_files"} if model.source_dir else set())
        | (KEV_RECEIPT_FIELDS if model.key == "kev" else set())
    )
    if set(receipt) != expected_fields or receipt.get("schema_version") != SCHEMA:
        raise ValueError("Baseline receipt schema differs")
    calibration = receipt["calibration_path"]
    if calibration is not None and (
        not isinstance(calibration, str) or not Path(calibration).is_absolute()
    ):
        raise ValueError("Baseline calibration path is invalid")
    rebuilt = _receipt(
        model,
        source_root,
        model_root,
        external_root,
        Path(calibration) if calibration else None,
        receipt["loaded_parameter_count"],
        (
            Path(receipt["native_loaded_count_receipt_path"])
            if model.key == "kev"
            and isinstance(receipt.get("native_loaded_count_receipt_path"), str)
            else None
        ),
    )
    if receipt != rebuilt:
        raise ValueError("Baseline package, runtime, or calibration differs")
    for field in (
        "model_id",
        "revision",
        "native_model_sha256",
        "adapter_sha256",
        "calibration_sha256",
        "size_b",
    ):
        if item[field] != receipt[field]:
            raise ValueError("Baseline attestation differs from receipt")
    return receipt


def _private_output(path: Path) -> None:
    if not path.is_absolute() or not path.parent.is_dir():
        raise ValueError("Private output needs an existing absolute parent directory")
    for part in (path.parent, *path.parent.parents):
        if stat.S_ISLNK(part.lstat().st_mode):
            raise ValueError("Private output parent contains a symlink")
    parent = path.parent.stat()
    if parent.st_uid != os.getuid() or parent.st_mode & 0o077:
        raise ValueError("Private output parent must be owned by caller and mode 0700")
    if path.exists() or path.is_symlink():
        raise ValueError("Private output already exists")


def _write_exclusive(path: Path, content: bytes) -> None:
    fd = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    try:
        with os.fdopen(fd, "wb", closefd=False) as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        path.unlink()
        raise
    finally:
        os.close(fd)


def build_attestation(
    model: Any,
    *,
    source_root: Path,
    model_root: Path,
    external_root: Path,
    receipt_output: Path,
    attestation_output: Path,
    loaded_parameter_count: int,
    calibration_path: Path | None = None,
    native_loaded_count_receipt_path: Path | None = None,
) -> None:
    """Write two exclusive private files; emit no package path on stdout."""
    if receipt_output == attestation_output:
        raise ValueError("Private output files must be distinct")
    _private_output(receipt_output)
    _private_output(attestation_output)
    receipt = _receipt(
        model,
        source_root,
        model_root,
        external_root,
        calibration_path,
        loaded_parameter_count,
        native_loaded_count_receipt_path,
    )
    receipt_bytes = _json_bytes(receipt) + b"\n"
    item = {
        "key": model.key,
        "model_id": model.model_id,
        "revision": model.revision,
        "size_b": receipt["size_b"],
        "native_model_sha256": receipt["native_model_sha256"],
        "adapter_sha256": receipt["adapter_sha256"],
        "calibration_sha256": receipt["calibration_sha256"],
        "receipt_path": str(receipt_output),
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
    }
    created: list[Path] = []
    try:
        _write_exclusive(receipt_output, receipt_bytes)
        created.append(receipt_output)
        _write_exclusive(attestation_output, _json_bytes(item) + b"\n")
        created.append(attestation_output)
        verify_attestation(
            item,
            model,
            source_root=source_root,
            model_root=model_root,
            external_root=external_root,
        )
    except BaseException:
        for path in created:
            path.unlink()
        raise


def main(argv: list[str] | None = None) -> int:
    from scripts.plan_final_eval import BASELINES

    parser = _QuietArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--key", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path)
    parser.add_argument("--attestation-output", type=Path, required=True)
    parser.add_argument("--calibration-file", type=Path)
    parser.add_argument("--loaded-parameter-count", type=int)
    parser.add_argument("--native-loaded-count-receipt", type=Path)
    args = parser.parse_args(argv)
    try:
        model = next(model for model in BASELINES if model.key == args.key)
        if args.command == "build":
            if args.receipt_output is None or args.loaded_parameter_count is None:
                raise ValueError("Build needs receipt output and measured loader count")
            build_attestation(
                model,
                source_root=args.source_root,
                model_root=args.model_root,
                external_root=args.external_root,
                receipt_output=args.receipt_output,
                attestation_output=args.attestation_output,
                loaded_parameter_count=args.loaded_parameter_count,
                calibration_path=args.calibration_file,
                native_loaded_count_receipt_path=args.native_loaded_count_receipt,
            )
        else:
            item = _json_object(args.attestation_output)
            verify_attestation(
                item,
                model,
                source_root=args.source_root,
                model_root=args.model_root,
                external_root=args.external_root,
            )
    except (OSError, ValueError, StopIteration):
        parser.exit(2, "Baseline attestation failed; inspect private inputs locally.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
