"""Bounded, non-executable readers for frozen benchmark exports."""

from __future__ import annotations

import csv
import math
import os
import stat
from collections.abc import Iterator, Sequence, Set
from pathlib import Path, PurePosixPath
from typing import Any

from cli.evaluation.benchmark_normalization_types import NativeArtifactRequirement
from cli.evaluation.canonical import strict_json_loads

_MAX_JSONL_LINE_BYTES = 16 * 1024 * 1024


class NormalizationError(ValueError):
    """A native export is missing, unsafe, ambiguous, or schema-invalid."""


def safe_export_root(value: str | Path) -> Path:
    root = Path(value).expanduser()
    if root.is_symlink():
        raise NormalizationError("native export root must not be a symlink")
    try:
        metadata = root.lstat()
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise NormalizationError("native export root is missing") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise NormalizationError("native export root must be a directory")
    return resolved


def safe_export_file(
    root: Path,
    relative_path: str,
    *,
    max_bytes: int,
) -> Path:
    relative = PurePosixPath(relative_path)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise NormalizationError("native artifact path is not a safe relative path")
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts[:-1]:
        current /= part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise NormalizationError(
                f"native artifact {relative_path!r} is missing"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise NormalizationError("native artifact parent must be a real directory")
    try:
        metadata = candidate.lstat()
        candidate.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise NormalizationError(
            f"native artifact {relative_path!r} is missing or outside its root"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise NormalizationError("native artifact must be a regular non-symlink file")
    if metadata.st_size > max_bytes:
        raise NormalizationError(f"native artifact {relative_path!r} exceeds its limit")
    return candidate


def required_file(root: Path, requirement: NativeArtifactRequirement) -> Path:
    return safe_export_file(
        root, requirement.relative_path, max_bytes=requirement.max_bytes
    )


def load_json(path: Path, *, max_bytes: int) -> Any:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise NormalizationError("could not read native JSON artifact") from exc
    if len(data) > max_bytes:
        raise NormalizationError("native JSON artifact exceeds its limit")
    try:
        return strict_json_loads(data)
    except (UnicodeDecodeError, ValueError) as exc:
        raise NormalizationError(
            "native JSON artifact is not valid UTF-8 JSON"
        ) from exc


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    try:
        handle = path.open("rb")
    except OSError as exc:
        raise NormalizationError("could not read native JSONL artifact") from exc
    with handle:
        for index, line in enumerate(handle, start=1):
            if len(line) > _MAX_JSONL_LINE_BYTES:
                raise NormalizationError(f"native JSONL line {index} is too large")
            if not line.endswith(b"\n") or not line.strip():
                raise NormalizationError(
                    f"native JSONL line {index} must be non-empty and LF terminated"
                )
            try:
                value = strict_json_loads(line)
            except (UnicodeDecodeError, ValueError) as exc:
                raise NormalizationError(
                    f"native JSONL line {index} is invalid UTF-8 JSON"
                ) from exc
            if not isinstance(value, dict):
                raise NormalizationError(f"native JSONL line {index} must be an object")
            yield value


def iter_csv(
    path: Path,
    expected_header: Sequence[str],
) -> Iterator[dict[str, str]]:
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise NormalizationError("could not read native CSV artifact") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != tuple(
            expected_header
        ):
            raise NormalizationError(
                "native CSV header mismatch; expected exactly: "
                + ",".join(expected_header)
            )
        for index, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise NormalizationError(f"native CSV row {index} has ragged columns")
            yield {key: value for key, value in row.items() if key is not None}


def require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise NormalizationError(f"{label} must be a JSON object")
    return value


def require_array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise NormalizationError(f"{label} must be a JSON array")
    return value


def exact_object(
    value: Any,
    *,
    required: set[str],
    optional: Set[str] = frozenset(),
    label: str,
) -> dict[str, Any]:
    result = require_object(value, label)
    keys = set(result)
    missing = required - keys
    unexpected = keys - required - optional
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unexpected:
            details.append("unexpected " + ", ".join(sorted(unexpected)))
        raise NormalizationError(f"{label} schema mismatch: {'; '.join(details)}")
    return result


def string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise NormalizationError(f"{label} must be a non-empty string")
    return value


def integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise NormalizationError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise NormalizationError(f"{label} must be an integer") from exc
    if str(result) != str(value).strip() and not isinstance(value, int):
        raise NormalizationError(f"{label} must use canonical integer syntax")
    if result < minimum:
        raise NormalizationError(f"{label} must be at least {minimum}")
    return result


def number(
    value: Any,
    label: str,
    *,
    minimum: float = 0,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise NormalizationError(f"{label} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise NormalizationError(f"{label} must be a number") from exc
    if not math.isfinite(result) or result < minimum:
        raise NormalizationError(f"{label} must be finite and at least {minimum}")
    if maximum is not None and result > maximum:
        raise NormalizationError(f"{label} must be at most {maximum}")
    return result


def boolean(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value in {"true", "false", "True", "False", "0", "1"}:
        return value in {"true", "True", "1"}
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise NormalizationError(f"{label} must be a boolean")


def no_duplicate(values: Sequence[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise NormalizationError(f"{label} contains duplicate identities")


def checked_media_file(root: Path, relative_path: str) -> tuple[Path, os.stat_result]:
    path = safe_export_file(root, relative_path, max_bytes=512 * 1024 * 1024)
    return path, path.lstat()
