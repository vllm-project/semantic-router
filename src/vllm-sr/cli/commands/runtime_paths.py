"""Path helpers for runtime-oriented CLI commands."""

from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from pathlib import Path

import yaml

from cli.consts import DEFAULT_STACK_NAME
from cli.runtime_stack import STACK_NAME_ENV, normalize_stack_name

DEFAULT_FILENAME_COMPONENT_LIMIT = 255


def _assert_direct_child(runtime_dir: Path, candidate: Path) -> None:
    """Fail closed unless candidate is a direct child of the owned directory."""
    if runtime_dir.is_symlink():
        raise ValueError(
            f"Runtime state directory must not be a symbolic link: {runtime_dir}"
        )
    if candidate.parent != runtime_dir:
        raise ValueError(
            f"Runtime state path must be a direct child of {runtime_dir}: {candidate}"
        )
    if candidate.is_symlink():
        raise ValueError(f"Runtime state file must not be a symbolic link: {candidate}")
    if candidate.parent.resolve(strict=True) != runtime_dir.resolve(strict=True):
        raise ValueError(
            f"Runtime state path escapes the owned directory {runtime_dir}: {candidate}"
        )


def _assert_filename_fits(runtime_dir: Path, filename: str) -> None:
    try:
        component_limit = os.pathconf(runtime_dir, "PC_NAME_MAX")
    except (AttributeError, OSError, ValueError):
        component_limit = DEFAULT_FILENAME_COMPONENT_LIMIT
    if component_limit <= 0:
        component_limit = DEFAULT_FILENAME_COMPONENT_LIMIT
    encoded_length = len(os.fsencode(filename))
    if encoded_length > component_limit:
        raise ValueError(
            f"Runtime state filename exceeds the {component_limit}-byte filesystem "
            f"limit after stack-name normalization: {encoded_length} bytes"
        )


def _runtime_config_output_dir(config_path: Path) -> Path:
    state_root = os.getenv("VLLM_SR_STATE_ROOT_DIR", "").strip()
    base_dir = Path(state_root).expanduser() if state_root else config_path.parent
    runtime_dir = base_dir / ".vllm-sr"
    if runtime_dir.is_symlink():
        raise ValueError(
            f"Runtime state directory must not be a symbolic link: {runtime_dir}"
        )
    runtime_dir.mkdir(parents=True, exist_ok=True)
    if runtime_dir.is_symlink() or not runtime_dir.is_dir():
        raise ValueError(
            f"Runtime state directory is not an owned directory: {runtime_dir}"
        )
    return runtime_dir


def _runtime_config_output_path(source_config_path: Path) -> Path:
    stack_name = normalize_stack_name(os.getenv(STACK_NAME_ENV))
    filename = "runtime-config.yaml"
    if stack_name != DEFAULT_STACK_NAME:
        filename = f"runtime-config.{stack_name}.yaml"
    runtime_dir = _runtime_config_output_dir(source_config_path)
    _assert_filename_fits(runtime_dir, filename)
    output_path = runtime_dir / filename
    _assert_direct_child(runtime_dir, output_path)
    return output_path


def _container_runtime_config_path(source_config_path: Path) -> str:
    return f"/app/.vllm-sr/{_runtime_config_output_path(source_config_path).name}"


def _container_source_config_path() -> str:
    return "/app/config.yaml"


def _write_runtime_config(source_config_path: Path, config: dict[str, object]) -> Path:
    runtime_config_path = _runtime_config_output_path(source_config_path)
    runtime_dir = runtime_config_path.parent
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{runtime_config_path.name}.", suffix=".tmp", dir=runtime_dir
    )
    temp_path = Path(temp_name)
    try:
        os.fchmod(fd, 0o600)
        handle = os.fdopen(fd, "w", encoding="utf-8")
        fd = -1
        with handle:
            yaml.dump(config, handle, default_flow_style=False, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        _assert_direct_child(runtime_dir, runtime_config_path)
        os.replace(temp_path, runtime_config_path)
    finally:
        if fd >= 0:
            os.close(fd)
        with suppress(FileNotFoundError):
            temp_path.unlink()
    return runtime_config_path
