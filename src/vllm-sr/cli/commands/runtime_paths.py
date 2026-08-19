"""Path helpers for runtime-oriented CLI commands."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from contextlib import suppress
from pathlib import Path

import yaml

from cli.consts import DEFAULT_STACK_NAME
from cli.runtime_stack import STACK_NAME_ENV, normalize_stack_name
from cli.utils import get_logger

DEFAULT_FILENAME_COMPONENT_LIMIT = 255
RUNTIME_CONFIG_PROVENANCE_SCHEMA = "vllm-sr/runtime-config-provenance/v1"
MAX_PROVENANCE_BYTES = 64 * 1024
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_PRIVATE_STATE_DIRECTORY = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_PRIVATE_DIRECTORY_MODE = 0o700
_RECIPE_ASSET_MODE = 0o644

log = get_logger(__name__)


def _current_posix_user_id() -> int | None:
    """Return the effective filesystem owner id where POSIX ownership applies."""

    get_user_id = getattr(os, "geteuid", None) or getattr(os, "getuid", None)
    return get_user_id() if os.name == "posix" and get_user_id is not None else None


def _create_or_harden_private_directory(
    directory: Path, *, parents: bool = False
) -> Path:
    """Create a private directory or fail closed on unsafe existing state."""

    if directory.is_symlink():
        raise ValueError(
            f"Runtime state directory must not be a symbolic link: {directory}"
        )
    # Validate an existing path below so callers receive one stable error.
    with suppress(FileExistsError):
        directory.mkdir(
            mode=_PRIVATE_DIRECTORY_MODE,
            parents=parents,
            exist_ok=True,
        )

    try:
        info = directory.lstat()
    except OSError as error:
        raise ValueError(
            f"Runtime state directory cannot be inspected: {directory}"
        ) from error
    if stat.S_ISLNK(info.st_mode):
        raise ValueError(
            f"Runtime state directory must not be a symbolic link: {directory}"
        )
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"Runtime state path must be a real directory: {directory}")

    current_user_id = _current_posix_user_id()
    if current_user_id is None:
        return directory
    if info.st_uid != current_user_id:
        raise ValueError(
            f"Runtime state directory must be owned by the current user: {directory}"
        )
    if stat.S_IMODE(info.st_mode) != _PRIVATE_DIRECTORY_MODE:
        try:
            os.chmod(directory, _PRIVATE_DIRECTORY_MODE, follow_symlinks=False)
        except (NotImplementedError, OSError) as error:
            raise ValueError(
                f"Runtime state directory permissions cannot be secured: {directory}"
            ) from error
        secured_info = directory.lstat()
        if (
            not stat.S_ISDIR(secured_info.st_mode)
            or secured_info.st_uid != current_user_id
            or stat.S_IMODE(secured_info.st_mode) != _PRIVATE_DIRECTORY_MODE
        ):
            raise ValueError(
                f"Runtime state directory permissions are not private: {directory}"
            )
    return directory


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


def _runtime_config_output_dir(
    config_path: Path, state_root_dir: str | Path | None = None
) -> Path:
    state_root = (
        str(state_root_dir).strip()
        if state_root_dir is not None
        else os.getenv("VLLM_SR_STATE_ROOT_DIR", "").strip()
    )
    base_dir = Path(state_root).expanduser() if state_root else config_path.parent
    runtime_dir = base_dir / ".vllm-sr"
    return _create_or_harden_private_directory(runtime_dir, parents=True)


def private_runtime_state_subdirectory(state_root_dir: str | Path, name: str) -> Path:
    """Create one owned private state directory without following symlinks."""

    if not _PRIVATE_STATE_DIRECTORY.fullmatch(name):
        raise ValueError(f"Runtime state directory name is invalid: {name}")
    state_root = Path(state_root_dir).expanduser().absolute()
    runtime_dir = _runtime_config_output_dir(
        state_root / "config.yaml", state_root_dir=state_root
    )
    directory = runtime_dir / name
    _create_or_harden_private_directory(directory)
    if directory.parent.resolve(strict=True) != runtime_dir.resolve(strict=True):
        raise ValueError(f"Runtime state directory escapes {runtime_dir}: {directory}")
    return directory


def private_runtime_state_nested_directory(
    state_root_dir: str | Path, parent_name: str, name: str
) -> Path:
    """Create one private child below a validated runtime-state directory."""

    if not _PRIVATE_STATE_DIRECTORY.fullmatch(name):
        raise ValueError(f"Runtime state directory name is invalid: {name}")
    parent = private_runtime_state_subdirectory(state_root_dir, parent_name)
    directory = parent / name
    _create_or_harden_private_directory(directory)
    if directory.parent.resolve(strict=True) != parent.resolve(strict=True):
        raise ValueError(f"Runtime state directory escapes {parent}: {directory}")
    return directory


def _runtime_config_filename(stack_name: str | None = None) -> str:
    stack_name = normalize_stack_name(
        stack_name if stack_name is not None else os.getenv(STACK_NAME_ENV)
    )
    filename = "runtime-config.yaml"
    if stack_name != DEFAULT_STACK_NAME:
        filename = f"runtime-config.{stack_name}.yaml"
    return filename


def _runtime_config_output_path(
    source_config_path: Path,
    *,
    state_root_dir: str | Path | None = None,
    stack_name: str | None = None,
) -> Path:
    filename = _runtime_config_filename(stack_name)
    runtime_dir = _runtime_config_output_dir(source_config_path, state_root_dir)
    _assert_filename_fits(runtime_dir, filename)
    output_path = runtime_dir / filename
    _assert_direct_child(runtime_dir, output_path)
    return output_path


def _container_runtime_config_path(
    source_config_path: Path, *, stack_name: str | None = None
) -> str:
    del source_config_path
    return f"/app/.vllm-sr/{_runtime_config_filename(stack_name)}"


def _container_readonly_source_config_path() -> str:
    return "/app/source-config.yaml"


def _runtime_config_provenance_path(runtime_config_path: Path) -> Path:
    return runtime_config_path.with_suffix(".provenance.json")


def _digest_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        directory_fd = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_write_bytes(path: Path, data: bytes, mode: int) -> None:
    runtime_dir = path.parent
    _assert_direct_child(runtime_dir, path)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=runtime_dir
    )
    temp_path = Path(temp_name)
    try:
        os.fchmod(fd, mode)
        handle = os.fdopen(fd, "wb")
        fd = -1
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _assert_direct_child(runtime_dir, path)
        os.replace(temp_path, path)
        _fsync_directory(runtime_dir)
    finally:
        if fd >= 0:
            os.close(fd)
        with suppress(FileNotFoundError):
            temp_path.unlink()


def _atomic_write_private_bytes(path: Path, data: bytes) -> None:
    _atomic_write_bytes(path, data, 0o600)


def write_runtime_config_bytes(path: Path, data: bytes) -> Path:
    """Atomically replace an explicit runtime-owned config path."""

    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError(
            f"Runtime config parent must be an owned directory: {path.parent}"
        )
    _atomic_write_private_bytes(path, data)
    return path


def write_runtime_recipe_asset_bytes(path: Path, data: bytes) -> Path:
    """Atomically write an embedded Recipe asset inside a private directory.

    The containing runtime-state directories remain owner-only. Files use the
    read-only asset mode expected by the non-root Dashboard after bind mount.
    """

    path = path.expanduser().absolute()
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError(
            f"Runtime Recipe parent must be an owned directory: {path.parent}"
        )
    _atomic_write_bytes(path, data, _RECIPE_ASSET_MODE)
    return path


def _provenance_document(source_data: bytes, active_data: bytes) -> dict[str, str]:
    return {
        "schema_version": RUNTIME_CONFIG_PROVENANCE_SCHEMA,
        "source_digest": _digest_bytes(source_data),
        "last_materialized_active_digest": _digest_bytes(active_data),
    }


def _write_provenance(path: Path, source_data: bytes, active_data: bytes) -> None:
    document = _provenance_document(source_data, active_data)
    encoded = (
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    _atomic_write_private_bytes(path, encoded)


def _load_provenance(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Runtime provenance must be a regular file: {path}")
    if path.stat().st_size > MAX_PROVENANCE_BYTES:
        raise ValueError(f"Runtime provenance exceeds {MAX_PROVENANCE_BYTES} bytes")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Runtime provenance is invalid: {error}") from error
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "source_digest",
        "last_materialized_active_digest",
    }:
        raise ValueError("Runtime provenance has an invalid field set")
    if document.get("schema_version") != RUNTIME_CONFIG_PROVENANCE_SCHEMA:
        raise ValueError("Runtime provenance has an unsupported schema_version")
    for field in ("source_digest", "last_materialized_active_digest"):
        value = document.get(field)
        if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
            raise ValueError(f"Runtime provenance {field} is invalid")
    return document


def materialize_runtime_config(
    source_config_path: Path,
    effective_data: bytes,
    *,
    state_root_dir: str | Path | None = None,
    stack_name: str | None = None,
) -> Path:
    """Reconcile one runtime-owned active config without overwriting edits.

    The CLI records the digest it last materialized. A Dashboard or package
    activation changes the active digest without changing that receipt, so a
    later ``serve`` preserves the active file and reports the divergence.
    """

    source_config_path = source_config_path.expanduser().absolute()
    source_data = source_config_path.read_bytes()
    runtime_config_path = _runtime_config_output_path(
        source_config_path,
        state_root_dir=state_root_dir,
        stack_name=stack_name,
    )
    provenance_path = _runtime_config_provenance_path(runtime_config_path)

    if runtime_config_path.exists():
        _assert_direct_child(runtime_config_path.parent, runtime_config_path)
        if not runtime_config_path.is_file():
            raise ValueError(
                f"Runtime config must be a regular file: {runtime_config_path}"
            )
        active_data = runtime_config_path.read_bytes()
        if active_data == effective_data:
            _write_provenance(provenance_path, source_data, active_data)
            return runtime_config_path

        try:
            provenance = _load_provenance(provenance_path)
        except ValueError as error:
            log.warning(
                "Preserving runtime config %s because its provenance is invalid: %s",
                runtime_config_path,
                error,
            )
            return runtime_config_path

        if provenance is None:
            log.warning(
                "Preserving pre-existing runtime config %s because no provenance "
                "receipt is available",
                runtime_config_path,
            )
            return runtime_config_path

        active_digest = _digest_bytes(active_data)
        if active_digest != provenance["last_materialized_active_digest"]:
            log.warning(
                "Preserving Dashboard or package changes in runtime config %s; "
                "source updates were not materialized",
                runtime_config_path,
            )
            return runtime_config_path

    _atomic_write_private_bytes(runtime_config_path, effective_data)
    _write_provenance(provenance_path, source_data, effective_data)
    return runtime_config_path


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
        _fsync_directory(runtime_dir)
    finally:
        if fd >= 0:
            os.close(fd)
        with suppress(FileNotFoundError):
            temp_path.unlink()
    return runtime_config_path
