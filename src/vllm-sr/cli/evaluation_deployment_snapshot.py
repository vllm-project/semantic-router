"""Secure, content-addressed snapshots for Dashboard deployment registries."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path, PurePosixPath

_REGISTRY_VERSION = "evaluation-deployments.v1"
_REGISTRY_NAME = "registry.json"
_MAX_REGISTRY_BYTES = 1 << 20
_MAX_CONFIG_BYTES = 16 << 20
_SNAPSHOT_DIRECTORY = "evaluation-deployment-snapshots"
_SNAPSHOT_DIGEST_DOMAIN = b"evaluation-deployment-snapshot.v3\0"
_PRIVATE_DIRECTORY_MODE = 0o700
# The Dashboard runs as a non-root UID with the runtime-selected reader GID as
# a supplementary group. Keep snapshots immutable and private from all other
# users while allowing that container identity to traverse and read the
# bind-mounted snapshot, including when the host CLI itself runs as root.
_SNAPSHOT_DIRECTORY_MODE = 0o550
_SNAPSHOT_FILE_MODE = 0o440
_STAGING_FILE_MODE = 0o600
_DEPLOYMENT_FIELDS = {
    "id",
    "name",
    "description",
    "config_file",
    "router_origin",
    "envoy_origin",
}


def materialize_evaluation_deployment_snapshot(
    source_root: Path,
    staging_root: Path,
    *,
    readable_gid: int,
) -> Path:
    """Copy one stable no-follow registry view into immutable server state."""

    _validate_readable_gid(readable_gid)
    source_fd = _open_directory_no_follow(source_root.absolute())
    try:
        registry = _read_relative_regular_file(
            source_fd, _REGISTRY_NAME, _MAX_REGISTRY_BYTES
        )
        config_paths = _registry_config_paths(registry)
        files = {_REGISTRY_NAME: registry}
        for relative in config_paths:
            files[relative] = _read_relative_regular_file(
                source_fd, relative, _MAX_CONFIG_BYTES
            )
    finally:
        os.close(source_fd)

    digest = _snapshot_digest(files, readable_gid)
    parent = staging_root.absolute() / _SNAPSHOT_DIRECTORY
    parent.mkdir(mode=_PRIVATE_DIRECTORY_MODE, parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("Evaluation deployment snapshot directory is unsafe")
    parent.chmod(_PRIVATE_DIRECTORY_MODE)
    if stat.S_IMODE(parent.stat().st_mode) != _PRIVATE_DIRECTORY_MODE:
        raise ValueError("Evaluation deployment snapshot directory is not private")
    destination = parent / digest
    if destination.exists() or destination.is_symlink():
        _verify_existing_snapshot(destination, files, readable_gid)
        return destination

    temporary = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=parent))
    published = False
    try:
        for relative, content in files.items():
            target = temporary.joinpath(*PurePosixPath(relative).parts)
            target.parent.mkdir(
                mode=_PRIVATE_DIRECTORY_MODE, parents=True, exist_ok=True
            )
            descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                _STAGING_FILE_MODE,
            )
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                os.close(descriptor)
        _freeze_snapshot_permissions(temporary, readable_gid)
        _verify_existing_snapshot(temporary, files, readable_gid)
        os.rename(temporary, destination)
        _fsync_directory(parent)
        published = True
        return destination
    except FileExistsError:
        _verify_existing_snapshot(destination, files, readable_gid)
        return destination
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)


def _open_directory_no_follow(path: Path) -> int:
    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        raise ValueError(
            "Evaluation deployments directory must be absolute and canonical"
        )
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except OSError as error:
        os.close(descriptor)
        raise ValueError(
            "Evaluation deployments path components must not be symlinks"
        ) from error


def _read_relative_regular_file(root_fd: int, relative: str, limit: int) -> bytes:
    _validate_relative_path(relative)
    descriptor = os.dup(root_fd)
    try:
        parts = PurePosixPath(relative).parts
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        for component in parts[:-1]:
            next_descriptor = os.open(component, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        file_descriptor = os.open(
            parts[-1], os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=descriptor
        )
    except OSError as error:
        raise ValueError(
            "Evaluation deployment files must be regular and contain no symlinks"
        ) from error
    finally:
        os.close(descriptor)

    try:
        before = os.fstat(file_descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 0
            or before.st_size > limit
        ):
            raise ValueError("Evaluation deployment file is not a bounded regular file")
        with os.fdopen(file_descriptor, "rb", closefd=False) as stream:
            content = stream.read(limit + 1)
        after = os.fstat(file_descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if (
            len(content) > limit
            or len(content) != after.st_size
            or identity_before != identity_after
        ):
            raise ValueError("Evaluation deployment file changed while being copied")
        return content
    finally:
        os.close(file_descriptor)


def _registry_config_paths(registry: bytes) -> tuple[str, ...]:
    try:
        value = json.loads(registry)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Evaluation deployment registry is not valid JSON") from error
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "deployments"}
        or value.get("schema_version") != _REGISTRY_VERSION
        or not isinstance(value.get("deployments"), list)
        or not value["deployments"]
    ):
        raise ValueError("Evaluation deployment registry has an invalid envelope")
    paths: set[str] = set()
    for deployment in value["deployments"]:
        if (
            not isinstance(deployment, dict)
            or not set(deployment) <= _DEPLOYMENT_FIELDS
        ):
            raise ValueError("Evaluation deployment registry contains unknown fields")
        required = {"id", "name", "config_file", "router_origin", "envoy_origin"}
        if not required <= set(deployment) or not all(
            isinstance(deployment[field], str) for field in required
        ):
            raise ValueError("Evaluation deployment registry definition is incomplete")
        relative = deployment["config_file"]
        _validate_relative_path(relative)
        paths.add(relative)
    return tuple(sorted(paths))


def _validate_relative_path(relative: str) -> None:
    path = PurePosixPath(relative)
    if (
        not relative
        or relative != relative.strip()
        or "\\" in relative
        or path.is_absolute()
        or path.as_posix() != relative
        or any(component in {"", ".", ".."} for component in path.parts)
    ):
        raise ValueError(
            "Evaluation deployment config path must stay inside its registry"
        )


def _validate_readable_gid(readable_gid: int) -> None:
    if isinstance(readable_gid, bool) or not isinstance(readable_gid, int):
        raise ValueError(
            "Evaluation deployment snapshot GID must be a positive integer"
        )
    if readable_gid <= 0:
        raise ValueError("Evaluation deployment snapshot GID must be positive")


def _snapshot_digest(files: dict[str, bytes], readable_gid: int) -> str:
    digest = hashlib.sha256(_SNAPSHOT_DIGEST_DOMAIN)
    digest.update(str(readable_gid).encode("ascii"))
    digest.update(b"\0")
    for relative in sorted(files):
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(files[relative]).digest())
    return digest.hexdigest()


def _freeze_snapshot_permissions(root: Path, readable_gid: int) -> None:
    for directory, _, filenames in os.walk(root, topdown=False, followlinks=False):
        current = Path(directory)
        for filename in filenames:
            path = current / filename
            _assign_snapshot_entry(
                path,
                readable_gid,
                _SNAPSHOT_FILE_MODE,
                directory=False,
            )
        _assign_snapshot_entry(
            current,
            readable_gid,
            _SNAPSHOT_DIRECTORY_MODE,
            directory=True,
        )


def _assign_snapshot_entry(
    path: Path,
    readable_gid: int,
    mode: int,
    *,
    directory: bool,
) -> None:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    if directory:
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        expected_type = (
            stat.S_ISDIR(metadata.st_mode)
            if directory
            else stat.S_ISREG(metadata.st_mode)
        )
        if not expected_type:
            raise ValueError("Evaluation deployment snapshot contains an unsafe entry")
        os.fchown(descriptor, -1, readable_gid)
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)


def _verify_existing_snapshot(
    destination: Path,
    files: dict[str, bytes],
    readable_gid: int,
) -> None:
    descriptor = _open_directory_no_follow(destination.absolute())
    try:
        if _snapshot_inventory(descriptor, readable_gid=readable_gid) != set(files):
            raise ValueError(
                "Existing Evaluation deployment snapshot has unexpected files"
            )
        for relative, expected in files.items():
            actual = _read_relative_regular_file(
                descriptor, relative, max(len(expected), 1)
            )
            if actual != expected:
                raise ValueError(
                    "Existing Evaluation deployment snapshot is inconsistent"
                )
    finally:
        os.close(descriptor)


def _snapshot_inventory(
    descriptor: int,
    prefix: str = "",
    *,
    readable_gid: int,
) -> set[str]:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != _SNAPSHOT_DIRECTORY_MODE
        or metadata.st_gid != readable_gid
    ):
        raise ValueError("Evaluation deployment snapshot directory is not immutable")
    files: set[str] = set()
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    for name in os.listdir(descriptor):
        relative = f"{prefix}/{name}" if prefix else name
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISREG(metadata.st_mode):
            if (
                stat.S_IMODE(metadata.st_mode) != _SNAPSHOT_FILE_MODE
                or metadata.st_gid != readable_gid
            ):
                raise ValueError("Evaluation deployment snapshot file is not immutable")
            files.add(relative)
            continue
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("Evaluation deployment snapshot contains an unsafe entry")
        child = os.open(name, directory_flags, dir_fd=descriptor)
        try:
            files.update(
                _snapshot_inventory(
                    child,
                    relative,
                    readable_gid=readable_gid,
                )
            )
        finally:
            os.close(child)
    return files


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
