#!/usr/bin/env python3
"""Prepare Dashboard bind mounts without following mutable path names."""

from __future__ import annotations

import argparse
import os
import secrets
import stat
from collections.abc import Iterator
from contextlib import suppress

DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
PATH_FLAGS = os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW


def _components(path: str) -> tuple[int, list[str]]:
    absolute = os.path.abspath(path)
    parts = [part for part in absolute.split(os.sep) if part]
    return os.open(os.sep, DIRECTORY_FLAGS), parts


def open_directory(path: str) -> int:
    descriptor, parts = _components(path)
    try:
        for part in parts:
            next_descriptor = os.open(part, DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def open_path(path: str, flags: int = PATH_FLAGS) -> int:
    parent, name = os.path.split(os.path.abspath(path))
    if not name:
        return open_directory(parent)
    directory = open_directory(parent)
    try:
        return os.open(name, flags, dir_fd=directory)
    finally:
        os.close(directory)


def secure_gid(path: str) -> int:
    descriptor = open_path(path)
    try:
        return os.fstat(descriptor).st_gid
    finally:
        os.close(descriptor)


def secure_socket_gid(path: str) -> int:
    absolute = os.path.abspath(path)
    # Debian keeps /var/run as a distribution-owned compatibility symlink to
    # /run.  Resolve only that fixed system alias; never follow an arbitrary
    # mutable parent or final-component symlink supplied by a bind mount.
    if absolute == "/var/run":
        absolute = "/run"
    elif absolute.startswith("/var/run/"):
        absolute = "/run/" + absolute[len("/var/run/") :]

    descriptor = open_path(absolute)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISSOCK(info.st_mode):
            raise OSError("container runtime path must be a Unix socket")
        if info.st_gid == 0:
            raise OSError("container runtime socket must not grant the root group")
        required_group_access = stat.S_IRGRP | stat.S_IWGRP
        if info.st_mode & required_group_access != required_group_access:
            raise OSError("container runtime socket must grant group read/write access")
        if info.st_mode & (stat.S_IROTH | stat.S_IWOTH):
            raise OSError("container runtime socket must not grant other access")
        return info.st_gid
    finally:
        os.close(descriptor)


def probe_runtime_config(state_dir: str, config_path: str) -> None:
    config_parent = os.path.dirname(os.path.abspath(config_path))
    if os.path.abspath(state_dir) != config_parent:
        raise OSError("runtime config must be located directly in its state directory")
    state = open_directory(state_dir)
    try:
        config_name = os.path.basename(os.path.abspath(config_path))
        config = os.open(
            config_name,
            os.O_WRONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=state,
        )
        try:
            info = os.fstat(config)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise OSError("runtime config must be a private regular file")
        finally:
            os.close(config)
        probe_name = f".dashboard-write-check-{os.getpid()}-{secrets.token_hex(8)}"
        probe = os.open(
            probe_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=state,
        )
        os.close(probe)
        os.unlink(probe_name, dir_fd=state)
    finally:
        os.close(state)


def probe_writable_directory(path: str) -> None:
    directory = open_directory(path)
    probe_name = f".dashboard-write-check-{os.getpid()}-{secrets.token_hex(8)}"
    probe: int | None = None
    created = False
    try:
        probe = os.open(
            probe_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory,
        )
        created = True
    finally:
        try:
            if probe is not None:
                os.close(probe)
            if created:
                os.unlink(probe_name, dir_fd=directory)
        finally:
            os.close(directory)


def ensure_writable_directory(path: str) -> None:
    """Create one private directory under an existing, pinned writable parent."""

    parent_path, name = os.path.split(os.path.abspath(path))
    if not name:
        raise OSError("writable Dashboard directory name is empty")
    parent = open_directory(parent_path)
    try:
        with suppress(FileExistsError):
            os.mkdir(name, 0o750, dir_fd=parent)
        descriptor = os.open(name, DIRECTORY_FLAGS, dir_fd=parent)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISDIR(info.st_mode):
                raise OSError("writable Dashboard path must be a directory")
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)
    probe_writable_directory(path)


def ensure_shared_directory(root_path: str, target_path: str, gid: int) -> None:
    """Create a shared directory below a pinned root and repair every ancestor."""

    root_absolute = os.path.abspath(root_path)
    target_absolute = os.path.abspath(target_path)
    if os.path.commonpath((root_absolute, target_absolute)) != root_absolute:
        raise OSError("shared Dashboard directory must stay below its state root")
    relative = os.path.relpath(target_absolute, root_absolute)
    parts = [part for part in relative.split(os.sep) if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise OSError("shared Dashboard directory must be below its state root")

    descriptor = open_directory(root_absolute)
    try:
        for part in parts:
            with suppress(FileExistsError):
                os.mkdir(part, 0o2770, dir_fd=descriptor)
            next_descriptor = os.open(part, DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
            info = os.fstat(descriptor)
            os.fchown(descriptor, -1, gid)
            os.fchmod(
                descriptor,
                stat.S_IMODE(info.st_mode)
                | stat.S_IRGRP
                | stat.S_IWGRP
                | stat.S_IXGRP
                | stat.S_ISGID,
            )
    finally:
        os.close(descriptor)
    probe_writable_directory(target_absolute)


def prepare_regular_file(path: str, gid: int) -> None:
    descriptor = open_path(
        path,
        os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise OSError("writable Dashboard file must be a private regular file")
        os.fchown(descriptor, -1, gid)
        os.fchmod(descriptor, stat.S_IMODE(info.st_mode) | stat.S_IRGRP | stat.S_IWGRP)
    finally:
        os.close(descriptor)


def prepare_directory(path: str, gid: int) -> None:
    descriptor = open_directory(path)
    try:
        info = os.fstat(descriptor)
        os.fchown(descriptor, -1, gid)
        os.fchmod(
            descriptor,
            stat.S_IMODE(info.st_mode)
            | stat.S_IRGRP
            | stat.S_IWGRP
            | stat.S_IXGRP
            | stat.S_ISGID,
        )
    finally:
        os.close(descriptor)


def prepare_private_directory(path: str, uid: int, gid: int) -> None:
    """Create one Dashboard-owned private directory below a pinned parent."""

    ensure_writable_directory(path)
    descriptor = open_directory(path)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise OSError("private Dashboard path must be a directory")
        os.fchown(descriptor, uid, gid)
        os.fchmod(descriptor, 0o700)
    finally:
        os.close(descriptor)


def _raise_walk_error(error: OSError) -> None:
    raise error


def _walk_directory(root_fd: int) -> Iterator[tuple[str, int, list[str], list[str]]]:
    yield from os.fwalk(
        ".",
        topdown=True,
        onerror=_raise_walk_error,
        follow_symlinks=False,
        dir_fd=root_fd,
    )


def prepare_shared_tree(
    path: str,
    gid: int,
    *,
    credential_relative_path: str | None = None,
    credential_uid: int = 65532,
    credential_gid: int = 65532,
) -> None:
    root = open_directory(path)
    try:
        for relative_dir, directories, files, directory_fd in _walk_directory(root):
            directory_info = os.fstat(directory_fd)
            os.fchown(directory_fd, -1, gid)
            os.fchmod(
                directory_fd,
                stat.S_IMODE(directory_info.st_mode)
                | stat.S_IRGRP
                | stat.S_IWGRP
                | stat.S_IXGRP
                | stat.S_ISGID,
            )
            for name in [*directories, *files]:
                info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    raise OSError(f"shared Dashboard tree contains symlink: {name}")
            for name in files:
                descriptor = os.open(
                    name,
                    os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
                try:
                    info = os.fstat(descriptor)
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                        raise OSError(
                            f"shared Dashboard tree contains unsafe file: {name}"
                        )
                    relative_file = os.path.normpath(os.path.join(relative_dir, name))
                    if relative_file.startswith("./"):
                        relative_file = relative_file[2:]
                    if credential_relative_path == relative_file:
                        os.fchown(descriptor, credential_uid, credential_gid)
                        os.fchmod(descriptor, 0o600)
                    else:
                        os.fchown(descriptor, -1, gid)
                        os.fchmod(
                            descriptor,
                            stat.S_IMODE(info.st_mode) | stat.S_IRGRP | stat.S_IWGRP,
                        )
                finally:
                    os.close(descriptor)
    finally:
        os.close(root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    gid = commands.add_parser("gid")
    gid.add_argument("path")
    socket_gid = commands.add_parser("socket-gid")
    socket_gid.add_argument("path")
    probe = commands.add_parser("probe-config")
    probe.add_argument("state_dir")
    probe.add_argument("config_path")
    writable_directory = commands.add_parser("probe-directory")
    writable_directory.add_argument("path")
    ensure_directory = commands.add_parser("ensure-directory")
    ensure_directory.add_argument("path")
    shared_directory = commands.add_parser("ensure-shared-directory")
    shared_directory.add_argument("root_path")
    shared_directory.add_argument("target_path")
    shared_directory.add_argument("gid", type=int)
    regular = commands.add_parser("prepare-file")
    regular.add_argument("path")
    regular.add_argument("gid", type=int)
    directory = commands.add_parser("prepare-directory")
    directory.add_argument("path")
    directory.add_argument("gid", type=int)
    private_directory = commands.add_parser("prepare-private-directory")
    private_directory.add_argument("path")
    private_directory.add_argument("uid", type=int)
    private_directory.add_argument("gid", type=int)
    tree = commands.add_parser("prepare-tree")
    tree.add_argument("path")
    tree.add_argument("gid", type=int)
    tree.add_argument("--credential-relative-path")
    tree.add_argument("--credential-uid", type=int, default=65532)
    tree.add_argument("--credential-gid", type=int, default=65532)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "gid":
        print(secure_gid(args.path))
    elif args.command == "socket-gid":
        print(secure_socket_gid(args.path))
    elif args.command == "probe-config":
        probe_runtime_config(args.state_dir, args.config_path)
    elif args.command == "probe-directory":
        probe_writable_directory(args.path)
    elif args.command == "ensure-directory":
        ensure_writable_directory(args.path)
    elif args.command == "ensure-shared-directory":
        ensure_shared_directory(args.root_path, args.target_path, args.gid)
    elif args.command == "prepare-file":
        prepare_regular_file(args.path, args.gid)
    elif args.command == "prepare-directory":
        prepare_directory(args.path, args.gid)
    elif args.command == "prepare-private-directory":
        prepare_private_directory(args.path, args.uid, args.gid)
    else:
        prepare_shared_tree(
            args.path,
            args.gid,
            credential_relative_path=args.credential_relative_path,
            credential_uid=args.credential_uid,
            credential_gid=args.credential_gid,
        )


if __name__ == "__main__":
    main()
