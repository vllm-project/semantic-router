#!/usr/bin/env python3
"""Prepare Dashboard bind mounts without following mutable path names."""

from __future__ import annotations

import argparse
import os
import stat
from collections.abc import Iterator
from contextlib import suppress

DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
PATH_FLAGS = os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW


def _normalize_system_path(path: str) -> str:
    """Normalize one immutable system alias without resolving mutable symlinks."""

    if os.pardir in path.split(os.sep):
        raise OSError("path must not contain parent traversal")

    absolute = os.path.abspath(path)
    if absolute == "/var/run":
        return "/run"
    if absolute.startswith("/var/run/"):
        return "/run/" + absolute[len("/var/run/") :]
    return absolute


def _components(path: str) -> tuple[int, list[str]]:
    absolute = _normalize_system_path(path)
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
    parent, name = os.path.split(_normalize_system_path(path))
    if not name:
        return open_directory(parent)
    directory = open_directory(parent)
    try:
        return os.open(name, flags, dir_fd=directory)
    finally:
        os.close(directory)


def secure_socket_gid(path: str) -> int:
    descriptor = open_path(path)
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


def prepare_bootstrap_token(path: str, uid: int, gid: int) -> None:
    """Hand one dedicated one-time token to the non-root Dashboard process."""

    absolute = _normalize_system_path(path)
    parent_path, name = os.path.split(absolute)
    if not name:
        raise OSError("bootstrap token filename is empty")
    directory = open_directory(parent_path)
    try:
        directory_info = os.fstat(directory)
        if not stat.S_ISDIR(directory_info.st_mode):
            raise OSError("bootstrap token parent must be a directory")
        os.fchown(directory, uid, gid)
        os.fchmod(directory, 0o700)
        try:
            token = os.open(
                name,
                os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=directory,
            )
        except FileNotFoundError:
            return
        try:
            info = os.fstat(token)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise OSError("bootstrap token must be a private regular file")
            os.fchown(token, uid, gid)
            os.fchmod(token, 0o600)
        finally:
            os.close(token)
    finally:
        os.close(directory)


def stage_private_file(source: str, destination: str, uid: int, gid: int) -> None:
    """Copy one private bind-mounted file into an isolated runtime directory."""

    source_descriptor = open_path(
        source,
        os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        source_info = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_info.st_mode) or source_info.st_nlink != 1:
            raise OSError("Dashboard secret source must be a private regular file")
        if source_info.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise OSError(
                "Dashboard secret source must not grant group or other access"
            )

        parent_path, name = os.path.split(_normalize_system_path(destination))
        if not name:
            raise OSError("Dashboard secret destination filename is empty")
        destination_directory = open_directory(parent_path)
        try:
            with suppress(FileNotFoundError):
                os.unlink(name, dir_fd=destination_directory)
            destination_descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=destination_directory,
            )
            try:
                while chunk := os.read(source_descriptor, 64 * 1024):
                    offset = 0
                    while offset < len(chunk):
                        offset += os.write(destination_descriptor, chunk[offset:])
                os.fchmod(destination_descriptor, 0o600)
                os.fchown(destination_descriptor, uid, gid)
                os.fsync(destination_descriptor)
            finally:
                os.close(destination_descriptor)
        finally:
            os.close(destination_directory)
    finally:
        os.close(source_descriptor)


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
) -> None:
    root = open_directory(path)
    try:
        for _relative_dir, directories, files, directory_fd in _walk_directory(root):
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
    socket_gid = commands.add_parser("socket-gid")
    socket_gid.add_argument("path")
    bootstrap_token = commands.add_parser("prepare-bootstrap-token")
    bootstrap_token.add_argument("path")
    bootstrap_token.add_argument("uid", type=int)
    bootstrap_token.add_argument("gid", type=int)
    private_file = commands.add_parser("stage-private-file")
    private_file.add_argument("source")
    private_file.add_argument("destination")
    private_file.add_argument("uid", type=int)
    private_file.add_argument("gid", type=int)
    tree = commands.add_parser("prepare-tree")
    tree.add_argument("path")
    tree.add_argument("gid", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "socket-gid":
        print(secure_socket_gid(args.path))
    elif args.command == "prepare-bootstrap-token":
        prepare_bootstrap_token(args.path, args.uid, args.gid)
    elif args.command == "stage-private-file":
        stage_private_file(
            args.source,
            args.destination,
            args.uid,
            args.gid,
        )
    else:
        prepare_shared_tree(args.path, args.gid)


if __name__ == "__main__":
    main()
