"""Bounded file and digest primitives for fail-closed Recipe recovery."""

import hashlib
import os
import re
import stat
from datetime import datetime
from pathlib import Path

from cli.commands.runtime_paths import _atomic_write_bytes, cli_user_share_gid

_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_RFC3339_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)
# The Dashboard shares the store with the CLI user's group: each side reads
# what the other wrote.
RECIPE_STORE_FILE_MODE = 0o640


class RecipeActivationRecoveryError(RuntimeError):
    """An interrupted activation could not be recovered safely."""


class RecipeStoreAccessError(RecipeActivationRecoveryError):
    """The CLI's user cannot read a Recipe store an earlier Dashboard wrote."""


def _unreadable(path: Path, message: str) -> RecipeActivationRecoveryError:
    """Explain a path below a Recipe store that this user may not open."""

    store = next(
        (
            candidate
            for candidate in (path, *path.parents)
            if candidate.parent.name == "recipe-store"
        ),
        None,
    )
    if store is None:
        return RecipeActivationRecoveryError(message)
    return RecipeStoreAccessError(
        f"This user cannot read the Recipe store {store}: a Dashboard from an "
        "earlier release kept it private to its own account. Share it with "
        f"your group once, then run `vllm-sr serve` again: sudo chgrp -R "
        f"{cli_user_share_gid()} {store} && sudo chmod -R g+rwX {store}"
    )


def write_recipe_store_bytes(path: Path, data: bytes) -> None:
    _atomic_write_bytes(path, data, RECIPE_STORE_FILE_MODE)


def _pending_journal_exists(store_dir: Path, journal_path: Path) -> bool:
    try:
        store_info = store_dir.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(store_info.st_mode) or not stat.S_ISDIR(store_info.st_mode):
        raise RecipeActivationRecoveryError(
            "The local Recipe package store is not a real directory."
        )
    try:
        journal_info = journal_path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(journal_info.st_mode) or not stat.S_ISREG(journal_info.st_mode):
        raise RecipeActivationRecoveryError(
            "The pending Recipe activation journal is not a regular file."
        )
    return True


def _require_real_directory(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except PermissionError as error:
        raise _unreadable(path, f"The {label} is not a real directory.") from error
    except OSError as error:
        raise RecipeActivationRecoveryError(
            f"The {label} is not a real directory."
        ) from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RecipeActivationRecoveryError(f"The {label} is not a real directory.")


def _require_bounded_regular_file(path: Path, limit: int, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError:
        raise
    except PermissionError as error:
        raise _unreadable(
            path, f"The {label} could not be inspected safely."
        ) from error
    except OSError as error:
        raise RecipeActivationRecoveryError(
            f"The {label} could not be inspected safely."
        ) from error
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_size > limit
    ):
        raise RecipeActivationRecoveryError(
            f"The {label} is not a bounded regular file."
        )
    return info


def _read_bounded_regular_file(path: Path, limit: int, label: str) -> bytes:
    before = _require_bounded_regular_file(path, limit, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except PermissionError as error:
        raise _unreadable(path, f"The {label} could not be opened safely.") from error
    except OSError as error:
        raise RecipeActivationRecoveryError(
            f"The {label} could not be opened safely."
        ) from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_size > limit
        ):
            raise RecipeActivationRecoveryError(
                f"The {label} changed while it was being read."
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            data = handle.read(limit + 1)
        if len(data) > limit:
            raise RecipeActivationRecoveryError(f"The {label} exceeds its size limit.")
        return data
    finally:
        os.close(descriptor)


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and _DIGEST_PATTERN.fullmatch(value) is not None


def _is_rfc3339(value: object) -> bool:
    if not isinstance(value, str) or _RFC3339_PATTERN.fullmatch(value) is None:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _digest_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"
