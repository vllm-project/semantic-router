"""Private, atomic instance registry for standalone Decision runtimes."""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from cli.decision_runtime.image_reference import (
    ImmutableImageReferenceError,
    validate_immutable_image_reference,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - local container serving is POSIX-only.
    fcntl = None  # type: ignore[assignment]


log = logging.getLogger(__name__)


REGISTRY_VERSION = 3
_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600
_ASCII_CONTROL_THRESHOLD = 32
_SHA256_HEX_LENGTH = 64
_INSTANCE_NAME = re.compile(r"[a-z0-9](?:[a-z0-9_.-]{0,62}[a-z0-9])?")
_STATE_TRANSITIONS = {
    "starting": frozenset({"starting", "running", "cleanup-required"}),
    "running": frozenset({"cleanup-required"}),
    "cleanup-required": frozenset({"cleanup-required"}),
}
_RECORD_FIELDS = frozenset(
    {
        "instance_id",
        "instance_name",
        "container_name",
        "model",
        "revision",
        "endpoint",
        "backend",
        "dtype",
        "artifact_digest",
        "identity_digest",
        "image",
        "container_id",
        "runtime",
        "max_batch",
        "max_concurrency",
        "max_queue",
        "state",
        "generation",
        "created_at",
    }
)


class DecisionRegistryError(RuntimeError):
    """The Decision runtime instance registry is unsafe or inconsistent."""


class DecisionRegistryStaleRecordError(DecisionRegistryError):
    """A lifecycle mutation lost its compare-and-swap race."""


@dataclass(frozen=True)
class DecisionInstanceRecord:
    """Secret-free identity and lifecycle state for one managed instance."""

    instance_id: str
    instance_name: str
    container_name: str
    model: str
    revision: str
    endpoint: str
    backend: str
    dtype: str
    artifact_digest: str
    identity_digest: str
    image: str
    container_id: str | None
    runtime: str
    max_batch: int
    max_concurrency: int
    max_queue: int
    state: str
    generation: int
    created_at: str


@dataclass(frozen=True)
class DecisionRegistryCommit:
    """Visible registry mutation and whether its directory entry is durable."""

    record: DecisionInstanceRecord | None
    durable: bool


class DecisionInstanceRegistry:
    """Serialize registry mutations under a private cross-process file lock."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = (path or default_registry_path()).expanduser().absolute()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def reserve(self, record: DecisionInstanceRecord) -> DecisionRegistryCommit:
        """Atomically reserve an instance name and endpoint."""

        _validate_record(record)
        with self._locked_document() as document:
            instances = document["instances"]
            if record.instance_name in instances:
                raise DecisionRegistryError(
                    f"Decision runtime instance {record.instance_name!r} already exists."
                )
            for existing in instances.values():
                existing_record = _decode_record(existing)
                if _bindings_conflict(existing_record.endpoint, record.endpoint):
                    raise DecisionRegistryError(
                        "Decision runtime host port conflicts with registered endpoint "
                        f"{existing_record.endpoint}."
                    )
                if existing["container_name"] == record.container_name:
                    raise DecisionRegistryError(
                        f"Decision runtime container {record.container_name!r} "
                        "is already registered."
                    )
            instances[record.instance_name] = asdict(record)
            return DecisionRegistryCommit(
                record=record,
                durable=self._write_document(document),
            )

    def transition(
        self,
        instance_name: str,
        instance_id: str,
        *,
        expected_generation: int,
        expected_state: str,
        state: str,
        container_id: str | None | object = ...,
    ) -> DecisionRegistryCommit:
        """Compare-and-swap one owned reservation to a permitted next state."""

        with self._locked_document() as document:
            current = self._owned_record(
                document,
                instance_name,
                instance_id,
                expected_generation=expected_generation,
                expected_state=expected_state,
            )
            if state not in _STATE_TRANSITIONS[current.state]:
                raise DecisionRegistryError(
                    "Decision runtime registry lifecycle transition is invalid."
                )
            updates: dict[str, object] = {
                "state": state,
                "generation": current.generation + 1,
            }
            if container_id is not ...:
                if (
                    current.container_id is not None
                    and container_id != current.container_id
                ):
                    raise DecisionRegistryError(
                        "Decision runtime registry container identity cannot change."
                    )
                updates["container_id"] = container_id
            updated = replace(current, **updates)
            _validate_record(updated)
            document["instances"][instance_name] = asdict(updated)
            return DecisionRegistryCommit(
                record=updated,
                durable=self._write_document(document),
            )

    def remove(
        self,
        instance_name: str,
        instance_id: str,
        *,
        expected_generation: int,
        expected_state: str,
    ) -> DecisionRegistryCommit:
        """Compare-and-swap removal of one exact lifecycle generation."""

        with self._locked_document() as document:
            self._owned_record(
                document,
                instance_name,
                instance_id,
                expected_generation=expected_generation,
                expected_state=expected_state,
            )
            del document["instances"][instance_name]
            return DecisionRegistryCommit(
                record=None,
                durable=self._write_document(document),
            )

    def records(self) -> tuple[DecisionInstanceRecord, ...]:
        """Return a consistent registry snapshot."""

        with self._locked_document() as document:
            return tuple(
                _decode_record(item) for item in document["instances"].values()
            )

    def get(self, instance_name: str) -> DecisionInstanceRecord:
        """Return one consistent record by public instance name."""

        with self._locked_document() as document:
            instances = document["instances"]
            if not isinstance(instances, dict):
                raise DecisionRegistryError("Decision runtime registry is invalid.")
            encoded = instances.get(instance_name)
            if encoded is None:
                raise DecisionRegistryError(
                    f"Decision runtime instance {instance_name!r} is not registered."
                )
            return _decode_record(encoded)

    def _owned_record(
        self,
        document: dict[str, object],
        instance_name: str,
        instance_id: str,
        *,
        expected_generation: int,
        expected_state: str,
    ) -> DecisionInstanceRecord:
        instances = document["instances"]
        if not isinstance(instances, dict):  # guarded by _read_document
            raise DecisionRegistryError("Decision runtime registry is invalid.")
        encoded = instances.get(instance_name)
        if encoded is None:
            raise DecisionRegistryStaleRecordError(
                f"Decision runtime instance {instance_name!r} is not registered."
            )
        current = _decode_record(encoded)
        if current.instance_id != instance_id:
            raise DecisionRegistryStaleRecordError(
                f"Decision runtime instance {instance_name!r} changed ownership."
            )
        if current.generation != expected_generation or current.state != expected_state:
            raise DecisionRegistryStaleRecordError(
                f"Decision runtime instance {instance_name!r} changed lifecycle state."
            )
        return current

    @contextmanager
    def _locked_document(self) -> Iterator[dict[str, object]]:
        if fcntl is None:
            raise DecisionRegistryError(
                "Decision runtime registry coordination requires a POSIX host."
            )
        _ensure_private_directory(self.path.parent)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        try:
            lock_fd = os.open(self.lock_path, flags, _PRIVATE_FILE_MODE)
        except OSError as error:
            raise DecisionRegistryError(
                "Decision runtime registry lock cannot be opened safely."
            ) from error
        try:
            try:
                info = os.fstat(lock_fd)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_nlink != 1
                    or info.st_uid != os.geteuid()
                ):
                    raise DecisionRegistryError(
                        "Decision runtime registry lock must be a private regular file."
                    )
                os.fchmod(lock_fd, _PRIVATE_FILE_MODE)
                os.set_inheritable(lock_fd, False)
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            except DecisionRegistryError:
                raise
            except OSError as error:
                raise DecisionRegistryError(
                    "Decision runtime registry lock cannot be secured."
                ) from error
            yield self._read_document()
        finally:
            with suppress(OSError):
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            with suppress(OSError):
                os.close(lock_fd)

    def _read_document(self) -> dict[str, object]:
        if not os.path.lexists(self.path):
            return {"version": REGISTRY_VERSION, "instances": {}}
        try:
            info = self.path.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or stat.S_ISLNK(info.st_mode)
                or info.st_nlink != 1
                or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) & 0o077
            ):
                raise DecisionRegistryError(
                    "Decision runtime registry must be a private regular file."
                )
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except DecisionRegistryError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise DecisionRegistryError(
                "Decision runtime registry cannot be read safely."
            ) from error
        if not isinstance(document, dict) or set(document) != {
            "version",
            "instances",
        }:
            raise DecisionRegistryError("Decision runtime registry is invalid.")
        if document["version"] != REGISTRY_VERSION or not isinstance(
            document["instances"], dict
        ):
            raise DecisionRegistryError("Decision runtime registry is invalid.")
        for name, encoded in document["instances"].items():
            record = _decode_record(encoded)
            if name != record.instance_name:
                raise DecisionRegistryError("Decision runtime registry is invalid.")
        return document

    def _write_document(self, document: dict[str, object]) -> bool:
        encoded = (
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        try:
            temporary_fd, temporary_name = tempfile.mkstemp(
                dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp"
            )
        except OSError as error:
            raise DecisionRegistryError(
                "Decision runtime registry temporary file cannot be created safely."
            ) from error
        temporary_path = Path(temporary_name)
        try:
            os.fchmod(temporary_fd, _PRIVATE_FILE_MODE)
            with os.fdopen(temporary_fd, "wb", closefd=True) as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
        except OSError as error:
            with suppress(OSError):
                os.close(temporary_fd)
            raise DecisionRegistryError(
                "Decision runtime registry was not committed atomically."
            ) from error
        finally:
            with suppress(OSError):
                temporary_path.unlink()

        # The atomic replacement above is the commit point. A directory fsync
        # failure can weaken crash durability, but raising here would report an
        # already-visible mutation as uncommitted and can trigger unsafe retry
        # cleanup. Log best-effort and keep the committed result authoritative.
        try:
            _fsync_directory(self.path.parent)
        except OSError as error:
            # Logging itself must never make a committed mutation look failed.
            with suppress(Exception):
                log.error(
                    "Decision runtime registry committed, but its directory could "
                    "not be synchronized: %s",
                    error,
                )
            return False
        return True


def default_registry_path() -> Path:
    """Return the per-user restart-safe Decision runtime registry path."""

    state_home = os.getenv("XDG_STATE_HOME", "").strip()
    root = Path(state_home).expanduser() if state_home else Path.home() / ".local/state"
    if not root.is_absolute():
        raise DecisionRegistryError("XDG_STATE_HOME must be an absolute path.")
    return root / "vllm-sr" / "decision-runtime" / "instances.json"


def _ensure_private_directory(path: Path) -> None:
    """Create and validate the task-owned registry directory."""

    missing: list[Path] = []
    cursor = path
    try:
        while not os.path.lexists(cursor):
            missing.append(cursor)
            parent = cursor.parent
            if parent == cursor:
                break
            cursor = parent
        for directory in reversed(missing):
            # A concurrent creator is acceptable only if final validation proves
            # the resulting path is the expected private directory.
            with suppress(FileExistsError):
                os.mkdir(directory, _PRIVATE_DIRECTORY_MODE)
            _fsync_directory(directory.parent)
        info = path.lstat()
    except OSError as error:
        raise DecisionRegistryError(
            "Decision runtime registry directory cannot be created durably."
        ) from error
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != os.geteuid()
    ):
        raise DecisionRegistryError(
            "Decision runtime registry directory must be user-owned."
        )
    try:
        os.chmod(path, _PRIVATE_DIRECTORY_MODE)
    except OSError as error:
        raise DecisionRegistryError(
            "Decision runtime registry directory cannot be secured."
        ) from error


def _fsync_directory(path: Path) -> None:
    """Synchronize one directory entry without leaking a file descriptor."""

    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _decode_record(value: object) -> DecisionInstanceRecord:
    if not isinstance(value, dict) or set(value) != _RECORD_FIELDS:
        raise DecisionRegistryError("Decision runtime registry record is invalid.")
    try:
        record = DecisionInstanceRecord(**value)
    except TypeError as error:
        raise DecisionRegistryError(
            "Decision runtime registry record is invalid."
        ) from error
    _validate_record(record)
    return record


def _validate_record(record: DecisionInstanceRecord) -> None:
    string_fields = (
        record.instance_id,
        record.instance_name,
        record.container_name,
        record.model,
        record.revision,
        record.endpoint,
        record.backend,
        record.dtype,
        record.artifact_digest,
        record.identity_digest,
        record.image,
        record.runtime,
        record.state,
        record.created_at,
    )
    positive_integer_fields = (record.max_batch, record.max_concurrency)
    if any(not isinstance(value, str) or not value for value in string_fields):
        raise DecisionRegistryError("Decision runtime registry record is invalid.")
    if (
        not _INSTANCE_NAME.fullmatch(record.instance_name)
        or record.container_name != f"vllm-sr-drun-{record.instance_name}"
        or any(
            character.isspace() or ord(character) < _ASCII_CONTROL_THRESHOLD
            for value in string_fields
            for character in value
        )
    ):
        raise DecisionRegistryError("Decision runtime registry record is invalid.")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in positive_integer_fields
    ):
        raise DecisionRegistryError("Decision runtime registry record is invalid.")
    if (
        isinstance(record.max_queue, bool)
        or not isinstance(record.max_queue, int)
        or record.max_queue < 0
    ):
        raise DecisionRegistryError("Decision runtime registry record is invalid.")
    if record.backend not in {"rocm", "cuda", "cpu", "mlx"}:
        raise DecisionRegistryError("Decision runtime registry record is invalid.")
    if record.runtime not in {"docker", "podman"}:
        raise DecisionRegistryError("Decision runtime registry record is invalid.")
    if record.state not in _STATE_TRANSITIONS:
        raise DecisionRegistryError("Decision runtime registry record is invalid.")
    if (
        isinstance(record.generation, bool)
        or not isinstance(record.generation, int)
        or record.generation < 0
    ):
        raise DecisionRegistryError("Decision runtime registry record is invalid.")
    if not _valid_sha256_digest(record.artifact_digest) or not _valid_sha256_digest(
        record.identity_digest
    ):
        raise DecisionRegistryError("Decision runtime registry record is invalid.")
    try:
        validate_immutable_image_reference(record.image)
    except ImmutableImageReferenceError as error:
        raise DecisionRegistryError(
            "Decision runtime registry record is invalid."
        ) from error
    if record.container_id is not None and (
        not isinstance(record.container_id, str)
        or not re.fullmatch(r"[0-9a-f]{64}", record.container_id)
    ):
        raise DecisionRegistryError("Decision runtime registry record is invalid.")
    if record.state == "running" and record.container_id is None:
        raise DecisionRegistryError("Decision runtime registry record is invalid.")
    try:
        created_at = datetime.fromisoformat(record.created_at)
    except ValueError as error:
        raise DecisionRegistryError(
            "Decision runtime registry record is invalid."
        ) from error
    if created_at.tzinfo is None:
        raise DecisionRegistryError("Decision runtime registry record is invalid.")
    _endpoint_binding(record.endpoint)


def _valid_sha256_digest(value: str) -> bool:
    algorithm, separator, digest = value.partition(":")
    return (
        separator == ":"
        and algorithm == "sha256"
        and len(digest) == _SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in digest)
    )


def _bindings_conflict(left_endpoint: str, right_endpoint: str) -> bool:
    left_host, left_port = _endpoint_binding(left_endpoint)
    right_host, right_port = _endpoint_binding(right_endpoint)
    if left_port != right_port:
        return False
    wildcard_hosts = {"0.0.0.0", "::"}
    return (
        left_host in wildcard_hosts
        or right_host in wildcard_hosts
        or left_host == right_host
    )


def _endpoint_binding(endpoint: str) -> tuple[str, int]:
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError as error:
        raise DecisionRegistryError(
            "Decision runtime registry endpoint is invalid."
        ) from error
    if (
        parsed.scheme != "http"
        or not parsed.hostname
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise DecisionRegistryError("Decision runtime registry endpoint is invalid.")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError as error:
        raise DecisionRegistryError(
            "Decision runtime registry endpoint is invalid."
        ) from error
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return str(address.ipv4_mapped), port
    return address.compressed, port
