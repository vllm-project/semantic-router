"""Path-safe local CAS and append-only run evidence store."""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cli.evaluation.canonical import (
    canonical_json_bytes,
    pretty_json_bytes,
    sha256_digest,
)
from cli.evaluation.constants import ARTIFACT_NAMES, SCHEMA_VERSION
from cli.evaluation.contracts import ArtifactRef

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PRIVATE_DIR_MODE = 0o700


class StoreError(ValueError):
    """Evaluation artifact store rejected unsafe or corrupt data."""


class LocalArtifactStore:
    def __init__(self, root: str | Path):
        expanded = Path(root).expanduser().absolute()
        if expanded.is_symlink():
            raise StoreError("artifact store root must not be a symlink")
        self.root = expanded.resolve()
        self._ensure_private_dir(self.root)
        object_root = self.root / "objects"
        self._ensure_private_dir(object_root)
        self.objects = self.root / "objects" / "sha256"
        self.runs = self.root / "runs"
        self.index = self.root / "index"
        for directory in (self.objects, self.runs, self.index):
            self._ensure_private_dir(directory)

    @classmethod
    def _ensure_private_dir(cls, path: Path) -> None:
        if path.exists():
            cls._reject_symlink(path)
            if not path.is_dir():
                raise StoreError(f"artifact store path is not a directory: {path}")
            mode = stat.S_IMODE(path.stat().st_mode)
            if mode != _PRIVATE_DIR_MODE:
                raise StoreError(
                    f"artifact store directory must have mode 0700, got {mode:04o}: {path}"
                )
            return
        path.mkdir(parents=True, mode=_PRIVATE_DIR_MODE)
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode != _PRIVATE_DIR_MODE:
            raise StoreError(
                f"artifact store cannot enforce mode 0700, got {mode:04o}: {path}"
            )

    @staticmethod
    def _reject_symlink(path: Path) -> None:
        if path.is_symlink():
            raise StoreError(f"symlink is not allowed in artifact store: {path}")

    def _within_root(self, path: Path) -> Path:
        resolved = path.resolve(strict=False)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise StoreError("artifact path escapes store root") from exc
        return path

    def _run_dir(self, run_id: str, *, create: bool = False) -> Path:
        if not _RUN_ID_RE.fullmatch(run_id):
            raise StoreError("invalid run id")
        path = self._within_root(self.runs / run_id)
        if create and not path.exists():
            path.mkdir(parents=False, mode=_PRIVATE_DIR_MODE)
        if path.exists():
            self._reject_symlink(path)
            mode = stat.S_IMODE(path.stat().st_mode)
            if mode != _PRIVATE_DIR_MODE:
                raise StoreError(
                    f"run directory must have mode 0700, got {mode:04o}: {path}"
                )
        return path

    @staticmethod
    def _object_hex(digest: str) -> str:
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise StoreError("invalid sha256 digest")
        return digest.removeprefix("sha256:")

    def _object_path(self, digest: str) -> Path:
        return self._within_root(self.objects / self._object_hex(digest))

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def put_bytes(self, data: bytes, media_type: str) -> ArtifactRef:
        digest = sha256_digest(data)
        target = self._object_path(digest)
        if target.exists():
            self._reject_symlink(target)
            if target.read_bytes() != data:
                raise StoreError(f"CAS object does not match digest {digest}")
        else:
            self._atomic_write(target, data)
        return ArtifactRef(digest=digest, media_type=media_type, size_bytes=len(data))

    def put_json(self, value: Any) -> ArtifactRef:
        return self.put_bytes(canonical_json_bytes(value), "application/json")

    def put_jsonl(self, values: Iterable[Any]) -> ArtifactRef:
        data = b"".join(canonical_json_bytes(value) + b"\n" for value in values)
        return self.put_bytes(data, "application/x-ndjson")

    def read_bytes(self, ref: ArtifactRef) -> bytes:
        path = self._object_path(ref.digest)
        if not path.exists():
            raise StoreError(f"missing CAS object {ref.digest}")
        self._reject_symlink(path)
        data = path.read_bytes()
        if len(data) != ref.size_bytes or sha256_digest(data) != ref.digest:
            raise StoreError(f"corrupt CAS object {ref.digest}")
        return data

    def read_json(self, ref: ArtifactRef) -> Any:
        return json.loads(self.read_bytes(ref))

    def write_run_bytes(self, run_id: str, name: str, data: bytes) -> ArtifactRef:
        if name not in ARTIFACT_NAMES:
            raise StoreError(f"unsupported run artifact name: {name}")
        run_dir = self._run_dir(run_id, create=True)
        target = self._within_root(run_dir / name)
        if target.exists() or target.is_symlink():
            self._reject_symlink(target)
        if target.exists() and name not in {"status.json", "events.jsonl"}:
            if target.read_bytes() != data:
                raise StoreError(f"immutable run artifact already exists: {name}")
        else:
            self._atomic_write(target, data)
        media_type = (
            "application/x-ndjson" if name.endswith(".jsonl") else "application/json"
        )
        if name.endswith(".md"):
            media_type = "text/markdown"
        elif name.endswith(".html"):
            media_type = "text/html"
        elif name.endswith(".sha256"):
            media_type = "text/plain"
        return self.put_bytes(data, media_type)

    def write_run_json(self, run_id: str, name: str, value: Any) -> ArtifactRef:
        return self.write_run_bytes(run_id, name, pretty_json_bytes(value))

    def write_run_jsonl(
        self, run_id: str, name: str, values: Iterable[Any]
    ) -> ArtifactRef:
        data = b"".join(canonical_json_bytes(value) + b"\n" for value in values)
        return self.write_run_bytes(run_id, name, data)

    def append_event(self, run_id: str, value: Any) -> None:
        run_dir = self._run_dir(run_id, create=True)
        target = self._within_root(run_dir / "events.jsonl")
        if target.exists() or target.is_symlink():
            self._reject_symlink(target)
        data = canonical_json_bytes(value) + b"\n"
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(descriptor, data)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def read_run_json(self, run_id: str, name: str) -> Any:
        return json.loads(self.read_run_bytes(run_id, name))

    def read_run_bytes(self, run_id: str, name: str) -> bytes:
        if name not in ARTIFACT_NAMES:
            raise StoreError(f"unsupported run artifact name: {name}")
        path = self._within_root(self._run_dir(run_id) / name)
        if not path.exists():
            raise StoreError(f"run artifact does not exist: {run_id}/{name}")
        self._reject_symlink(path)
        return path.read_bytes()

    def read_run_text(self, run_id: str, name: str) -> str:
        return self.read_run_bytes(run_id, name).decode("utf-8")

    def reference_run_artifact(self, run_id: str, name: str) -> ArtifactRef:
        """Import an already staged run file into CAS without rewriting it."""

        if name not in ARTIFACT_NAMES:
            raise StoreError(f"unsupported run artifact name: {name}")
        path = self._within_root(self._run_dir(run_id) / name)
        if not path.exists():
            raise StoreError(f"run artifact does not exist: {run_id}/{name}")
        self._reject_symlink(path)
        data = path.read_bytes()
        media_type = "application/json"
        if name.endswith(".jsonl"):
            media_type = "application/x-ndjson"
        elif name.endswith(".md"):
            media_type = "text/markdown"
        elif name.endswith(".html"):
            media_type = "text/html"
        elif name.endswith(".sha256"):
            media_type = "text/plain"
        return self.put_bytes(data, media_type)

    def set_status(self, run_id: str, value: Any) -> None:
        self.write_run_json(run_id, "status.json", value)

    def update_index(self, run: Any) -> None:
        target = self._within_root(self.index / "runs.json")
        rows: list[dict[str, Any]] = []
        if target.exists():
            self._reject_symlink(target)
            loaded = json.loads(target.read_bytes())
            if isinstance(loaded, dict) and isinstance(loaded.get("runs"), list):
                rows = loaded["runs"]
        run_value = (
            run.model_dump(mode="json", exclude_none=True)
            if hasattr(run, "model_dump")
            else run
        )
        rows = [row for row in rows if row.get("id") != run_value.get("id")]
        rows.append(run_value)
        rows.sort(
            key=lambda row: (row.get("created_at", ""), row.get("id", "")), reverse=True
        )
        self._atomic_write(
            target, pretty_json_bytes({"schema_version": SCHEMA_VERSION, "runs": rows})
        )
