"""Verify a downloaded Decision 2.0 package, load it, and answer System One requests.

``Decision2.from_pretrained(path)`` checks every packaged byte against
``MODEL_MANIFEST.json`` before any model code or weights are loaded, then
dispatches on the package profile. ``system_one(state=..., questions=...)``
returns typed Choice / Noul / Score answers; there is no text generation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST_SCHEMA = "dev2-package-manifest/1"
PACKAGE_SCHEMA = "dev2-package/1"
PROFILES = ("kai-native", "qwen-full", "qwen-adapter")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
HUB_ADDED = (".gitattributes",)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _transient(relative: PurePosixPath) -> bool:
    if relative.parts[:2] == (".cache", "huggingface"):
        return True
    return "__pycache__" in relative.parts and relative.suffix == ".pyc"


def _inventory(root: Path) -> dict[str, str]:
    files = {}
    for path in sorted(root.rglob("*")):
        relative = PurePosixPath(path.relative_to(root).as_posix())
        if path.is_symlink():
            raise ValueError(f"Package contains a link: {relative}")
        if path.is_dir() or _transient(relative) or relative.as_posix() in HUB_ADDED:
            continue
        name = relative.as_posix()
        if (
            not path.is_file()
            or relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
        ):
            raise ValueError(f"Unsafe package entry: {name}")
        files[name] = _sha_file(path)
    return files


def _tensor_count(path: Path) -> int:
    with path.open("rb") as stream:
        (size,) = struct.unpack("<Q", stream.read(8))
        if not 2 <= size <= 256 << 20:
            raise ValueError(f"Invalid safetensors header: {path.name}")
        header = json.loads(stream.read(size))
    return sum(
        math.prod(meta["shape"])
        for name, meta in header.items()
        if name != "__metadata__"
    )


def verify_bundle(path: str | Path) -> dict[str, Any]:
    """Exact inventory, per-file SHA-256 and header parameter counts; no weights loaded."""
    if Path(path).is_symlink():
        raise ValueError("Package root cannot be a link")
    root = Path(path).resolve(strict=True)
    pointer = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if (
        pointer.get("decision_format") != "vllm-sr-decision"
        or pointer.get("format_version") != 2
        or pointer.get("package_schema") != PACKAGE_SCHEMA
    ):
        raise ValueError("config.json is not a Decision 2.0 package pointer")
    manifest = json.loads((root / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    files = manifest.get("files_sha256") if isinstance(manifest, dict) else None
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("profile") not in PROFILES
        or not isinstance(files, dict)
        or "MODEL_MANIFEST.json" in files
        or any(
            not isinstance(v, str) or not SHA256.fullmatch(v) for v in files.values()
        )
    ):
        raise ValueError("Unknown or malformed Decision 2.0 package manifest")
    if _inventory(root) != {
        **files,
        "MODEL_MANIFEST.json": _sha_file(root / "MODEL_MANIFEST.json"),
    }:
        raise ValueError("Package files differ from MODEL_MANIFEST.json")
    if pointer.get("model_name") != manifest.get("model_name"):
        raise ValueError("Pointer and manifest name different models")
    counted = {
        group: sum(_tensor_count(root / name) for name in names)
        for group, names in manifest["parameters"]["packaged_files"].items()
    }
    if counted != manifest["parameters"]["packaged"]:
        raise ValueError("Packaged tensor counts differ from the manifest")
    return manifest


class Decision2:
    """Native Choice, Noul and Score decisions over one verified package."""

    def __init__(self, backend: Any, manifest: dict[str, Any], root: Path):
        self.backend = backend
        self.manifest = manifest
        self.root = root
        self.model_name = manifest["model_name"]
        self.max_input_tokens = manifest["max_input_tokens"]

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        device: str | None = None,
        base_path: str | Path | None = None,
        threads: int | None = None,
    ) -> Decision2:
        """Load a verified package. ``device`` defaults to cuda:0 if present, else cpu.

        ``base_path`` is only for base-bound adapter packages: a local copy of
        the pinned base; otherwise the listed base files are downloaded from
        the pinned Hub revision and verified byte for byte.
        """
        manifest = verify_bundle(path)
        root = Path(path).resolve(strict=True)
        import torch

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        profile = manifest["profile"]
        if profile == "kai-native":
            from .kai_native import KaiNative

            backend = KaiNative.load(root, manifest, device=device, threads=threads)
        else:
            from .qwen import QwenDecision

            backend = QwenDecision.load(
                root, manifest, device=device, base_path=base_path, threads=threads
            )
        loaded = backend.parameter_count()
        if loaded != manifest["parameters"]["loaded"]:
            raise ValueError(
                f"Loaded {loaded:,} parameters; the manifest declares "
                f"{manifest['parameters']['loaded']:,}"
            )
        return cls(backend, manifest, root)

    def system_one(self, *, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        """Answer named typed questions about one state; over-budget input is never truncated."""
        if (
            not isinstance(questions, dict)
            or not questions
            or any(not isinstance(key, str) or not key for key in questions)
        ):
            raise ValueError("questions must be a nonempty mapping of question IDs")
        answers, tokens = self.backend.system_one(state, questions)
        return {
            "model": self.model_name,
            "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }
