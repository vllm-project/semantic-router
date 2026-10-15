"""The Vela 2.0 package formats: what a package holds and how its bytes are verified.

``config.json`` names the member: ``vela2-unified`` (0.3B, a ModernBERT
encoder with marker tokens) or ``vela2-decoder`` (0.8B, 4B, 9B, a Qwen3.5
backbone read as a tree). The family loads the config, the calibration, the tokenizer
and the safetensors weights (plus the broad span head where a release ships
one). Every loaded byte must match the built-in table's digest for a pinned
revision, the package's ``SHA256SUMS`` when present, and the weight digests of
its ``MODEL_MANIFEST.json``; bundled Python is never read. The identity is the
SHA-256 of the weight files' digests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...errors import PackageError
from ...registry.artifacts import named_files, read_json, safe_relative, sha256_json
from .calibration import Calibration

ENCODER, DECODER = "vela2-unified", "vela2-decoder"
FORMAT_VERSION = 1
BROAD_HEAD_FILE = "broad_head.safetensors"
MANIFEST_FILE = "MODEL_MANIFEST.json"
SUMS_FILE = "SHA256SUMS"
COMMON_FILES = ("config.json", "calibration.json", "tokenizer.json")


def member_of(root: Path) -> str | None:
    """The member a package directory holds, read from its ``config.json`` only."""
    try:
        config = json.loads((Path(root) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict) or config.get("format_version") != FORMAT_VERSION:
        return None
    kind = config.get("model_type")
    return kind if kind in (ENCODER, DECODER) else None


def weight_files(root: Path, member: str) -> list[str]:
    """The safetensors files that hold the model (index order for sharded decoders)."""
    if member == ENCODER:
        return ["model.safetensors"]
    index = read_json(Path(root) / "model.safetensors.index.json", mapping=True)
    shards = sorted(set(index.get("weight_map", {}).values()))
    if not shards or not all(isinstance(s, str) and safe_relative(s) for s in shards):
        raise PackageError("model.safetensors.index.json lists no valid shards")
    return shards


def loaded_files(root: Path, member: str) -> list[str]:
    """Every file the family reads from a package."""
    names = list(COMMON_FILES) + weight_files(root, member)
    if member == DECODER:
        names.append("model.safetensors.index.json")
        if (Path(root) / BROAD_HEAD_FILE).is_file():
            names.append(BROAD_HEAD_FILE)
        if (Path(root) / MANIFEST_FILE).is_file():
            names.append(MANIFEST_FILE)
    return names


@dataclass(frozen=True)
class Vela2Package:
    """A verified Vela 2.0 package."""

    root: Path
    member: str
    config: dict[str, Any]
    calibration: Calibration
    weights: tuple[Path, ...]
    broad_head: Path | None
    files: dict[str, str]
    model_sha256: str
    manifest_sha256: str

    @property
    def max_input_tokens(self) -> int:
        return int(self.config["max_length"])


def _sums(root: Path) -> dict[str, str]:
    path = root / SUMS_FILE
    if not path.is_file():
        return {}
    sums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.strip().partition(" ")
        name = name.strip().lstrip("*")
        if digest and name:
            sums[name] = digest
    return sums


def verify(root: Path, expected: dict[str, str] | None) -> Vela2Package:
    """Check every loaded file; ``expected`` is the built-in table's digests for a pinned revision."""
    root = Path(root)
    member = member_of(root)
    if member is None:
        raise PackageError("config.json does not describe a Vela 2.0 package")
    files = named_files(root, loaded_files(root, member))
    if expected is not None:
        wrong = sorted(
            name
            for name in set(expected) | set(files)
            if expected.get(name) != files.get(name)
        )
        if wrong:
            raise PackageError(f"files differ from the pinned revision: {wrong[:5]}")
    sums = _sums(root)
    mismatched = sorted(
        name for name, digest in files.items() if name in sums and sums[name] != digest
    )
    if mismatched:
        raise PackageError(f"files differ from SHA256SUMS: {mismatched[:5]}")
    weights = weight_files(root, member)
    manifest_sha256 = ""
    if MANIFEST_FILE in files:
        manifest = read_json(root / MANIFEST_FILE, mapping=True)
        listed = manifest.get("files_sha256") or {}
        stale = sorted(
            name for name in weights if listed.get(name) not in (None, files[name])
        )
        if stale:
            raise PackageError(f"weights differ from {MANIFEST_FILE}: {stale}")
        manifest_sha256 = files[MANIFEST_FILE]
    config = read_json(root / "config.json", mapping=True)
    calibration = Calibration(read_json(root / "calibration.json", mapping=True))
    broad = root / BROAD_HEAD_FILE if BROAD_HEAD_FILE in files else None
    identity = {
        name: files[name] for name in weights + ([BROAD_HEAD_FILE] if broad else [])
    }
    return Vela2Package(
        root=root,
        member=member,
        config=config,
        calibration=calibration,
        weights=tuple(root / name for name in weights),
        broad_head=broad,
        files=files,
        model_sha256=sha256_json(identity),
        manifest_sha256=manifest_sha256,
    )
