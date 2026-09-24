"""Mapper artifact directory: manifest.json, weights.safetensors, SHA256SUMS."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.training.kv_mapper.mapper_id import normalize_precision

WEIGHTS_FILE = "weights.safetensors"
MANIFEST_FILE = "manifest.json"
CHECKSUMS_FILE = "SHA256SUMS"


@dataclass(frozen=True)
class CompatibilitySpec:
    """Fields the connector checks against the running deployment."""

    source_model: str
    source_revision: str
    target_model: str
    target_revision: str
    variant: str  # full_head | per_head
    precision: str
    source_tp: int
    target_tp: int
    head_order: str
    num_kv_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "precision", normalize_precision(self.precision))


@dataclass
class Manifest:
    mapper_id: str
    compatibility: CompatibilitySpec
    topk: int
    ridge_alpha: float
    centered_inputs: bool
    rope_stripped_on_keys: bool
    source_layers_per_target: dict[str, dict[str, list[int]]]
    calibration: dict[str, Any] = field(default_factory=dict)
    evaluation_report: str | None = None
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["compatibility"] = asdict(self.compatibility)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Manifest:
        compat = CompatibilitySpec(**data["compatibility"])
        return cls(
            mapper_id=data["mapper_id"],
            compatibility=compat,
            topk=int(data["topk"]),
            ridge_alpha=float(data["ridge_alpha"]),
            centered_inputs=bool(data["centered_inputs"]),
            rope_stripped_on_keys=bool(data["rope_stripped_on_keys"]),
            source_layers_per_target=data["source_layers_per_target"],
            calibration=dict(data.get("calibration", {})),
            evaluation_report=data.get("evaluation_report"),
            schema_version=int(data.get("schema_version", 1)),
        )


def tensor_keys_for_layers(target_layers: int) -> list[str]:
    keys: list[str] = []
    for layer in range(target_layers):
        for channel in ("k", "v"):
            keys.append(f"target.{layer}.{channel}.W")
            keys.append(f"target.{layer}.{channel}.b")
    return keys


def write_artifact(
    out_dir: Path,
    manifest: Manifest,
    tensors: dict[str, np.ndarray],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    from safetensors.numpy import save_file

    save_file(tensors, str(out_dir / WEIGHTS_FILE))
    (out_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    _write_checksums(out_dir)


def read_artifact(out_dir: Path) -> tuple[Manifest, dict[str, np.ndarray]]:
    manifest_path = out_dir / MANIFEST_FILE
    weights_path = out_dir / WEIGHTS_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing {manifest_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"missing {weights_path}")
    verify_checksums(out_dir)
    manifest = Manifest.from_dict(json.loads(manifest_path.read_text()))
    from safetensors.numpy import load_file

    tensors = dict(load_file(str(weights_path)))
    return manifest, tensors


def verify_compatibility(manifest: Manifest, deployment: CompatibilitySpec) -> None:
    """Refuse load when deployment layout does not match the fitted artifact."""
    fitted = manifest.compatibility
    checks: list[tuple[str, Any, Any]] = [
        ("source_model", fitted.source_model, deployment.source_model),
        ("source_revision", fitted.source_revision, deployment.source_revision),
        ("target_model", fitted.target_model, deployment.target_model),
        ("target_revision", fitted.target_revision, deployment.target_revision),
        ("variant", fitted.variant, deployment.variant),
        ("precision", fitted.precision, deployment.precision),
        ("source_tp", fitted.source_tp, deployment.source_tp),
        ("target_tp", fitted.target_tp, deployment.target_tp),
        ("head_order", fitted.head_order, deployment.head_order),
        ("num_kv_heads", fitted.num_kv_heads, deployment.num_kv_heads),
        ("head_dim", fitted.head_dim, deployment.head_dim),
    ]
    mismatches = [
        f"{name}: artifact={artifact!r} deployment={live!r}"
        for name, artifact, live in checks
        if artifact != live
    ]
    if mismatches:
        raise ValueError(
            "mapper compatibility check failed for "
            f"{manifest.mapper_id}: " + "; ".join(mismatches)
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksums(out_dir: Path) -> None:
    lines: list[str] = []
    for name in (MANIFEST_FILE, WEIGHTS_FILE):
        path = out_dir / name
        if path.is_file():
            lines.append(f"{_sha256_file(path)}  {name}")
    (out_dir / CHECKSUMS_FILE).write_text("\n".join(lines) + "\n")


def verify_checksums(out_dir: Path) -> None:
    sums_path = out_dir / CHECKSUMS_FILE
    if not sums_path.is_file():
        raise FileNotFoundError(f"missing {sums_path}")
    checksums: dict[str, str] = {}
    required = {MANIFEST_FILE, WEIGHTS_FILE}
    for line in sums_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"invalid checksum entry in {sums_path}: {line!r}")
        digest, name = parts
        if name not in required or name in checksums or len(digest) != 64 or any(
            char not in "0123456789abcdefABCDEF" for char in digest
        ):
            raise ValueError(f"invalid checksum entry for {name} in {sums_path}")
        checksums[name] = digest
    if set(checksums) != required:
        missing = sorted(required - set(checksums))
        raise ValueError(f"missing checksum entries for {missing} in {sums_path}")
    for name, digest in checksums.items():
        path = out_dir / name
        if _sha256_file(path) != digest.lower():
            raise ValueError(f"checksum mismatch for {name} in {out_dir}")
