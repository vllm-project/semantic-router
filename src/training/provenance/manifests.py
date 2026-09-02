from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar

SCHEMA_VERSION = 1


@dataclass
class Manifest:
    kind: ClassVar[str]
    schema_version: int = SCHEMA_VERSION
    name: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Manifest:
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known - {"kind"})
        if unknown:
            raise ValueError(f"{cls.kind}: unknown fields {unknown}")
        return cls(**{key: value for key, value in data.items() if key != "kind"})


@dataclass
class DatasetManifest(Manifest):
    kind: ClassVar[str] = "dataset"
    sources: list[dict[str, Any]] = field(default_factory=list)
    splits: dict[str, int] = field(default_factory=dict)
    preprocessing: list[str] = field(default_factory=list)
    label_mapping: dict[str, int] = field(default_factory=dict)


@dataclass
class RunManifest(Manifest):
    kind: ClassVar[str] = "run"
    task: str = ""
    dataset_id: str = ""
    base_model: dict[str, str] = field(default_factory=dict)
    code_revision: str = ""
    dependencies: dict[str, str] = field(default_factory=dict)
    seed: int | None = None
    hyperparameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationManifest(Manifest):
    kind: ClassVar[str] = "evaluation"
    run_id: str = ""
    dataset_id: str = ""
    split: str = ""
    sample_count: int = 0
    command: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class ArtifactManifest(Manifest):
    kind: ClassVar[str] = "artifact"
    run_id: str = ""
    evaluation_id: str = ""
    format: str = ""
    files: dict[str, str] = field(default_factory=dict)
    digest: str = ""
    label_mapping: dict[str, int] = field(default_factory=dict)


MANIFEST_TYPES: dict[str, type[Manifest]] = {
    cls.kind: cls
    for cls in (DatasetManifest, RunManifest, EvaluationManifest, ArtifactManifest)
}


def canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def manifest_id(manifest: Manifest) -> str:
    return (
        "sha256:"
        + hashlib.sha256(canonical_json(manifest.to_dict()).encode()).hexdigest()
    )


def manifest_from_dict(data: dict[str, Any]) -> Manifest:
    kind = data.get("kind")
    if kind not in MANIFEST_TYPES:
        raise ValueError(f"unknown manifest kind: {kind!r}")
    return MANIFEST_TYPES[kind].from_dict(data)


def load_manifest(path: Path) -> Manifest:
    with path.open("r", encoding="utf-8") as handle:
        return manifest_from_dict(json.load(handle))


def dump_manifest(manifest: Manifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
