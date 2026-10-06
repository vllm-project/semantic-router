"""The built-in model entry and the recorded references shared by every family table."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

ORG = "vllm-sr"
REGISTRY_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BuiltinModel:
    """One pinned first-party package.

    ``manifest_sha256`` is the digest of the package's own manifest, or empty
    when the package has none. ``files`` maps every file the family loads to
    its SHA-256 at ``revision``, for packages whose manifest does not cover
    them (Decision 1.0, Vela 1.0, Vela 2.0, Omni); the resolver downloads
    exactly these files and the family verifies them. ``access`` is
    ``public``, ``gated`` or ``private`` (a token with access is needed).
    ``engines`` names the engine ``auto`` tries first per device class
    (``{"cpu": "onnxruntime"}``), where the performance records show it faster.
    ``reduced`` maps ``gpu`` / ``cpu`` to the reduced-precision copy the
    package's records support (``DtypePolicy.reduced_gpu`` / ``reduced_cpu``);
    a device it does not name gets none.
    """

    repo_id: str
    revision: str
    family: str
    model_sha256: str
    manifest_sha256: str
    loaded_parameters: int
    backbone: str
    min_device_memory_gib: float
    base: tuple[str, str] | None = None
    files: Mapping[str, str] = field(default_factory=dict)
    access: str = "public"
    engines: Mapping[str, str] = field(default_factory=dict)
    golden_answers: dict[str, Any] = field(default_factory=dict)
    kernel_choices: dict[str, Any] = field(default_factory=dict)
    reduced: Mapping[str, str] = field(default_factory=dict)


def with_recorded(
    models: tuple[BuiltinModel, ...], file_name: str, entry_field: str, model_field: str
) -> tuple[BuiltinModel, ...]:
    """Attach references recorded for the pinned revision (``registry/<file_name>``)."""
    path = REGISTRY_DIR / file_name
    if not path.is_file():
        return models
    recorded = json.loads(path.read_text(encoding="utf-8"))
    result = []
    for model in models:
        entry = recorded.get(model.repo_id)
        if entry and entry.get("revision") == model.revision:
            result.append(replace(model, **{model_field: entry[entry_field]}))
        else:
            result.append(model)
    return tuple(result)
