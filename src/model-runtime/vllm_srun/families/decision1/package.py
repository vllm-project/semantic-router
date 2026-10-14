"""The Decision 1.0 package format (``vllm-sr-decision`` format 1), verified without running package code.

A package's root ``config.json`` is its file map: the model name, the runtime
(``vela-encoder`` or ``qwen3.5-decision``) and the paths of the model config,
the backbone config and weights, the tokenizer files, the decision weights and
the calibration. No manifest covers these files, so a built-in revision pins
the SHA-256 of every file the family loads; a package that ships its own
``MANIFEST.json`` (Route) is checked against it as well. The model identity is
the SHA-256 of the canonical JSON of the model files' digests. Route's
``QUESTIONS.json`` declares its router questions, served as presets. Bundled
Python is never imported.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...errors import PackageError
from ...registry.artifacts import named_files, read_json, safe_relative, sha256_json

POINTER = {"decision_format": "vllm-sr-decision", "format_version": 1}
VELA = "vela-encoder"
QWEN = "qwen3.5-decision"
RUNTIMES = (VELA, QWEN)
POINTER_FILE = "config.json"
MANIFEST_FILE = "MANIFEST.json"
PRESETS_FILE = "QUESTIONS.json"
VELA_WEIGHTS = ("choice_encoder", "score_encoder", "decision_heads")
QWEN_WEIGHTS = ("decision_head",)
TOKENIZER_FILES = ("json", "config", "special_tokens_map")
KINDS = ("choice", "noul", "score")


@dataclass(frozen=True)
class FileMap:
    """The pointer's file map: every path inference reads, relative to the package root."""

    model_name: str
    runtime: str
    model_config: str
    backbone_config: str
    backbone_weights: tuple[str, ...]
    backbone_index: str | None
    tokenizer: Mapping[str, str]
    decision_weights: Mapping[str, str]
    temperature: float | None = None
    temperature_file: str | None = None

    def model_files(self) -> list[str]:
        """The files that determine answers; the model identity covers exactly these."""
        names = [self.model_config, self.backbone_config, *self.backbone_weights]
        if self.backbone_index:
            names.append(self.backbone_index)
        names += [
            self.tokenizer[key] for key in TOKENIZER_FILES if key in self.tokenizer
        ]
        names += list(self.decision_weights.values())
        if self.temperature_file:
            names.append(self.temperature_file)
        return sorted(set(names))


@dataclass(frozen=True)
class Decision1Package:
    """What the family needs after verification."""

    root: Path
    files: FileMap
    model_config: dict[str, Any]
    backbone_config: dict[str, Any]
    digests: dict[str, str]
    model_sha256: str
    manifest_sha256: str
    verification: str
    temperatures: dict[str, float] | None = None
    presets: dict[str, dict[str, Any]] = field(default_factory=dict)

    def path(self, name: str) -> Path:
        return self.root / name


def read_pointer(root: Path) -> dict[str, Any] | None:
    """The root pointer when it names a Decision 1.0 runtime, else None (cheap; never raises)."""
    try:
        pointer = json.loads((Path(root) / POINTER_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(pointer, dict) or any(
        pointer.get(key) != value for key, value in POINTER.items()
    ):
        return None
    return pointer if pointer.get("runtime_family") in RUNTIMES else None


def _path(value: Any, what: str) -> str:
    if not isinstance(value, str) or not safe_relative(value):
        raise PackageError(f"config.json names {what} with an invalid path: {value!r}")
    return value


def _paths(value: Any, keys: tuple[str, ...], what: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise PackageError(f"config.json {what} must name exactly {list(keys)}")
    return {key: _path(value[key], f"{what}.{key}") for key in keys}


def file_map(pointer: dict[str, Any]) -> FileMap:
    """The validated file map of a Decision 1.0 pointer."""
    runtime = pointer.get("runtime_family")
    if runtime not in RUNTIMES:
        raise PackageError(f"unsupported Decision 1.0 runtime {runtime!r}")
    name = pointer.get("model_name")
    if not isinstance(name, str) or not name.strip():
        raise PackageError("config.json has no model_name")
    backbone = pointer.get("backbone")
    if not isinstance(backbone, dict):
        raise PackageError("config.json names no backbone")
    weights = backbone.get("weights")
    if not isinstance(weights, list) or not weights:
        raise PackageError("config.json names no backbone weights")
    index = backbone.get("index")
    tokenizer = pointer.get("tokenizer")
    if not isinstance(tokenizer, dict) or "json" not in tokenizer:
        raise PackageError("config.json names no tokenizer.json")
    calibration = pointer.get("calibration") or {}
    if not isinstance(calibration, dict):
        raise PackageError("config.json calibration must be an object")
    temperature = calibration.get("temperature")
    temperature_file = calibration.get("temperature_file")
    if runtime == QWEN and (temperature is None) == (temperature_file is None):
        raise PackageError(
            "a qwen3.5-decision package needs one calibration temperature"
        )
    if runtime == VELA and calibration:
        raise PackageError("vela-encoder packages are uncalibrated")
    return FileMap(
        model_name=name,
        runtime=runtime,
        model_config=_path(pointer.get("model_config"), "model_config"),
        backbone_config=_path(backbone.get("config"), "backbone.config"),
        backbone_weights=tuple(_path(item, "backbone.weights") for item in weights),
        backbone_index=None if index is None else _path(index, "backbone.index"),
        tokenizer={
            key: _path(tokenizer[key], f"tokenizer.{key}")
            for key in TOKENIZER_FILES
            if key in tokenizer
        },
        decision_weights=_paths(
            pointer.get("decision_weights"),
            VELA_WEIGHTS if runtime == VELA else QWEN_WEIGHTS,
            "decision_weights",
        ),
        temperature=None if temperature is None else _temperature(temperature),
        temperature_file=(
            None
            if temperature_file is None
            else _path(temperature_file, "calibration.temperature_file")
        ),
    )


def _temperature(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise PackageError("calibration temperatures must be finite and positive")
    return float(value)


def temperatures(files: FileMap, root: Path) -> dict[str, float] | None:
    """Per-type temperatures of a decoder package: one constant, or a file's per-type values."""
    if files.runtime != QWEN:
        return None
    if files.temperature is not None:
        return dict.fromkeys(KINDS, files.temperature)
    assert files.temperature_file is not None
    document = read_json(root / files.temperature_file)
    if not isinstance(document, dict):
        raise PackageError("the temperature file must be an object")
    per_type = document.get("temperatures")
    if isinstance(per_type, dict) and set(per_type) == set(KINDS):
        return {kind: _temperature(per_type[kind]) for kind in KINDS}
    return dict.fromkeys(KINDS, _temperature(document.get("temperature")))


def presets(root: Path) -> dict[str, dict[str, Any]]:
    """Named System One questions from ``QUESTIONS.json`` (Route's router signals); empty without one.

    A Choice entry names ``options``; a Noul entry its ``true`` and ``false``
    criteria; a Noul entry with ``categories`` and an ``instructions_template``
    becomes one question per category, named ``<entry>.<category>``.
    """
    path = Path(root) / PRESETS_FILE
    if not path.is_file():
        return {}
    document = read_json(path)
    if not isinstance(document, dict):
        raise PackageError(f"{PRESETS_FILE} must be an object of named questions")
    found: dict[str, dict[str, Any]] = {}
    for name, entry in document.items():
        kind = entry.get("type") if isinstance(entry, dict) else None
        if not isinstance(kind, str):
            raise PackageError(f"{PRESETS_FILE}: {name!r} has no type")
        kind = kind.split(",", 1)[0].strip().lower()
        if kind == "choice" and isinstance(entry.get("options"), dict):
            found[name] = {
                "type": "choice",
                "instructions": entry.get("instructions"),
                "criteria": dict(entry["options"]),
            }
        elif kind == "noul":
            criteria = {key: entry.get(key) for key in ("false", "true")}
            categories = entry.get("categories")
            template = entry.get("instructions_template")
            if isinstance(categories, dict) and isinstance(template, str):
                for category, text in categories.items():
                    found[f"{name}.{category}"] = {
                        "type": "noul",
                        "instructions": template.format(text=text),
                        "criteria": dict(criteria),
                    }
            else:
                found[name] = {
                    "type": "noul",
                    "instructions": entry.get("instructions"),
                    "criteria": criteria,
                }
        else:
            raise PackageError(
                f"{PRESETS_FILE}: {name!r} is not a Choice or Noul question"
            )
    return found


def inventory(files: FileMap, root: Path) -> list[str]:
    """Every file the family reads: the pointer, the model files and the optional extras present."""
    names = [POINTER_FILE, *files.model_files()]
    names += [name for name in (MANIFEST_FILE, PRESETS_FILE) if (root / name).is_file()]
    return sorted(set(names))


def check_manifest(root: Path, digests: dict[str, str]) -> str:
    """A package's own ``MANIFEST.json`` must agree with every loaded file it lists; returns its digest."""
    manifest = read_json(root / MANIFEST_FILE)
    listed = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(listed, dict):
        raise PackageError(f"{MANIFEST_FILE} lists no files")
    for name, digest in digests.items():
        entry = listed.get(name)
        if name == MANIFEST_FILE or entry is None:
            continue
        if not isinstance(entry, dict) or entry.get("sha256") != digest:
            raise PackageError(f"{name} differs from the package's {MANIFEST_FILE}")
    return digests[MANIFEST_FILE]


def verify(root: Path, pinned: Mapping[str, str] | None) -> Decision1Package:
    """Hash every file the family loads and check it against the pinned digests (or report a local package).

    ``pinned`` is a built-in revision's ``files``: each loaded file must be
    pinned and match. Without pins the package is served as ``local``.
    """
    root = Path(root)
    pointer = read_pointer(root)
    if pointer is None:
        raise PackageError("config.json is not a Decision 1.0 package pointer")
    files = file_map(pointer)
    digests = named_files(root, inventory(files, root))
    if pinned is not None:
        unpinned = sorted(set(digests) - set(pinned))
        if unpinned:
            raise PackageError(
                f"loaded files are not pinned by the built-in entry: {unpinned[:5]}"
            )
        changed = sorted(name for name in digests if digests[name] != pinned[name])
        if changed:
            raise PackageError(
                f"package files differ from the pinned revision: {changed[:5]}"
            )
    manifest_sha256 = check_manifest(root, digests) if MANIFEST_FILE in digests else ""
    model_config = read_json(root / files.model_config)
    backbone_config = read_json(root / files.backbone_config)
    if not isinstance(model_config, dict) or not isinstance(backbone_config, dict):
        raise PackageError("model and backbone configs must be JSON objects")
    return Decision1Package(
        root=root,
        files=files,
        model_config=model_config,
        backbone_config=backbone_config,
        digests=digests,
        model_sha256=sha256_json({name: digests[name] for name in files.model_files()}),
        manifest_sha256=manifest_sha256,
        verification="pinned" if pinned is not None else "local",
        temperatures=temperatures(files, root),
        presets=presets(root),
    )
