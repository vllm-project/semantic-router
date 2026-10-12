"""The Decision 2.0 package format (``dev2-package/1``), verified without running package code.

A package is the scored model files at checkpoint-relative paths, a root
``config.json`` pointer, a ``MODEL_MANIFEST.json`` inventory with per-file
SHA-256, a card, and bundled Python that this runtime never imports. The
checks reproduce the released runtime's ``verify_bundle`` and
``checkpoint_fingerprint``.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...errors import PackageError
from ...registry.artifacts import (
    inventory,
    read_json,
    safetensors_elements,
    safetensors_header,
    sha256_file,
    sha256_json,
)
from ...systemone import MAX_OPTIONS, MIN_OPTIONS

POINTER = {
    "decision_format": "vllm-sr-decision",
    "format_version": 2,
    "package_schema": "dev2-package/1",
}
MANIFEST_NAME = "MODEL_MANIFEST.json"
MANIFEST_SCHEMA = "dev2-package-manifest/1"
SUPPORTED_PROFILES = ("qwen-full", "qwen-adapter")
KNOWN_PROFILES = ("kai-native", "qwen-full", "qwen-adapter", "encoder-marker")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REVISION = re.compile(r"[0-9a-f]{40}\Z")

PROMPT_VERSION = "decision2-segmented-options-global-query-v1"
ARCHITECTURES = {
    "qwen3-text-endpoints-global-query-shared-bilinear-mlp": "qwen3",
    "qwen3.5-text-endpoints-global-query-shared-bilinear-mlp": "qwen3_5_text",
}
LORA_FORMAT = "peft-lora/1"
# checkpoint_fingerprint: root files that are model inputs, and model suffixes under backbone/ and adapter/.
MODEL_ROOT_FILES = {
    "decision_config.json",
    "decision_head.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "chat_template.jinja",
}
MODEL_SUFFIXES = {".json", ".safetensors", ".bin", ".model", ".txt"}
SOURCE_SUFFIXES = {".json", ".safetensors", ".bin", ".model", ".txt"}
SCORE_BIAS_FORMAT = "dev2-score-bias-v1"
CALIBRATION_VERSION = "decision2-per-type-temperature/1"
TASK_TYPES = ("choice", "noul", "score")
MIN_INPUT_TOKENS = 16
MIN_TEMPERATURE, MAX_TEMPERATURE = 0.05, 20.0


@dataclass(frozen=True)
class Decision2Package:
    """What the family needs after verification."""

    root: Path
    manifest: dict[str, Any]
    pointer: dict[str, Any]
    decision_config: dict[str, Any]
    profile: str
    backbone_type: str
    base_root: Path | None
    model_sha256: str
    temperatures: dict[str, float]
    score_bias: dict[int, list[float]] | None


def is_package(root: Path) -> bool:
    pointer_path = Path(root) / "config.json"
    if not pointer_path.is_file():
        return False
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except ValueError:
        return False
    return isinstance(pointer, dict) and all(
        pointer.get(key) == value for key, value in POINTER.items()
    )


def verify_manifest(root: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Pointer, exact inventory, per-file SHA-256 and header tensor counts; no weights loaded."""
    pointer = read_json(root / "config.json")
    if not isinstance(pointer, dict) or any(
        pointer.get(key) != value for key, value in POINTER.items()
    ):
        raise PackageError("config.json is not a Decision 2.0 package pointer")
    manifest = read_json(root / MANIFEST_NAME)
    files = manifest.get("files_sha256") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("profile") not in KNOWN_PROFILES
        or not isinstance(files, dict)
        or MANIFEST_NAME in files
        or any(
            not isinstance(value, str) or not SHA256.fullmatch(value)
            for value in files.values()
        )
    ):
        raise PackageError("unknown or malformed Decision 2.0 package manifest")
    if manifest["profile"] not in SUPPORTED_PROFILES:
        raise PackageError(
            f"package profile {manifest['profile']!r} is not served by the decision2 family yet"
        )
    manifest_sha256 = sha256_file(root / MANIFEST_NAME)
    actual = inventory(root)
    expected = {**files, MANIFEST_NAME: manifest_sha256}
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(
            name
            for name in set(actual) & set(expected)
            if actual[name] != expected[name]
        )
        raise PackageError(
            "package files differ from MODEL_MANIFEST.json"
            f" (missing {missing[:5]}, extra {extra[:5]}, changed {changed[:5]})"
        )
    if pointer.get("model_name") != manifest.get("model_name"):
        raise PackageError("pointer and manifest name different models")
    parameters = manifest.get("parameters")
    if not isinstance(parameters, dict) or not isinstance(
        parameters.get("packaged_files"), dict
    ):
        raise PackageError("manifest has no packaged parameter counts")
    counted = {
        group: sum(safetensors_elements(root / name) for name in names)
        for group, names in parameters["packaged_files"].items()
    }
    if counted != parameters.get("packaged"):
        raise PackageError("packaged tensor counts differ from the manifest")
    limit = manifest.get("max_input_tokens")
    if type(limit) is not int or limit < MIN_INPUT_TOKENS:
        raise PackageError("manifest max_input_tokens must be an integer >= 16")
    return pointer, manifest, manifest_sha256


def checkpoint_files(root: Path) -> dict[str, str]:
    """The model-input files the scored identity covers (``checkpoint_fingerprint``)."""
    files = [
        path
        for path in root.iterdir()
        if path.is_file() and path.name in MODEL_ROOT_FILES
    ]
    for folder in ("backbone", "adapter"):
        subtree = root / folder
        if subtree.is_dir():
            files.extend(
                path
                for path in subtree.rglob("*")
                if path.is_file() and path.suffix in MODEL_SUFFIXES
            )
    return {
        path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(files)
    }


def source_files(base_root: Path) -> dict[str, str]:
    """``source_fingerprint`` of a pinned base: root model files (and a backbone/ subtree)."""
    candidates = list(base_root.iterdir())
    if (base_root / "backbone").is_dir():
        candidates.extend((base_root / "backbone").rglob("*"))
    files = [
        path
        for path in candidates
        if path.is_file()
        and path.suffix in SOURCE_SUFFIXES
        and path.name not in {"trainer_state.pt", "checkpoint.json"}
    ]
    if not any(path.suffix in {".safetensors", ".bin"} for path in files):
        raise PackageError("the pinned base contains no weight files")
    return {
        path.relative_to(base_root).as_posix(): sha256_file(path)
        for path in sorted(files)
    }


def model_identity(
    root: Path, decision_config: dict[str, Any], base_root: Path | None
) -> str:
    hashes = checkpoint_files(root)
    if decision_config.get("checkpoint_format") == LORA_FORMAT:
        if base_root is None:
            raise PackageError("an adapter package needs its pinned base")
        contract = decision_config.get("lora")
        expected = (
            contract.get("source_fingerprint") if isinstance(contract, dict) else None
        )
        if not isinstance(expected, dict) or not isinstance(
            expected.get("files_sha256"), dict
        ):
            raise PackageError("adapter package has no base fingerprint")
        actual = source_files(base_root)
        if actual != expected["files_sha256"]:
            raise PackageError(
                "base files differ from the adapter's source fingerprint"
            )
        hashes = {
            **{f"checkpoint/{key}": value for key, value in hashes.items()},
            **{f"source/{key}": value for key, value in actual.items()},
        }
    elif not any(
        name.startswith("backbone/") and name.endswith((".safetensors", ".bin"))
        for name in hashes
    ):
        raise PackageError("checkpoint is missing backbone weight files")
    return sha256_json(hashes)


def verify_base(base_root: Path, base: dict[str, Any]) -> None:
    expected = base.get("files_sha256")
    if not isinstance(expected, dict) or not expected:
        raise PackageError("manifest base lists no files")
    for name, digest in expected.items():
        path = base_root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise PackageError(f"base file differs from the pinned revision: {name}")


def check_decision_config(config: Any) -> str:
    """The backbone model type of a supported Decision 2.0 checkpoint."""
    if not isinstance(config, dict):
        raise PackageError("decision_config.json must be an object")
    if config.get("prompt_version") != PROMPT_VERSION:
        raise PackageError(
            f"unsupported prompt version {config.get('prompt_version')!r}"
        )
    architecture = config.get("architecture")
    if architecture not in ARCHITECTURES:
        raise PackageError(f"unsupported Decision 2.0 architecture {architecture!r}")
    if config.get("head_variant", "shared") != "shared":
        raise PackageError("only the shared candidate head is supported")
    if config.get("dec_residual") is not None or config.get("readout") == "label_token":
        raise PackageError("residual and label-token readouts are not supported yet")
    if config.get("checkpoint_format") not in (None, "full", LORA_FORMAT):
        raise PackageError(
            f"unknown checkpoint format {config.get('checkpoint_format')!r}"
        )
    if type(config.get("head_dim")) is not int or config["head_dim"] < 1:
        raise PackageError("decision_config.json needs a positive head_dim")
    return ARCHITECTURES[architecture]


def load_score_bias(
    root: Path, manifest: dict[str, Any], model_sha256: str
) -> dict[int, list[float]] | None:
    entry = manifest.get("score_bias")
    if entry is None:
        return None
    if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
        raise PackageError("manifest score_bias entry is malformed")
    path = root / entry["file"]
    if (
        manifest["files_sha256"].get(entry["file"]) != entry.get("sha256")
        or sha256_file(path) != entry["sha256"]
    ):
        raise PackageError("score offsets differ from the packaged score_bias file")
    report = read_json(path)
    if not isinstance(report, dict) or report.get("format") != SCORE_BIAS_FORMAT:
        raise PackageError("unknown score bias format")
    if set(report) != {"format", "model_sha256", "offsets", "fit"} or not isinstance(
        report["fit"], dict
    ):
        raise PackageError("score bias keys differ from the format")
    if report.get("model_sha256") != model_sha256:
        raise PackageError("score bias model hash differs from the checkpoint")
    offsets = validate_offsets(report.get("offsets"))
    if {str(key): value for key, value in offsets.items()} != entry.get("offsets"):
        raise PackageError("score offsets differ from MODEL_MANIFEST.json")
    return offsets


def validate_offsets(value: Any) -> dict[int, list[float]]:
    if not isinstance(value, dict) or not value:
        raise PackageError("score bias offsets must be a nonempty object")
    offsets: dict[int, list[float]] = {}
    for key, row in value.items():
        if (
            not isinstance(key, str)
            or not key.isdigit()
            or str(int(key)) != key
            or not MIN_OPTIONS <= int(key) <= MAX_OPTIONS
        ):
            raise PackageError(
                f"score bias level count {key!r} is not in 2..{MAX_OPTIONS}"
            )
        if not isinstance(row, list) or len(row) != int(key):
            raise PackageError(
                f"score bias offsets for L={key} need exactly {key} values"
            )
        if any(
            type(item) not in (int, float) or not math.isfinite(item) for item in row
        ):
            raise PackageError(f"score bias offsets for L={key} must be finite numbers")
        offsets[int(key)] = [float(item) for item in row]
    return offsets


def load_temperatures(
    root: Path, manifest: dict[str, Any], model_sha256: str
) -> dict[str, float]:
    calibration = manifest.get("calibration")
    if not calibration:
        return dict.fromkeys(TASK_TYPES, 1.0)
    if not isinstance(calibration, dict) or not isinstance(
        calibration.get("file"), str
    ):
        raise PackageError("manifest calibration entry is malformed")
    report = read_json(root / calibration["file"])
    if (
        not isinstance(report, dict)
        or report.get("calibration_version") != CALIBRATION_VERSION
    ):
        raise PackageError("unknown Decision 2.0 calibration format")
    if report.get("model_sha256") != model_sha256:
        raise PackageError("calibration model hash differs from the checkpoint")
    inference = report.get("inference") or {}
    if inference.get("max_length") not in (None, manifest["max_input_tokens"]):
        raise PackageError("calibration context differs from the package limit")
    values = report.get("temperature_by_type")
    if not isinstance(values, dict) or set(values) != set(TASK_TYPES):
        raise PackageError("calibration needs a temperature for choice, noul and score")
    temperatures = {}
    for kind in TASK_TYPES:
        value = values[kind]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not MIN_TEMPERATURE <= value <= MAX_TEMPERATURE
        ):
            raise PackageError(f"calibration temperature for {kind} is out of range")
        temperatures[kind] = float(value)
    return temperatures


def verify_adapter(root: Path, decision_config: dict[str, Any]) -> dict[str, Any]:
    """``verify_adapter_config``: the adapter matches the LoRA contract in decision_config."""
    contract = decision_config.get("lora")
    if not isinstance(contract, dict):
        raise PackageError("adapter package has no LoRA contract")
    config = read_json(root / "adapter" / "adapter_config.json")
    targets = contract.get("target_modules")
    if (
        not isinstance(config, dict)
        or config.get("peft_type") != "LORA"
        or not isinstance(targets, list)
    ):
        raise PackageError("adapter is not a PEFT LoRA adapter")
    full_targets, suffix_targets = set(targets), {
        name.rsplit(".", 1)[-1] for name in targets
    }
    serialized = config.get("target_modules")
    if (
        config.get("r") != contract.get("rank")
        or config.get("lora_alpha") != contract.get("alpha")
        or config.get("bias") != "none"
        or not isinstance(serialized, list)
        or set(serialized) not in (full_targets, suffix_targets)
    ):
        raise PackageError("adapter config differs from the Decision 2.0 LoRA contract")
    dimensions = contract.get("target_dimensions")
    if not isinstance(dimensions, dict) or set(dimensions) != full_targets:
        raise PackageError("LoRA contract has no complete projection dimensions")
    header = {
        name: value
        for name, value in safetensors_header(
            root / "adapter" / "adapter_model.safetensors"
        ).items()
        if name != "__metadata__"
    }
    expected = {}
    for name, (in_features, out_features) in dimensions.items():
        stem = f"base_model.model.{name}"
        expected[f"{stem}.lora_A.weight"] = [contract["rank"], in_features]
        expected[f"{stem}.lora_B.weight"] = [out_features, contract["rank"]]
    if set(header) != set(expected) or any(
        header[name].get("shape") != shape or header[name].get("dtype") != "F32"
        for name, shape in expected.items()
    ):
        raise PackageError("adapter tensors differ from the pinned projections")
    return config
