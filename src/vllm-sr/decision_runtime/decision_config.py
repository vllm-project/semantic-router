"""Strict, data-only contract for published Decision model repositories."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any

from .family_registry import family_registration
from .runtime_profile import RuntimeProfileError, validate_relative_artifact_path

_RESERVED_PATHS = frozenset({"config.json", ".vllm-sr-artifact.json"})


class DecisionConfigError(ValueError):
    """A published Decision root config is malformed or unsupported."""


@dataclass(frozen=True, slots=True)
class DecisionRepositoryConfig:
    model_name: str
    family: str
    model_config: str
    backbone_config: str
    backbone_weights: tuple[str, ...]
    backbone_index: str | None
    tokenizer_json: str
    tokenizer_config: str
    tokenizer_special_tokens_map: str | None
    tokenizer_chat_template: str | None
    decision_weights: Mapping[str, str]
    temperature: float | None
    temperature_file: str | None
    files: tuple[str, ...]

    @property
    def data_root_relative(self) -> str:
        parent = PurePosixPath(self.model_config).parent
        return "" if parent == PurePosixPath(".") else parent.as_posix()


def parse_decision_config(
    payload: bytes, *, model_name: str, family: str
) -> DecisionRepositoryConfig:
    """Validate a model-owned root descriptor without trusting repository code.

    Paths are authoritative except where Transformers requires conventional
    filenames for local discovery. Neither a particular model commit nor a
    weight digest is built into code.
    """

    try:
        document = json.loads(payload, object_pairs_hook=_unique_object)
    except (TypeError, ValueError, RecursionError) as error:
        raise DecisionConfigError("Decision config is not valid JSON") from error
    root = _mapping(document, "config")
    required = {
        "decision_format",
        "format_version",
        "model_name",
        "runtime_family",
        "model_config",
        "backbone",
        "tokenizer",
        "decision_weights",
    }
    if set(root) not in (required, required | {"calibration"}):
        raise DecisionConfigError("Decision config fields are unsupported")
    if (
        root["decision_format"] != "vllm-sr-decision"
        or type(root["format_version"]) is not int
        or root["format_version"] != 1
    ):
        raise DecisionConfigError("Decision config format is unsupported")
    if root["model_name"] != model_name:
        raise DecisionConfigError(
            "Decision config model name does not match the catalog"
        )
    registration = family_registration(family)
    if (
        registration is None
        or root["runtime_family"] != registration.config_runtime_family
    ):
        raise DecisionConfigError(
            "Decision config runtime family does not match the profile"
        )

    is_qwen = family == "qwen3.5"
    model_config = _path(root["model_config"], "model_config")
    if not model_config.endswith(".json"):
        raise DecisionConfigError("Decision model config must be JSON")
    data_root = PurePosixPath(model_config).parent

    backbone = _mapping(root["backbone"], "backbone")
    if set(backbone) not in ({"config", "weights"}, {"config", "weights", "index"}):
        raise DecisionConfigError("Decision backbone fields are unsupported")
    backbone_config = _path(backbone["config"], "backbone.config")
    backbone_directory = PurePosixPath(backbone_config).parent
    if PurePosixPath(backbone_config).name != "config.json":
        raise DecisionConfigError("Transformers backbone config must be config.json")
    weights = backbone["weights"]
    if (
        not isinstance(weights, list)
        or not weights
        or any(not isinstance(item, str) for item in weights)
    ):
        raise DecisionConfigError("Decision backbone weights must be a nonempty list")
    backbone_weights = tuple(_path(item, "backbone.weights") for item in weights)
    if any(
        PurePosixPath(item).parent != backbone_directory
        or not item.endswith(".safetensors")
        for item in backbone_weights
    ):
        raise DecisionConfigError("Decision backbone weights must share one directory")
    backbone_index = (
        _path(backbone["index"], "backbone.index") if "index" in backbone else None
    )
    if is_qwen:
        if len(backbone_weights) == 1 and backbone_index is None:
            if PurePosixPath(backbone_weights[0]).name != "model.safetensors":
                raise DecisionConfigError(
                    "Transformers single weight must be model.safetensors"
                )
        elif (
            backbone_index is None
            or PurePosixPath(backbone_index).parent != backbone_directory
            or PurePosixPath(backbone_index).name != "model.safetensors.index.json"
        ):
            raise DecisionConfigError("Decision Qwen indexed weight layout is invalid")
    elif len(backbone_weights) != 1 or backbone_index is not None:
        raise DecisionConfigError("Decision Vela requires one encoder weight file")

    tokenizer = _mapping(root["tokenizer"], "tokenizer")
    tokenizer_required = {"json", "config"}
    tokenizer_optional = {"special_tokens_map", "chat_template"}
    if (
        not tokenizer_required <= set(tokenizer)
        or not set(tokenizer) <= tokenizer_required | tokenizer_optional
    ):
        raise DecisionConfigError("Decision tokenizer fields are unsupported")
    tokenizer_json = _path(tokenizer["json"], "tokenizer.json")
    tokenizer_config = _path(tokenizer["config"], "tokenizer.config")
    tokenizer_directory = PurePosixPath(tokenizer_json).parent
    special_tokens_map = (
        _path(tokenizer["special_tokens_map"], "tokenizer.special_tokens_map")
        if "special_tokens_map" in tokenizer
        else None
    )
    chat_template = (
        _path(tokenizer["chat_template"], "tokenizer.chat_template")
        if "chat_template" in tokenizer
        else None
    )
    if (
        PurePosixPath(tokenizer_json).name != "tokenizer.json"
        or PurePosixPath(tokenizer_config).name != "tokenizer_config.json"
        or PurePosixPath(tokenizer_config).parent != tokenizer_directory
    ):
        raise DecisionConfigError("Transformers tokenizer layout is unsupported")
    if special_tokens_map is not None and (
        PurePosixPath(special_tokens_map).name != "special_tokens_map.json"
        or PurePosixPath(special_tokens_map).parent != tokenizer_directory
    ):
        raise DecisionConfigError(
            "Decision tokenizer special tokens layout is unsupported"
        )
    if chat_template is not None and (
        PurePosixPath(chat_template).name != "chat_template.jinja"
        or PurePosixPath(chat_template).parent != tokenizer_directory
    ):
        raise DecisionConfigError(
            "Decision tokenizer chat template layout is unsupported"
        )

    raw_decision_weights = _mapping(root["decision_weights"], "decision_weights")
    expected_roles = (
        {"decision_head"}
        if is_qwen
        else {"choice_encoder", "score_encoder", "decision_heads"}
    )
    if set(raw_decision_weights) != expected_roles:
        raise DecisionConfigError("Decision head weight fields are unsupported")
    decision_weights = {
        key: _path(value, f"decision_weights.{key}")
        for key, value in raw_decision_weights.items()
    }
    if any(not path.endswith(".safetensors") for path in decision_weights.values()):
        raise DecisionConfigError("Decision head weights must be safetensors")

    temperature: float | None = None
    temperature_file: str | None = None
    if is_qwen:
        calibration = _mapping(root.get("calibration"), "calibration")
        if set(calibration) == {"temperature"}:
            temperature = _positive_finite(calibration["temperature"])
        elif set(calibration) == {"temperature_file"}:
            temperature_file = _path(
                calibration["temperature_file"], "calibration.temperature_file"
            )
            if not temperature_file.endswith(".json"):
                raise DecisionConfigError("Decision calibration file must be JSON")
        else:
            raise DecisionConfigError("Decision Qwen calibration is unsupported")
    elif "calibration" in root:
        raise DecisionConfigError("Decision Vela calibration is unsupported")

    paths = (
        model_config,
        backbone_config,
        *backbone_weights,
        *((backbone_index,) if backbone_index else ()),
        tokenizer_json,
        tokenizer_config,
        *((special_tokens_map,) if special_tokens_map else ()),
        *((chat_template,) if chat_template else ()),
        *decision_weights.values(),
        *((temperature_file,) if temperature_file else ()),
    )
    if len(paths) != len(set(paths)):
        raise DecisionConfigError("Decision config contains duplicate file paths")
    if any(
        path in _RESERVED_PATHS or not _within_data_root(path, data_root)
        for path in paths
    ):
        raise DecisionConfigError("Decision config files must stay in their data root")
    return DecisionRepositoryConfig(
        model_name=model_name,
        family=family,
        model_config=model_config,
        backbone_config=backbone_config,
        backbone_weights=backbone_weights,
        backbone_index=backbone_index,
        tokenizer_json=tokenizer_json,
        tokenizer_config=tokenizer_config,
        tokenizer_special_tokens_map=special_tokens_map,
        tokenizer_chat_template=chat_template,
        decision_weights=MappingProxyType(decision_weights),
        temperature=temperature,
        temperature_file=temperature_file,
        files=tuple(sorted(paths)),
    )


def validate_decision_weight_index(
    payload: bytes, config: DecisionRepositoryConfig
) -> None:
    """Ensure a Transformers shard index cannot select undeclared files."""

    if config.backbone_index is None:
        return
    try:
        document = json.loads(payload, object_pairs_hook=_unique_object)
    except (TypeError, ValueError, RecursionError) as error:
        raise DecisionConfigError("Decision weight index is not valid JSON") from error
    root = _mapping(document, "weight index")
    if set(root) not in ({"weight_map"}, {"metadata", "weight_map"}):
        raise DecisionConfigError("Decision weight index fields are unsupported")
    weight_map = _mapping(root["weight_map"], "weight_map")
    if not weight_map or any(not isinstance(key, str) or not key for key in weight_map):
        raise DecisionConfigError("Decision weight index map is empty or invalid")
    expected = {PurePosixPath(item).name for item in config.backbone_weights}
    if (
        any(not isinstance(value, str) for value in weight_map.values())
        or set(weight_map.values()) != expected
    ):
        raise DecisionConfigError("Decision weight index differs from config shards")
    if "metadata" in root:
        _mapping(root["metadata"], "weight index metadata")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DecisionConfigError("Decision JSON contains duplicate keys")
        result[key] = value
    return result


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise DecisionConfigError(f"{field} must be an object")
    return value


def _path(value: object, field: str) -> str:
    try:
        return validate_relative_artifact_path(value, field=field)
    except RuntimeProfileError as error:
        raise DecisionConfigError(str(error)) from error


def _within_data_root(path: str, data_root: PurePosixPath) -> bool:
    if data_root == PurePosixPath("."):
        return True
    return PurePosixPath(path).is_relative_to(data_root)


def _positive_finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionConfigError("Decision calibration temperature is invalid")
    try:
        temperature = float(value)
    except OverflowError as error:
        raise DecisionConfigError(
            "Decision calibration temperature is invalid"
        ) from error
    if not math.isfinite(temperature) or temperature <= 0:
        raise DecisionConfigError("Decision calibration temperature is invalid")
    return temperature
