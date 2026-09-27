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
from .release_artifacts import ReleaseArtifactError, select_qwen_weight_files
from .runtime_profile import RuntimeProfileError, validate_relative_artifact_path


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

    Paths are constrained to the installed family loader's stable layout. A
    changing number of Qwen weight shards is selected from the descriptor;
    neither a particular model commit nor weight digest is built into code.
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
    prefix = "" if is_qwen else "native/"
    backbone_directory = f"{prefix}{'backbone' if is_qwen else 'encoder'}"
    tokenizer_directory = "" if is_qwen else "native/tokenizer/"

    model_config = _path(root["model_config"], "model_config")
    if model_config != f"{prefix}decision_config.json":
        raise DecisionConfigError("Decision model config layout is unsupported")

    backbone = _mapping(root["backbone"], "backbone")
    if set(backbone) not in ({"config", "weights"}, {"config", "weights", "index"}):
        raise DecisionConfigError("Decision backbone fields are unsupported")
    backbone_config = _path(backbone["config"], "backbone.config")
    if backbone_config != f"{backbone_directory}/config.json":
        raise DecisionConfigError("Decision backbone config layout is unsupported")
    weights = backbone["weights"]
    if (
        not isinstance(weights, list)
        or not weights
        or any(not isinstance(item, str) for item in weights)
    ):
        raise DecisionConfigError("Decision backbone weights must be a nonempty list")
    backbone_weights = tuple(_path(item, "backbone.weights") for item in weights)
    if any(
        PurePosixPath(item).parent.as_posix() != backbone_directory
        or not item.endswith(".safetensors")
        for item in backbone_weights
    ):
        raise DecisionConfigError("Decision backbone weight layout is unsupported")
    backbone_index = (
        _path(backbone["index"], "backbone.index") if "index" in backbone else None
    )
    if is_qwen:
        try:
            selected_weights = select_qwen_weight_files(
                set(backbone_weights) | ({backbone_index} if backbone_index else set())
            )
        except ReleaseArtifactError as error:
            raise DecisionConfigError(str(error)) from error
        if set(selected_weights) != set(backbone_weights) | (
            {backbone_index} if backbone_index else set()
        ):
            raise DecisionConfigError("Decision Qwen weight layout is unsupported")
    elif (
        backbone_weights != ("native/encoder/model.safetensors",)
        or backbone_index is not None
    ):
        raise DecisionConfigError("Decision Vela weight layout is unsupported")

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
        tokenizer_json != f"{tokenizer_directory}tokenizer.json"
        or tokenizer_config != f"{tokenizer_directory}tokenizer_config.json"
    ):
        raise DecisionConfigError("Decision tokenizer layout is unsupported")
    if (
        special_tokens_map is not None
        and special_tokens_map != f"{tokenizer_directory}special_tokens_map.json"
    ):
        raise DecisionConfigError(
            "Decision tokenizer special tokens layout is unsupported"
        )
    if chat_template is not None and (
        not is_qwen or chat_template != "chat_template.jinja"
    ):
        raise DecisionConfigError(
            "Decision tokenizer chat template layout is unsupported"
        )

    raw_decision_weights = _mapping(root["decision_weights"], "decision_weights")
    expected_weights = (
        {"decision_head": "decision_head.safetensors"}
        if is_qwen
        else {
            "choice_encoder": "native/choice_encoder.safetensors",
            "score_encoder": "native/score_encoder.safetensors",
            "decision_heads": "native/decision_heads.safetensors",
        }
    )
    if set(raw_decision_weights) != set(expected_weights):
        raise DecisionConfigError("Decision head weight fields are unsupported")
    decision_weights = {
        key: _path(value, f"decision_weights.{key}")
        for key, value in raw_decision_weights.items()
    }
    if decision_weights != expected_weights:
        raise DecisionConfigError("Decision head weight layout is unsupported")

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
            if temperature_file != "temperature.json":
                raise DecisionConfigError(
                    "Decision calibration file layout is unsupported"
                )
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
