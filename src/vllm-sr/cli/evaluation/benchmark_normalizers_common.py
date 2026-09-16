"""Shared invariants used by explicit benchmark-family parsers."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from cli.evaluation.benchmark_normalization_io import (
    NormalizationError,
    exact_object,
    load_json,
    no_duplicate,
    require_array,
    string,
)
from cli.evaluation.benchmark_normalization_types import (
    BenchmarkNormalizerDescriptor,
)
from cli.evaluation.canonical import digest_value
from cli.evaluation.case_plan import applicable_track_ids
from cli.evaluation.contract_primitives import Message
from cli.evaluation.contracts import CaseGrading, CaseVisible

_MIN_DENSE_POOL_MODELS = 2


def opaque_id(kind: str, *parts: object) -> str:
    encoded = "\x00".join(str(part) for part in parts).encode("utf-8")
    return f"{kind}-{hashlib.sha256(encoded).hexdigest()[:24]}"


def arm_map(models: Iterable[str]) -> dict[str, str]:
    ordered = tuple(sorted(models))
    no_duplicate(ordered, "model manifest")
    if len(ordered) < _MIN_DENSE_POOL_MODELS:
        raise NormalizationError("dense model-pool export requires at least two models")
    return {model: opaque_id("arm", model) for model in ordered}


def load_model_manifest(path: Path, *, max_bytes: int) -> tuple[str, ...]:
    payload = exact_object(
        load_json(path, max_bytes=max_bytes),
        required={"models"},
        label="model manifest",
    )
    values = require_array(payload["models"], "model manifest models")
    models = tuple(string(item, "model manifest model") for item in values)
    no_duplicate(models, "model manifest")
    if tuple(sorted(models)) != models:
        raise NormalizationError("model manifest must use canonical lexical order")
    return models


def text_case(
    source_key: str,
    prompt: str,
    *,
    descriptor: BenchmarkNormalizerDescriptor,
    tags: Iterable[str] = (),
    expected_route: str | None = None,
    expected_answer: str | None = None,
    trajectory_id: str | None = None,
) -> tuple[CaseVisible, CaseGrading]:
    case_id = opaque_id("case", source_key)
    return (
        CaseVisible(
            id=case_id,
            track_ids=applicable_track_ids(
                descriptor.track_ids,
                modality="text",
            ),
            messages=(Message(role="user", content=prompt),),
            tags=tuple(tags),
            trajectory_id=trajectory_id,
        ),
        CaseGrading(
            case_id=case_id,
            expected_route=expected_route,
            expected_answer=expected_answer,
        ),
    )


def native_digest(value: Mapping[str, Any] | list[Any]) -> str:
    return digest_value(value)


def messages(value: Any, label: str) -> tuple[Message, ...]:
    rows = require_array(value, label)
    parsed: list[Message] = []
    for index, item in enumerate(rows):
        row = exact_object(
            item,
            required={"role", "content"},
            label=f"{label}[{index}]",
        )
        parsed.append(
            Message(
                role=string(row["role"], f"{label}[{index}].role"),
                content=string(row["content"], f"{label}[{index}].content"),
            )
        )
    if not parsed:
        raise NormalizationError(f"{label} must not be empty")
    return tuple(parsed)
