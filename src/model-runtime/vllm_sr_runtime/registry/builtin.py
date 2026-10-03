"""First-party models the runtime serves out of the box, pinned by revision and identity.

A new model revision is a new entry; a built-in model is never resolved
through a moving branch. Golden requests gate readiness; their reference
answers per device class live in ``golden_answers.json``, recorded with
``tools/golden_answers.py`` for the pinned revision. ``kernel_choices.json``
holds, per device class, the autotuned kernel configurations the released
runtime ran with (``tools/kernel_choices.py``); the runtime pins them so every
process computes the released numerics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

ORG = "vllm-sr"


@dataclass(frozen=True)
class BuiltinModel:
    repo_id: str
    revision: str
    family: str
    model_sha256: str
    manifest_sha256: str
    loaded_parameters: int
    backbone: str
    min_device_memory_gib: float
    base: tuple[str, str] | None = None
    golden_answers: dict[str, Any] = field(default_factory=dict)
    kernel_choices: dict[str, Any] = field(default_factory=dict)


DECISION2_MODELS = (
    BuiltinModel(
        repo_id=f"{ORG}/Decision-2.0-Kai-0.6B",
        revision="cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764",
        family="decision2",
        model_sha256="bb806f31a14d4532a1b5e00984442128f47f46cbbc68e57e7623327c81cfb983",
        manifest_sha256="ca7bf2cff1494a5bfc51f4271f52557da0f3ce3b6f0f6e7a2b4118cb4965e92e",
        loaded_parameters=597_103_104,
        backbone="qwen3",
        min_device_memory_gib=4,
    ),
    BuiltinModel(
        repo_id=f"{ORG}/Decision-2.0-Eos-0.8B",
        revision="3594047d69f476f1d01cf84c593e213fc3a4dfe0",
        family="decision2",
        model_sha256="d97127991870ae202017a99eb496bdc56a62afd038594a67febfab0aea9d76fe",
        manifest_sha256="e8b1081be4a76deca5247792a4031c8e19e2b52b95d91775c13d9e257c407101",
        loaded_parameters=753_446_208,
        backbone="qwen3_5_text",
        min_device_memory_gib=4,
    ),
    BuiltinModel(
        repo_id=f"{ORG}/Decision-2.0-Sol-2B",
        revision="64235bef55dad29387dd16da7c90e038bf2f0972",
        family="decision2",
        model_sha256="e20df76c14edb0d28ef17593f49a1870fdea9438ff4805594402e39766f3546c",
        manifest_sha256="a91c440ca371668d488a788bdfe12cc574a97123df60730afedf419737ced970",
        loaded_parameters=1_883_930_944,
        backbone="qwen3_5_text",
        min_device_memory_gib=8,
    ),
    BuiltinModel(
        repo_id=f"{ORG}/Decision-2.0-Nox-4B",
        revision="25e8f67d1b486c647222df3aac640d2d5d736bbe",
        family="decision2",
        model_sha256="31d4ee9213686eeed6283d08ed7bce52ebef94ec47b7f97f7e1209f916fe51ea",
        manifest_sha256="49771ea33a451274687ad2a246ce5fb78f8f42ec904fc91a467477179975bb16",
        loaded_parameters=4_208_383_488,
        backbone="qwen3_5_text",
        min_device_memory_gib=16,
    ),
    BuiltinModel(
        repo_id=f"{ORG}/Decision-2.0-Lux-9B",
        revision="78bf3c03d9147aeb30b641edfe0e30ed04887ca5",
        family="decision2",
        model_sha256="0ece5faa210173f443f353474b419fd913a253b06645da3e8370c2db0e1c3339",
        manifest_sha256="4e0f4ca0cf3c833933f9bed3021012d061910016c9f4ca43b34f5d66e8d0a260",
        loaded_parameters=7_940_895_744,
        backbone="qwen3_5_text",
        min_device_memory_gib=24,
    ),
    BuiltinModel(
        repo_id=f"{ORG}/Decision-2.0-Vega-27B",
        revision="7aec49ae11a18741706da549ab626b9052795fe7",
        family="decision2",
        model_sha256="584697310502cd20e91728e69f724c028b47b95daf3790256dea3e434c48be82",
        manifest_sha256="24dcf251f5b50a2259201cc5e11889b6d2545afd4ccecabb0e2344549542bdd3",
        loaded_parameters=29_365_153_792,
        backbone="qwen3_5_text",
        min_device_memory_gib=72,
        base=("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
    ),
)


def _with_recorded(
    models: tuple[BuiltinModel, ...], file_name: str, entry_field: str, model_field: str
) -> tuple[BuiltinModel, ...]:
    recorded = json.loads(
        Path(__file__).with_name(file_name).read_text(encoding="utf-8")
    )
    result = []
    for model in models:
        entry = recorded.get(model.repo_id)
        if entry and entry.get("revision") == model.revision:
            result.append(replace(model, **{model_field: entry[entry_field]}))
        else:
            result.append(model)
    return tuple(result)


DECISION2_MODELS = _with_recorded(
    DECISION2_MODELS, "golden_answers.json", "answers", "golden_answers"
)
DECISION2_MODELS = _with_recorded(
    DECISION2_MODELS, "kernel_choices.json", "choices", "kernel_choices"
)

_BY_REPO = {model.repo_id.lower(): model for model in DECISION2_MODELS}
_BY_NAME = {model.repo_id.split("/", 1)[1].lower(): model for model in DECISION2_MODELS}


def lookup(model: str) -> BuiltinModel | None:
    """A built-in model by repository ID or bare model name (case-insensitive)."""
    key = model.strip().lower()
    return _BY_REPO.get(key) or _BY_NAME.get(key)


def by_identity(model_sha256: str) -> BuiltinModel | None:
    for model in DECISION2_MODELS:
        if model.model_sha256 == model_sha256:
            return model
    return None


def all_models() -> tuple[BuiltinModel, ...]:
    return DECISION2_MODELS
