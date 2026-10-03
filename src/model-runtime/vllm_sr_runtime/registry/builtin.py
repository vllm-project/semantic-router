"""First-party models the runtime serves out of the box, pinned by revision and identity.

A new model revision is a new entry; a built-in model is never resolved
through a moving branch. ``golden`` requests gate readiness; reference
answers per device class are recorded by the parity runs
(``docs/records/``) and added here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


DECISION2_MODELS = (
    BuiltinModel(
        repo_id=f"{ORG}/Decision-2.0-Kai-0.6B",
        revision="881bee413681d80ebeac86afcda8b4138dae516e",
        family="decision2",
        model_sha256="bb806f31a14d4532a1b5e00984442128f47f46cbbc68e57e7623327c81cfb983",
        manifest_sha256="605ec7fe150e61adcaf57de8314ab2394819c0dea04189258142d84b6cb33638",
        loaded_parameters=597_103_104,
        backbone="qwen3",
        min_device_memory_gib=4,
    ),
    BuiltinModel(
        repo_id=f"{ORG}/Decision-2.0-Eos-0.8B",
        revision="ad0aa724c924f7c4194be94b1b8441caf2d61c01",
        family="decision2",
        model_sha256="d97127991870ae202017a99eb496bdc56a62afd038594a67febfab0aea9d76fe",
        manifest_sha256="c728ca1893a4bd6e5bf65082a176510794da80de74c1872664a687112f873f1e",
        loaded_parameters=753_446_208,
        backbone="qwen3_5_text",
        min_device_memory_gib=4,
    ),
    BuiltinModel(
        repo_id=f"{ORG}/Decision-2.0-Sol-2B",
        revision="4b75b52114583b4519001492e8dfb0926c89cfe1",
        family="decision2",
        model_sha256="e20df76c14edb0d28ef17593f49a1870fdea9438ff4805594402e39766f3546c",
        manifest_sha256="420ce3676ab79ed7097db9b44ec5aeef5804a817d644f3e9cdface499cb91393",
        loaded_parameters=1_883_930_944,
        backbone="qwen3_5_text",
        min_device_memory_gib=8,
    ),
    BuiltinModel(
        repo_id=f"{ORG}/Decision-2.0-Nox-4B",
        revision="ce1bdc9d91333aae2bf496ec48c66e1a913eb0a0",
        family="decision2",
        model_sha256="31d4ee9213686eeed6283d08ed7bce52ebef94ec47b7f97f7e1209f916fe51ea",
        manifest_sha256="0818bbec604bef5a631fa041d2188ff01ccceea6cb3f4ed131025b7717cce063",
        loaded_parameters=4_208_383_488,
        backbone="qwen3_5_text",
        min_device_memory_gib=16,
    ),
    BuiltinModel(
        repo_id=f"{ORG}/Decision-2.0-Lux-9B",
        revision="214ffa4322bc1bce3215c1bd5de6168402c76969",
        family="decision2",
        model_sha256="0ece5faa210173f443f353474b419fd913a253b06645da3e8370c2db0e1c3339",
        manifest_sha256="6e8e7f7dbe98cc100ed2d478eec9aba3ae1bc1711ac2ffe323f60512684d913b",
        loaded_parameters=7_940_895_744,
        backbone="qwen3_5_text",
        min_device_memory_gib=24,
    ),
    BuiltinModel(
        repo_id=f"{ORG}/Decision-2.0-Vega-27B",
        revision="9b067a95560284dac8c98ef4130fd5a2c5a92ff9",
        family="decision2",
        model_sha256="584697310502cd20e91728e69f724c028b47b95daf3790256dea3e434c48be82",
        manifest_sha256="0e90656f0e5b2a11a461d4ceb8f3a996c07813b5b643b0548adbc90d2f435414",
        loaded_parameters=29_365_153_792,
        backbone="qwen3_5_text",
        min_device_memory_gib=72,
        base=("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
    ),
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
