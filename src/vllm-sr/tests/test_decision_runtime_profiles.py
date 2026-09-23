"""Catalog/profile binding tests for the owned Decision runtime."""

from __future__ import annotations

import json
from dataclasses import replace
from importlib import resources

import pytest
from decision_runtime import catalog_adapter
from decision_runtime.catalog_adapter import (
    RuntimeModelResolutionError,
    resolve_decision_runtime_model,
)
from decision_runtime.runtime_profile import (
    DEFAULT_PHYSICAL_BATCH_SIZE,
    RuntimeProfileError,
    UnsupportedRuntimeBackendError,
    load_runtime_profile,
    parse_runtime_profile,
)

MODELS = {
    "llm-semantic-router/Decision-1.0-Kai-0.6B": (
        "7185f514f54b8f93c55998b1e8f9c5cc67f0d029",
        "vela",
        1024,
        None,
    ),
    "llm-semantic-router/Decision-1.0-Lex-0.6B": (
        "ee8e74d912fca8328a353c11d174b44da3f91781",
        "vela",
        1024,
        None,
    ),
    "llm-semantic-router/Decision-1.0-Eos-0.8B": (
        "3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd",
        "qwen3.5",
        16384,
        1.0389139156246665,
    ),
    "llm-semantic-router/Decision-1.0-Sol-2B": (
        "0665a41108e8f0b33a9515c98311c45947b99399",
        "qwen3.5",
        16384,
        1.3003552029656025,
    ),
    "llm-semantic-router/Decision-1.0-Nox-4B": (
        "0bb833504965c0eabdb9630b7bbd385cb2fe5cd4",
        "qwen3.5",
        16384,
        1.3231350559653137,
    ),
    "llm-semantic-router/Decision-1.0-Lux-9B": (
        "bd45a30aee8c84032791c245c70f86dee5389cc8",
        "qwen3.5",
        16384,
        2.0054410339959294,
    ),
}
PROMPT_POLICIES = {
    "llm-semantic-router/Decision-1.0-Kai-0.6B": "render_key",
    "llm-semantic-router/Decision-1.0-Lex-0.6B": "render_key",
    "llm-semantic-router/Decision-1.0-Eos-0.8B": "preserve_json_null",
    "llm-semantic-router/Decision-1.0-Sol-2B": "preserve_json_null",
    "llm-semantic-router/Decision-1.0-Nox-4B": "render_key",
    "llm-semantic-router/Decision-1.0-Lux-9B": "preserve_json_null",
}


@pytest.mark.parametrize("model_id", MODELS)
def test_catalog_exactly_selects_revision_profile(model_id: str) -> None:
    revision, family, max_tokens, temperature = MODELS[model_id]

    resolved = resolve_decision_runtime_model(model_id, backend="rocm", target="gfx942")

    assert resolved.catalog.model_id == model_id
    assert resolved.catalog.revision == revision
    assert resolved.repository_id == model_id
    assert resolved.profile.revision == revision
    assert resolved.profile.family == family
    assert resolved.profile.max_input_tokens == max_tokens
    assert resolved.profile.physical_batch_size == DEFAULT_PHYSICAL_BATCH_SIZE
    assert resolved.profile.temperature == temperature
    assert (
        resolved.profile.prompt_policy.choice_null_description
        == PROMPT_POLICIES[model_id]
    )


def test_profile_package_has_exact_catalog_revision_set() -> None:
    profile_files = {
        item.name.removesuffix(".json")
        for item in resources.files("decision_runtime.profiles").iterdir()
        if item.name.endswith(".json")
    }

    assert profile_files == {revision for revision, *_ in MODELS.values()}


@pytest.mark.parametrize(
    "model_id",
    (
        "llm-semantic-router/decision-1.0-kai-0.6b",
        "Decision-1.0-Kai-0.6B",
        "llm-semantic-router/Decision-1.0-KAI-0.6B",
        "llm-semantic-router/Decision-1.0-Missing",
        " llm-semantic-router/Decision-1.0-Kai-0.6B",
    ),
)
def test_catalog_adapter_rejects_aliases_and_unknown_models(model_id: str) -> None:
    with pytest.raises(RuntimeModelResolutionError):
        resolve_decision_runtime_model(model_id)


def test_catalog_adapter_fails_closed_on_profile_revision_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = resolve_decision_runtime_model(
        "llm-semantic-router/Decision-1.0-Kai-0.6B"
    )
    monkeypatch.setattr(
        catalog_adapter,
        "resolve_catalog_provider_model",
        lambda *args, **kwargs: replace(resolved.catalog, revision="a" * 40),
    )

    with pytest.raises(RuntimeModelResolutionError, match="no packaged"):
        resolve_decision_runtime_model(resolved.catalog.model_id)


def test_catalog_adapter_fails_closed_on_family_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = resolve_decision_runtime_model(
        "llm-semantic-router/Decision-1.0-Kai-0.6B"
    )
    monkeypatch.setattr(
        catalog_adapter,
        "load_runtime_profile",
        lambda revision: replace(resolved.profile, family="qwen3.5"),
    )

    with pytest.raises(RuntimeModelResolutionError, match="revision and family"):
        resolve_decision_runtime_model(resolved.catalog.model_id)


def test_unqualified_backends_and_targets_are_rejected() -> None:
    profile = load_runtime_profile(
        MODELS["llm-semantic-router/Decision-1.0-Kai-0.6B"][0]
    )

    profile.require_backend("rocm", target="gfx942")
    with pytest.raises(UnsupportedRuntimeBackendError):
        profile.require_backend("cuda")
    with pytest.raises(UnsupportedRuntimeBackendError):
        profile.require_backend("cpu")
    with pytest.raises(UnsupportedRuntimeBackendError):
        profile.require_backend("mlx")
    with pytest.raises(UnsupportedRuntimeBackendError):
        profile.require_backend("rocm", target="gfx1100")
    with pytest.raises(
        RuntimeModelResolutionError, match="requires an explicit backend"
    ):
        resolve_decision_runtime_model(
            "llm-semantic-router/Decision-1.0-Kai-0.6B", target="gfx942"
        )


def test_profile_parser_rejects_traversal_and_revision_fields() -> None:
    revision = MODELS["llm-semantic-router/Decision-1.0-Kai-0.6B"][0]
    packaged = resources.files("decision_runtime.profiles").joinpath(f"{revision}.json")
    document = json.loads(packaged.read_bytes())
    document["artifact"]["files"][0] = "../weights.safetensors"

    with pytest.raises(RuntimeProfileError, match="unsafe path"):
        parse_runtime_profile(json.dumps(document).encode(), revision=revision)

    document = json.loads(packaged.read_bytes())
    document["revision"] = revision
    with pytest.raises(RuntimeProfileError, match="fields do not match"):
        parse_runtime_profile(json.dumps(document).encode(), revision=revision)


def test_physical_batch_is_a_tunable_positive_profile_value() -> None:
    revision = MODELS["llm-semantic-router/Decision-1.0-Kai-0.6B"][0]
    packaged = resources.files("decision_runtime.profiles").joinpath(f"{revision}.json")
    document = json.loads(packaged.read_bytes())
    document["physical_batch_size"] = 4

    profile = parse_runtime_profile(json.dumps(document).encode(), revision=revision)

    assert profile.physical_batch_size == 4


@pytest.mark.parametrize("model_id", MODELS)
def test_cpu_is_explicit_and_unqualified_until_hardware_evidence(model_id: str) -> None:
    revision = MODELS[model_id][0]
    profile = load_runtime_profile(revision)
    cpu = profile.backends["cpu"]
    assert not cpu.qualified
    assert cpu.targets == ()
    if model_id.endswith(("Kai-0.6B", "Lex-0.6B", "Eos-0.8B")):
        assert cpu.backbone_dtype == "float32"
    else:
        assert cpu.backbone_dtype is None


def test_profile_rejects_qualified_backend_without_dtype() -> None:
    revision = MODELS["llm-semantic-router/Decision-1.0-Kai-0.6B"][0]
    packaged = resources.files("decision_runtime.profiles").joinpath(f"{revision}.json")
    document = json.loads(packaged.read_bytes())
    document["backends"]["cpu"] = {
        "qualified": True,
        "targets": ["x86_64"],
        "backbone_dtype": None,
    }

    with pytest.raises(RuntimeProfileError, match="dtype is required"):
        parse_runtime_profile(json.dumps(document).encode(), revision=revision)


@pytest.mark.parametrize("revision", ("main", "A" * 40, "a" * 39, "a" * 41))
def test_profile_loader_requires_full_lowercase_revision(revision: str) -> None:
    with pytest.raises(RuntimeProfileError, match="full lowercase Git SHA"):
        load_runtime_profile(revision)
