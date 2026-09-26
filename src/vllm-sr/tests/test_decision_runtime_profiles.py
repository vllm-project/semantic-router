"""Catalog/profile binding tests for the owned Decision runtime."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from importlib import resources
from pathlib import Path

import pytest
from decision_runtime import catalog_adapter
from decision_runtime.catalog_adapter import (
    RuntimeModelResolutionError,
    resolve_decision_runtime_model,
)
from decision_runtime.family_registry import (
    catalog_family_registration,
    family_adapter,
    family_registration,
    registered_families,
)
from decision_runtime.runtime_profile import (
    DEFAULT_PHYSICAL_BATCH_SIZE,
    RuntimeProfileError,
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


def _packaged_profile(model_id: str):
    family = MODELS[model_id][1]
    directory = "vela" if family == "vela" else "qwen35"
    return resources.files("decision_runtime.profiles").joinpath(
        directory, f"{model_id.rsplit('/', 1)[-1]}.json"
    )


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
    rocm_policy = resolved.profile.execution.get("rocm")
    assert (rocm_policy.gated_delta if rocm_policy else "accelerated") == (
        "native_torch" if "Eos-0.8B" in model_id else "accelerated"
    )
    assert (rocm_policy.max_physical_batch_size if rocm_policy else None) == (
        8 if "Eos-0.8B" in model_id else None
    )
    expected_turn_rows = {
        "Decision-1.0-Eos-0.8B": 16,
        "Decision-1.0-Nox-4B": 32,
        "Decision-1.0-Lux-9B": 32,
    }.get(model_id.rsplit("/", 1)[-1], 1)
    assert resolved.profile.rows_per_job_turn("rocm", 8) == expected_turn_rows
    assert resolved.profile.rows_per_job_turn("cpu", 8) == 1
    assert resolved.profile.use_short_b8_graph("rocm", 8) == model_id.endswith(
        ("Sol-2B", "Nox-4B")
    )
    assert not resolved.profile.use_short_b8_graph("rocm", 16)
    assert not resolved.profile.use_short_b8_graph("cpu", 8)
    assert (rocm_policy.graph_prewarm_padded_tokens if rocm_policy else ()) == (
        (128,) if model_id.endswith("Nox-4B") else ()
    )


def test_profile_package_has_exact_catalog_model_set() -> None:
    profile_files = {
        item.name.removesuffix(".json")
        for family in ("vela", "qwen35")
        for item in resources.files("decision_runtime.profiles")
        .joinpath(family)
        .iterdir()
        if item.name.endswith(".json")
    }

    assert profile_files == {model_id.rsplit("/", 1)[-1] for model_id in MODELS}


def test_family_registry_owns_catalog_binding_profile_layout_and_adapter() -> None:
    expected = {
        "vela": ("decision-encoder", "vela"),
        "qwen3.5": ("decision-qwen3.5", "qwen35"),
    }
    assert {item.family for item in registered_families()} == set(expected)
    for family, (catalog_family, directory) in expected.items():
        registration = family_registration(family)
        assert registration is not None
        assert registration.catalog_family == catalog_family
        assert registration.profile_directory == directory
        assert registration.image_environment == directory
        assert registration.container_python == (
            f"/opt/vllm-sr/venvs/{directory}/bin/python"
        )
        assert registration.rocm_target == "gfx942"
        assert registration.cpu_below_billions == 1.0
        assert catalog_family_registration(catalog_family) is registration
        assert family_adapter(family).family == family
        assert family_adapter(family).manifest_paths == registration.manifest_paths
    with pytest.raises(ValueError, match="no owned adapter"):
        family_adapter("unregistered")


def test_profile_and_family_registration_do_not_import_torch() -> None:
    project_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from decision_runtime.runtime_profile import load_runtime_profile; "
            "import sys; "
            "load_runtime_profile('Decision-1.0-Eos-0.8B', revision='3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd'); "
            "assert 'torch' not in sys.modules",
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": str(project_root)},
        capture_output=True,
        text=True,
    )


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


@pytest.mark.parametrize("model_id", MODELS)
def test_catalog_new_revision_reuses_model_template(
    monkeypatch: pytest.MonkeyPatch, model_id: str
) -> None:
    resolved = resolve_decision_runtime_model(model_id)
    monkeypatch.setattr(
        catalog_adapter,
        "resolve_catalog_provider_model",
        lambda *args, **kwargs: replace(resolved.catalog, revision="a" * 40),
    )

    updated = resolve_decision_runtime_model(resolved.catalog.model_id)
    assert updated.catalog.revision == "a" * 40
    assert updated.profile.revision == "a" * 40
    assert updated.template_id == resolved.template_id
    assert updated.template_id == model_id.rsplit("/", 1)[-1]
    assert updated.profile.prompt_policy == resolved.profile.prompt_policy
    assert updated.profile.execution == resolved.profile.execution


@pytest.mark.parametrize(
    "execution",
    (
        None,
        {},
        {"cuda": {"backbone_graph": "short_b8"}},
        {"rocm": {"backbone_graph": "unknown"}},
        {"rocm": {"backbone_graph": "short_b8", "qualified": True}},
        {"rocm": {"job_turn_batches": 0}},
        {"rocm": {"job_turn_batches": 5}},
        {"rocm": {"job_turn_batches": True}},
        {"rocm": {"graph_prewarm_padded_tokens": [128]}},
        {
            "rocm": {
                "backbone_graph": "short_b8",
                "graph_prewarm_padded_tokens": [128, 128],
            }
        },
        {
            "rocm": {
                "backbone_graph": "short_b8",
                "graph_prewarm_padded_tokens": [32, 64, 96],
            }
        },
        {
            "rocm": {
                "backbone_graph": "short_b8",
                "graph_prewarm_padded_tokens": [True],
            }
        },
        {
            "rocm": {
                "backbone_graph": "short_b8",
                "graph_prewarm_padded_tokens": [129],
            }
        },
    ),
)
def test_execution_policy_rejects_unknown_modes_and_backend_claims(
    execution: object,
) -> None:
    revision = MODELS["llm-semantic-router/Decision-1.0-Sol-2B"][0]
    document = json.loads(
        _packaged_profile("llm-semantic-router/Decision-1.0-Sol-2B").read_bytes()
    )
    document["execution"] = execution

    with pytest.raises(RuntimeProfileError, match="execution"):
        parse_runtime_profile(json.dumps(document).encode(), revision=revision)


def test_explicit_revision_requires_an_immutable_commit() -> None:
    model = "llm-semantic-router/Decision-1.0-Kai-0.6B"
    updated = resolve_decision_runtime_model(model, revision="a" * 40)
    assert updated.catalog.revision == "a" * 40
    with pytest.raises(RuntimeModelResolutionError, match="full lowercase Git SHA"):
        resolve_decision_runtime_model(model, revision="main")


def test_catalog_adapter_fails_closed_on_family_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = resolve_decision_runtime_model(
        "llm-semantic-router/Decision-1.0-Kai-0.6B"
    )
    monkeypatch.setattr(
        catalog_adapter,
        "load_runtime_profile",
        lambda profile_id, *, revision: replace(resolved.profile, family="qwen3.5"),
    )

    with pytest.raises(RuntimeModelResolutionError, match="model family"):
        resolve_decision_runtime_model(resolved.catalog.model_id)


def test_unqualified_backends_and_targets_are_rejected() -> None:
    model_id = "llm-semantic-router/Decision-1.0-Kai-0.6B"
    resolve_decision_runtime_model(model_id, backend="rocm", target="gfx942")
    with pytest.raises(RuntimeModelResolutionError, match="no installed"):
        resolve_decision_runtime_model(model_id, backend="cuda")
    with pytest.raises(RuntimeModelResolutionError, match="no installed"):
        resolve_decision_runtime_model(model_id, backend="mlx")
    with pytest.raises(RuntimeModelResolutionError, match="unsupported"):
        resolve_decision_runtime_model(model_id, backend="rocm", target="gfx1100")
    with pytest.raises(
        RuntimeModelResolutionError, match="requires an explicit backend"
    ):
        resolve_decision_runtime_model(
            "llm-semantic-router/Decision-1.0-Kai-0.6B", target="gfx942"
        )


def test_runtime_owned_backend_capabilities_cover_six_rocm_and_sub_1b_cpu() -> None:
    for model_id in MODELS:
        assert resolve_decision_runtime_model(
            model_id, backend="rocm", target="gfx942"
        ).profile.family in {"vela", "qwen3.5"}
        with pytest.raises(RuntimeModelResolutionError, match="no installed"):
            resolve_decision_runtime_model(model_id, backend="cuda")
        with pytest.raises(RuntimeModelResolutionError, match="target"):
            resolve_decision_runtime_model(model_id, backend="rocm", target="gfx1100")
        if model_id.endswith(("Kai-0.6B", "Lex-0.6B", "Eos-0.8B")):
            resolve_decision_runtime_model(model_id, backend="cpu")
        else:
            with pytest.raises(RuntimeModelResolutionError, match="no installed"):
                resolve_decision_runtime_model(model_id, backend="cpu")


def test_profile_parser_rejects_traversal_and_revision_fields() -> None:
    revision = MODELS["llm-semantic-router/Decision-1.0-Kai-0.6B"][0]
    packaged = _packaged_profile("llm-semantic-router/Decision-1.0-Kai-0.6B")
    document = json.loads(packaged.read_bytes())
    document["artifact"]["manifest_path"] = "../weights.safetensors"

    with pytest.raises(RuntimeProfileError, match="unsafe path"):
        parse_runtime_profile(json.dumps(document).encode(), revision=revision)

    document = json.loads(packaged.read_bytes())
    document["artifact"]["files"] = ["backbone/model.safetensors"]
    with pytest.raises(RuntimeProfileError, match="fields do not match"):
        parse_runtime_profile(json.dumps(document).encode(), revision=revision)

    document = json.loads(packaged.read_bytes())
    document["revision"] = revision
    with pytest.raises(RuntimeProfileError, match="fields do not match"):
        parse_runtime_profile(json.dumps(document).encode(), revision=revision)


@pytest.mark.parametrize(
    ("field_path", "invalid", "message"),
    (
        (("family",), [], "profile.family"),
        (("family",), {}, "profile.family"),
        (("dtype", "backbone"), [], "dtype.backbone"),
        (("dtype", "backbone"), {}, "dtype.backbone"),
        (("dtype", "head"), [], "dtype.head"),
        (("dtype", "head"), {}, "dtype.head"),
        (
            ("prompt_policy", "choice_null_description"),
            [],
            "prompt_policy.choice_null_description",
        ),
        (
            ("prompt_policy", "choice_null_description"),
            {},
            "prompt_policy.choice_null_description",
        ),
    ),
)
def test_profile_parser_rejects_nested_container_values(
    field_path: tuple[str, ...], invalid: object, message: str
) -> None:
    revision = MODELS["llm-semantic-router/Decision-1.0-Kai-0.6B"][0]
    document = json.loads(
        _packaged_profile("llm-semantic-router/Decision-1.0-Kai-0.6B").read_bytes()
    )
    target = document
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = invalid

    with pytest.raises(RuntimeProfileError, match=message):
        parse_runtime_profile(json.dumps(document).encode(), revision=revision)


def test_profile_parser_rejects_oversized_calibration_without_overflow() -> None:
    revision = MODELS["llm-semantic-router/Decision-1.0-Eos-0.8B"][0]
    document = json.loads(
        _packaged_profile("llm-semantic-router/Decision-1.0-Eos-0.8B").read_bytes()
    )
    document["calibration"]["temperature"] = 10**400

    with pytest.raises(RuntimeProfileError, match=r"calibration\.temperature"):
        parse_runtime_profile(json.dumps(document).encode(), revision=revision)


def test_profile_parser_requires_integer_schema_version() -> None:
    revision = MODELS["llm-semantic-router/Decision-1.0-Kai-0.6B"][0]
    document = json.loads(
        _packaged_profile("llm-semantic-router/Decision-1.0-Kai-0.6B").read_bytes()
    )
    document["schema_version"] = 4.0

    with pytest.raises(RuntimeProfileError, match="profile schema"):
        parse_runtime_profile(json.dumps(document).encode(), revision=revision)


def test_physical_batch_is_a_tunable_positive_profile_value() -> None:
    revision = MODELS["llm-semantic-router/Decision-1.0-Kai-0.6B"][0]
    packaged = _packaged_profile("llm-semantic-router/Decision-1.0-Kai-0.6B")
    document = json.loads(packaged.read_bytes())
    document["physical_batch_size"] = 4

    profile = parse_runtime_profile(json.dumps(document).encode(), revision=revision)

    assert profile.physical_batch_size == 4


def test_model_profile_rejects_backend_qualification_declarations() -> None:
    revision = MODELS["llm-semantic-router/Decision-1.0-Kai-0.6B"][0]
    packaged = _packaged_profile("llm-semantic-router/Decision-1.0-Kai-0.6B")
    document = json.loads(packaged.read_bytes())
    document["backends"] = {"cpu": {"qualified": True}}

    with pytest.raises(RuntimeProfileError, match="fields do not match"):
        parse_runtime_profile(json.dumps(document).encode(), revision=revision)


@pytest.mark.parametrize(
    ("policy", "message"),
    (
        (
            {"gated_delta": "native_torch", "max_physical_batch_size": 0},
            "positive integer",
        ),
        (
            {"gated_delta": "native_torch", "max_physical_batch_size": True},
            "positive integer",
        ),
        (
            {"gated_delta": "accelerated", "max_physical_batch_size": 8},
            "requires native_torch",
        ),
        ({"gated_delta": [], "max_physical_batch_size": 8}, "gated_delta"),
        (
            {
                "gated_delta": "native_torch",
                "max_physical_batch_size": 8,
                "backbone_graph": "short_b8",
            },
            "cannot combine",
        ),
    ),
)
def test_qwen_execution_policy_rejects_invalid_envelopes(policy, message) -> None:
    revision = MODELS["llm-semantic-router/Decision-1.0-Eos-0.8B"][0]
    document = json.loads(
        _packaged_profile("llm-semantic-router/Decision-1.0-Eos-0.8B").read_bytes()
    )
    document["execution"] = {"rocm": policy}
    with pytest.raises(RuntimeProfileError, match=message):
        parse_runtime_profile(json.dumps(document).encode(), revision=revision)


def test_vela_profile_cannot_select_qwen_execution_policy() -> None:
    revision = MODELS["llm-semantic-router/Decision-1.0-Kai-0.6B"][0]
    document = json.loads(
        _packaged_profile("llm-semantic-router/Decision-1.0-Kai-0.6B").read_bytes()
    )
    document["execution"] = {
        "rocm": {"gated_delta": "native_torch", "max_physical_batch_size": 8}
    }
    with pytest.raises(RuntimeProfileError, match="execution"):
        parse_runtime_profile(json.dumps(document).encode(), revision=revision)


@pytest.mark.parametrize("revision", ("main", "A" * 40, "a" * 39, "a" * 41))
def test_profile_loader_requires_full_lowercase_revision(revision: str) -> None:
    with pytest.raises(RuntimeProfileError, match="full lowercase Git SHA"):
        load_runtime_profile("Decision-1.0-Kai-0.6B", revision=revision)


@pytest.mark.parametrize("profile_id", ("../Kai", ".hidden", "Kai/other", ""))
def test_profile_loader_rejects_unsafe_model_names(profile_id: str) -> None:
    with pytest.raises(RuntimeProfileError, match="profile ID is invalid"):
        load_runtime_profile(profile_id, revision="a" * 40)
