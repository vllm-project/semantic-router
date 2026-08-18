"""Focused contracts for the packaged built-in virtual-model catalog."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from cli import model_catalog
from cli.model_bundle import MODEL_BUNDLE_FILES, model_bundle_digest
from cli.model_catalog import (
    CatalogComponentVersions,
    ModelCatalogError,
    available_catalog_versions,
    find_catalog_model,
    fork_catalog_models,
    load_model_catalog,
    materialize_catalog_models,
)

DEFAULT_MODEL = "vllm-sr/mom-v1-blend"
LITE_MODEL = "vllm-sr/mom-v1-lite"
FLASH_MODEL = "vllm-sr/mom-v1-flash"
CATALOG_MODELS = {
    DEFAULT_MODEL,
    LITE_MODEL,
    FLASH_MODEL,
    "vllm-sr/mom-v1-ultra",
    "vllm-sr/mom-v1-vault",
}


def _system_prompts(value: Any) -> list[str]:
    prompts: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "system_prompt" and isinstance(item, str):
                prompts.append(item)
            else:
                prompts.extend(_system_prompts(item))
    elif isinstance(value, list):
        for item in value:
            prompts.extend(_system_prompts(item))
    return prompts


def _target_at(document: dict[str, Any], path: tuple[str | int, ...]) -> Any:
    target: Any = document
    for part in path:
        target = target[part]
    return target


def _stage_catalog_asset(
    source_version,
    fixture_version: Path,
    asset: dict[str, Any],
    document: dict[str, Any],
    mutate_asset: Callable[[str, dict[str, Any]], None] | None,
) -> None:
    bundle = asset.get("bundle")
    if not isinstance(bundle, str) or Path(bundle).name != bundle:
        return
    fixture_bundle = fixture_version / bundle
    fixture_bundle.mkdir()
    for name in MODEL_BUNDLE_FILES:
        fixture_bundle.joinpath(name).write_bytes(
            source_version.joinpath(bundle, name).read_bytes()
        )
    if mutate_asset is not None:
        _mutate_staged_catalog_asset(
            fixture_bundle, bundle, asset, document, mutate_asset
        )


def _mutate_staged_catalog_asset(
    fixture_bundle: Path,
    bundle: str,
    asset: dict[str, Any],
    document: dict[str, Any],
    mutate_asset: Callable[[str, dict[str, Any]], None],
) -> None:
    config_path = fixture_bundle / "config.yaml"
    asset_document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mutate_asset(bundle, asset_document)
    config_path.write_text(
        yaml.safe_dump(asset_document, sort_keys=False), encoding="utf-8"
    )
    digest = model_bundle_digest(fixture_bundle)
    asset["sha256"] = digest
    for model in document.get("models", []):
        if model.get("asset") == asset.get("id"):
            model["verification"]["asset_sha256"] = digest


def _load_mutated_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, Any]], None],
    mutate_asset: Callable[[str, dict[str, Any]], None] | None = None,
) -> None:
    packaged_root = model_catalog.resources.files("cli.model_assets")
    source_version = packaged_root.joinpath("latest")
    document = yaml.safe_load(
        source_version.joinpath("catalog.yaml").read_text(encoding="utf-8")
    )
    mutate(document)

    fixture_root = tmp_path / "model-assets"
    fixture_version = fixture_root / "latest"
    fixture_version.mkdir(parents=True)
    for asset in document.get("assets", []):
        _stage_catalog_asset(
            source_version, fixture_version, asset, document, mutate_asset
        )

    (fixture_version / "catalog.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setattr(model_catalog.resources, "files", lambda package: fixture_root)
    load_model_catalog("latest")


def test_packaged_latest_catalog_is_verified() -> None:
    assert available_catalog_versions() == ("latest",)

    catalog = load_model_catalog("latest")

    assert catalog.version == "latest"
    assert catalog.channel == "latest"
    assert catalog.default_model == DEFAULT_MODEL
    assert catalog.enabled_models == (DEFAULT_MODEL,)
    assert {model.id for model in catalog.models} == CATALOG_MODELS
    assert all(model.compatibility.compatible for model in catalog.models)
    assert all(model.verified for model in catalog.models)


def test_packaged_mom_prompts_use_model_ids_and_community_attribution() -> None:
    packaged_root = model_catalog.resources.files("cli.model_assets")
    document = yaml.safe_load(
        packaged_root.joinpath("latest", "mom-v1", "config.yaml").read_text(
            encoding="utf-8"
        )
    )
    expected_identities = {
        "balance": "mom-v1-blend",
        "cost": "mom-v1-lite",
        "speed": "mom-v1-flash",
        "accuracy": "mom-v1-ultra",
        "vault": "mom-v1-vault",
    }
    seen: set[str] = set()

    for recipe in document["recipes"]:
        recipe_name = recipe["name"]
        if recipe_name not in expected_identities:
            continue
        prompts = _system_prompts(recipe)
        identity = expected_identities[recipe_name]
        prefix = f"You are {identity}, built by vLLM-SR Community."
        assert prompts, f"{recipe_name} has no system prompts"
        assert all(prompt.startswith(prefix) for prompt in prompts)
        assert all("built by AMD" not in prompt for prompt in prompts)
        seen.add(recipe_name)

    assert seen == set(expected_identities)


@pytest.mark.parametrize(
    ("component_versions", "reason"),
    [
        (CatalogComponentVersions(cli="0.2.9", router="0.3.0"), "requires cli"),
        (CatalogComponentVersions(cli="0.3.0", router="0.2.9"), "requires router"),
    ],
)
def test_catalog_evaluates_cli_and_router_versions_independently(
    component_versions: CatalogComponentVersions, reason: str
) -> None:
    catalog = load_model_catalog("latest", component_versions=component_versions)

    assert all(not model.compatibility.compatible for model in catalog.models)
    assert all(reason in model.compatibility.reason for model in catalog.models)
    assert all(model.verified for model in catalog.models)


@pytest.mark.parametrize(
    ("component_versions", "compatible", "reason"),
    [
        (
            CatalogComponentVersions(cli="0.3.0-rc.1", router="0.3.0"),
            False,
            "requires cli >= 0.3.0",
        ),
        (
            CatalogComponentVersions(cli="0.5.0-rc.1", router="0.3.0"),
            True,
            "compatible",
        ),
    ],
)
def test_catalog_uses_semver_prerelease_precedence(
    component_versions: CatalogComponentVersions,
    compatible: bool,
    reason: str,
) -> None:
    catalog = load_model_catalog("latest", component_versions=component_versions)

    assert all(model.compatibility.compatible is compatible for model in catalog.models)
    assert all(model.compatibility.reason == reason for model in catalog.models)
    assert all(model.verified for model in catalog.models)


def test_catalog_accepts_a_prerelease_to_release_version_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mutate(document: dict[str, Any]) -> None:
        document["compatibility"]["cli"] = {
            "min": "0.4.0-rc.1",
            "max_exclusive": "0.4.0",
        }

    _load_mutated_catalog(tmp_path, monkeypatch, mutate)


def test_available_catalog_versions_keeps_latest_first_and_sorts_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_root = tmp_path / "model-assets"
    for version in ("v1.2", "latest", "v1.10", "invalid"):
        version_root = fixture_root / version
        version_root.mkdir(parents=True)
        (version_root / "catalog.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(model_catalog.resources, "files", lambda package: fixture_root)

    assert available_catalog_versions() == ("latest", "v1.10", "v1.2")


def test_find_catalog_model_returns_public_binding_metadata() -> None:
    catalog, model = find_catalog_model(FLASH_MODEL)

    assert catalog.channel == "latest"
    assert model.id == FLASH_MODEL
    assert model.entrypoint == FLASH_MODEL
    assert model.recipe == "speed"
    assert model.catalog_version == "latest"
    assert model.verification["authority"]


def test_packaged_catalog_roles_cover_each_recipe_provider_reference() -> None:
    for installed_version in available_catalog_versions():
        catalog = load_model_catalog(installed_version)

        assert {model.id for model in catalog.models} == CATALOG_MODELS


def test_fork_without_selection_materializes_catalog_defaults(tmp_path: Path) -> None:
    destination = tmp_path / "default.yaml"

    result = fork_catalog_models((), destination)
    document = yaml.safe_load(destination.read_text(encoding="utf-8"))

    assert result == {
        "path": str(destination.resolve()),
        "catalog_version": "latest",
        "enabled": [DEFAULT_MODEL],
        "default": DEFAULT_MODEL,
        "verification": "verified",
    }
    assert document["entrypoints"] == [
        {"model_names": [DEFAULT_MODEL], "recipe": "balance"}
    ]
    assert [recipe["name"] for recipe in document["recipes"]] == ["balance"]
    # A virtual entrypoint is never a provider fallback model.
    assert document["providers"]["defaults"]["default_model"] == "local/qwen3.5-9b"


def test_fork_materializes_multiple_models_and_explicit_default(tmp_path: Path) -> None:
    destination = tmp_path / "multi.yaml"

    result = fork_catalog_models(
        (LITE_MODEL, FLASH_MODEL),
        destination,
        default_model=FLASH_MODEL,
    )
    document = yaml.safe_load(destination.read_text(encoding="utf-8"))

    # The default is persisted without adding non-runtime metadata: it is the
    # first selected entrypoint, and the rest retain request order.
    assert result["enabled"] == [FLASH_MODEL, LITE_MODEL]
    assert result["default"] == FLASH_MODEL
    assert result["verification"] == "verified"
    assert document["entrypoints"] == [
        {"model_names": [FLASH_MODEL], "recipe": "speed"},
        {"model_names": [LITE_MODEL], "recipe": "cost"},
    ]
    assert [recipe["name"] for recipe in document["recipes"]] == ["speed", "cost"]
    assert (
        materialize_catalog_models(
            (LITE_MODEL, FLASH_MODEL),
            default_model=FLASH_MODEL,
        ).document
        == document
    )


def test_fork_refuses_to_overwrite_existing_config(tmp_path: Path) -> None:
    destination = tmp_path / "existing.yaml"
    destination.write_text("sentinel\n", encoding="utf-8")

    with pytest.raises(ModelCatalogError, match="refusing to overwrite"):
        fork_catalog_models((DEFAULT_MODEL,), destination)

    assert destination.read_text(encoding="utf-8") == "sentinel\n"


def test_missing_catalog_version_fails_closed() -> None:
    with pytest.raises(ModelCatalogError, match="is not installed"):
        load_model_catalog("v9.9")


@pytest.mark.parametrize(
    "path",
    (
        (),
        ("defaults",),
        ("compatibility",),
        ("compatibility", "cli"),
        ("assets", 0),
        ("models", 0),
        ("models", 0, "roles", 0),
        ("models", 0, "verification"),
    ),
)
def test_catalog_rejects_unknown_fields_at_every_manifest_layer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str | int, ...],
) -> None:
    def mutate(document: dict[str, Any]) -> None:
        _target_at(document, path)["unexpected_contract"] = True

    with pytest.raises(ModelCatalogError, match="unknown fields: unexpected_contract"):
        _load_mutated_catalog(tmp_path, monkeypatch, mutate)


def test_catalog_rejects_unknown_fields_in_model_compatibility_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mutate(document: dict[str, Any]) -> None:
        document["models"][0]["compatibility"] = {
            "cli": {"min": "0.3.0", "unsupported_bound": "0.4.0"}
        }

    with pytest.raises(ModelCatalogError, match="unknown fields: unsupported_bound"):
        _load_mutated_catalog(tmp_path, monkeypatch, mutate)


@pytest.mark.parametrize("field", ("api_key", "auth_token", "client_secret"))
def test_catalog_rejects_secret_like_manifest_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    def mutate(document: dict[str, Any]) -> None:
        document["models"][0][field] = "must-not-enter-a-package"

    with pytest.raises(ModelCatalogError, match=rf"secret-like field: .*\.{field}"):
        _load_mutated_catalog(tmp_path, monkeypatch, mutate)


@pytest.mark.parametrize(
    "literal",
    (
        "Bearer abcdefghijklmnopqrstuvwxyz",
        "https://catalog-user:catalog-password@example.invalid/models",
        "api_key=" + "sk" + "-abcdefghijklmnopqrstuvwxyz012345",
    ),
)
def test_catalog_rejects_credential_like_literals_without_echoing_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    literal: str,
) -> None:
    def mutate(document: dict[str, Any]) -> None:
        document["models"][0]["description"] = literal

    with pytest.raises(
        ModelCatalogError, match=r"credential-like literal at .*\.description"
    ) as raised:
        _load_mutated_catalog(tmp_path, monkeypatch, mutate)
    assert literal not in str(raised.value)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("catalog_version",), "0.4", "vMAJOR.MINOR"),
        (("release",), "v0.5", "latest catalog must use"),
        (("channel",), "preview", "unsupported value"),
        (("compatibility", "cli", "min"), "v0.3", "semantic version"),
        (("assets", 0, "id"), "../mom", "lowercase slug"),
        (("assets", 0, "bundle"), "../mom", "lowercase slug"),
        (("models", 0, "id"), "mom-v1", "public model ID"),
        (("models", 0, "kind"), "concrete", "unsupported value"),
        (("models", 0, "family"), "MoM", "lowercase slug"),
        (("models", 0, "policy_version"), "1", "semantic version"),
        (("models", 0, "protocols"), ["openai"], "unsupported values"),
        (
            ("models", 0, "roles", 0, "minimum_candidates"),
            99,
            "exceeds its recommended pool",
        ),
    ),
)
def test_catalog_rejects_invalid_identity_version_enum_and_cardinality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str | int, ...],
    value: Any,
    message: str,
) -> None:
    def mutate(document: dict[str, Any]) -> None:
        parent = _target_at(document, path[:-1])
        parent[path[-1]] = value

    with pytest.raises(ModelCatalogError, match=message):
        _load_mutated_catalog(tmp_path, monkeypatch, mutate)


def test_catalog_rejects_role_pool_missing_nested_algorithm_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mutate_asset(bundle: str, document: dict[str, Any]) -> None:
        if bundle != "mom-v1":
            return
        accuracy = next(
            recipe for recipe in document["recipes"] if recipe["name"] == "accuracy"
        )
        fusion = next(
            decision
            for decision in accuracy["routing"]["decisions"]
            if decision["name"] == "accuracy_expert_fusion"
        )
        fusion["algorithm"]["fusion"]["model"] = "local/qwen3.5-9b"

    with pytest.raises(
        ModelCatalogError,
        match=rf"{LITE_MODEL.removesuffix('-lite')}-ultra.*local/qwen3\.5-9b",
    ):
        _load_mutated_catalog(tmp_path, monkeypatch, lambda _: None, mutate_asset)


def test_catalog_allows_additional_recommended_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mutate(document: dict[str, Any]) -> None:
        lite = next(model for model in document["models"] if model["id"] == LITE_MODEL)
        lite["roles"][0]["recommended_pool"].append("operator/alternative-model")

    _load_mutated_catalog(tmp_path, monkeypatch, mutate)


@pytest.mark.parametrize("version", ("../latest", "v1.2.3", "preview"))
def test_catalog_rejects_invalid_resource_version_before_path_resolution(
    version: str,
) -> None:
    with pytest.raises(ModelCatalogError, match="catalog version is invalid"):
        load_model_catalog(version)
