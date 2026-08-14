"""Evolution contracts for multi-asset built-in model catalogs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml
from cli import model_catalog
from cli.model_bundle import MODEL_BUNDLE_FILES, model_bundle_digest
from cli.model_catalog import (
    CatalogCompatibility,
    CatalogModel,
    ModelCatalog,
    ModelCatalogError,
)


def _asset_document(model_id: str, recipe: str, provider: str) -> dict:
    return {
        "version": "v0.3",
        "listeners": [{"name": "http", "address": "0.0.0.0", "port": 8899}],
        "global": {"router": {"strategy": "priority"}},
        "providers": {
            "defaults": {"default_model": "local/fallback"},
            "models": [
                {"name": "local/fallback"},
                {"name": provider},
            ],
        },
        "routing": {
            "strategy": "priority",
            "modelCards": [
                {"name": "local/fallback"},
                {"name": provider},
            ],
        },
        "entrypoints": [{"model_names": [model_id], "recipe": recipe}],
        "recipes": [{"name": recipe, "routing": {"decisions": []}}],
    }


def _catalog_model(model_id: str, asset: str, recipe: str) -> CatalogModel:
    return CatalogModel(
        id=model_id,
        display_name=model_id,
        description=model_id,
        kind="virtual",
        family="chorus",
        generation=1 if model_id.endswith("v1") else 2,
        policy_version="1.0.0",
        asset=asset,
        entrypoint=model_id,
        recipe=recipe,
        protocols=("openai_chat",),
        traits=("chat",),
        roles=({"name": "default"},),
        verification={"authority": "maintainers", "asset_sha256": "sha256:" + "0" * 64},
        catalog_version="latest",
        channel="latest",
        compatibility=CatalogCompatibility(True, "compatible"),
        enabled_by_default=False,
        default=False,
    )


def test_materialization_merges_compatible_chorus_generations(monkeypatch) -> None:
    v1 = "vllm-sr/chorus-v1"
    v2 = "vllm-sr/chorus-v2"
    catalog = ModelCatalog(
        version="latest",
        channel="latest",
        default_model=v1,
        enabled_models=(v1,),
        assets={
            "chorus-v1": {"bundle": "chorus-v1", "sha256": "sha256:" + "0" * 64},
            "chorus-v2": {"bundle": "chorus-v2", "sha256": "sha256:" + "0" * 64},
        },
        models=(
            _catalog_model(v1, "chorus-v1", "balance"),
            _catalog_model(v2, "chorus-v2", "next"),
        ),
    )
    documents = {
        "chorus-v1": _asset_document(v1, "balance", "local/v1"),
        "chorus-v2": _asset_document(v2, "next", "local/v2"),
    }
    monkeypatch.setattr(
        model_catalog,
        "load_model_catalog",
        lambda _version, **_kwargs: catalog,
    )
    monkeypatch.setattr(
        model_catalog,
        "_load_asset_yaml",
        lambda _version, asset: documents[asset["bundle"]],
    )

    materialized = model_catalog.materialize_catalog_models((v1, v2), default_model=v2)

    assert materialized.enabled_models == (v2, v1)
    assert [
        entry["model_names"][0] for entry in materialized.document["entrypoints"]
    ] == [v2, v1]
    assert [recipe["name"] for recipe in materialized.document["recipes"]] == [
        "next",
        "balance",
    ]
    assert {
        model["name"] for model in materialized.document["providers"]["models"]
    } == {
        "local/fallback",
        "local/v1",
        "local/v2",
    }


def test_materialization_deduplicates_a_recipe_shared_by_virtual_models(
    monkeypatch,
) -> None:
    first = "vllm-sr/chorus-shared-a"
    second = "vllm-sr/chorus-shared-b"
    shared_recipe = "shared"
    asset_document = _asset_document(first, shared_recipe, "local/shared")
    asset_document["entrypoints"].append(
        {"model_names": [second], "recipe": shared_recipe}
    )
    catalog = ModelCatalog(
        version="latest",
        channel="latest",
        default_model=first,
        enabled_models=(first,),
        assets={
            "chorus-shared": {
                "bundle": "chorus-shared",
                "sha256": "sha256:" + "0" * 64,
            }
        },
        models=(
            _catalog_model(first, "chorus-shared", shared_recipe),
            _catalog_model(second, "chorus-shared", shared_recipe),
        ),
    )
    monkeypatch.setattr(
        model_catalog,
        "load_model_catalog",
        lambda _version, **_kwargs: catalog,
    )
    monkeypatch.setattr(
        model_catalog,
        "_load_asset_yaml",
        lambda _version, _asset: asset_document,
    )

    materialized = model_catalog.materialize_catalog_models((first, second))

    assert [recipe["name"] for recipe in materialized.document["recipes"]] == [
        shared_recipe
    ]


def test_multi_asset_materialization_rejects_global_conflicts() -> None:
    first = _asset_document("vllm-sr/chorus-v1", "balance", "local/v1")
    second = _asset_document("vllm-sr/chorus-v2", "next", "local/v2")
    second["listeners"][0]["port"] = 9999

    with pytest.raises(ModelCatalogError, match="runtime-global field: listeners"):
        model_catalog._merge_asset_documents([first, second])


def test_role_validation_scopes_future_generations_to_their_asset_and_recipe() -> None:
    v1 = _asset_document("vllm-sr/chorus-v1", "balance", "local/v1")
    v2 = _asset_document("vllm-sr/chorus-v2", "next", "local/v2")
    v1["recipes"][0]["routing"]["decisions"] = [{"modelRefs": [{"model": "local/v1"}]}]
    v2["recipes"][0]["routing"]["decisions"] = [{"modelRefs": [{"model": "local/v2"}]}]

    model_catalog._validate_catalog_model_role_pool(
        "vllm-sr/chorus-v1",
        "balance",
        ({"recommended_pool": ["local/v1", "operator/alternative"]},),
        v1,
    )
    model_catalog._validate_catalog_model_role_pool(
        "vllm-sr/chorus-v2",
        "next",
        ({"recommended_pool": ["local/v2"]},),
        v2,
    )

    with pytest.raises(ModelCatalogError, match=r"vllm-sr/chorus-v2.*local/v2"):
        model_catalog._validate_catalog_model_role_pool(
            "vllm-sr/chorus-v2",
            "next",
            ({"recommended_pool": ["local/v1"]},),
            v2,
        )


def test_sync_script_discovers_every_declared_catalog_asset(
    tmp_path: Path, monkeypatch
) -> None:
    script_path = (
        Path(__file__).resolve().parents[3] / "tools/release/sync_model_catalog.py"
    )
    spec = importlib.util.spec_from_file_location(
        "sync_model_catalog_test", script_path
    )
    assert spec is not None and spec.loader is not None
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    version = source / "latest"
    version.mkdir(parents=True)
    destination.mkdir()
    (destination / "__init__.py").write_text("", encoding="utf-8")
    digests: dict[str, str] = {}
    for bundle in ("chorus-v1", "chorus-v2"):
        bundle_path = version / bundle
        bundle_path.mkdir()
        for name in MODEL_BUNDLE_FILES:
            content = (
                yaml.safe_dump(
                    _asset_document(
                        f"vllm-sr/{bundle}",
                        bundle.removeprefix("chorus-"),
                        f"local/{bundle}",
                    ),
                    sort_keys=False,
                )
                if name == "config.yaml"
                else f"{bundle}:{name}\n"
            )
            (bundle_path / name).write_text(content, encoding="utf-8")
        digests[bundle] = model_bundle_digest(bundle_path)
    (version / "catalog.yaml").write_text(
        "assets:\n"
        "  - id: chorus-v1\n"
        "    bundle: chorus-v1\n"
        f"    sha256: {digests['chorus-v1']}\n"
        "  - id: chorus-v2\n"
        "    bundle: chorus-v2\n"
        f"    sha256: {digests['chorus-v2']}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sync, "SOURCE", source)
    monkeypatch.setattr(sync, "DESTINATION", destination)

    assert sync.sync() == 0
    assert (destination / "latest/chorus-v1/config.yaml").is_file()
    assert (destination / "latest/chorus-v2/metadata.yaml").is_file()

    (version / "undeclared.yaml").write_text("version: v0.3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contents differ from the declared assets"):
        sync.check()
