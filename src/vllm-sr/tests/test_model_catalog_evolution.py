"""Evolution contracts for versioned built-in model catalogs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml
from cli import model_catalog
from cli.model_bundle import MODEL_BUNDLE_FILES, model_bundle_digest
from cli.model_catalog import ModelCatalogError


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


def test_role_validation_scopes_future_generations_to_their_asset_and_recipe() -> None:
    v1 = _asset_document("vllm-sr/mom-v1-blend", "balance", "local/v1")
    v2 = _asset_document("vllm-sr/mom-v2-blend", "next", "local/v2")
    v1["recipes"][0]["routing"]["decisions"] = [{"modelRefs": [{"model": "local/v1"}]}]
    v2["recipes"][0]["routing"]["decisions"] = [{"modelRefs": [{"model": "local/v2"}]}]

    model_catalog._validate_catalog_model_role_pool(
        "vllm-sr/mom-v1-blend",
        "balance",
        ({"recommended_pool": ["local/v1", "operator/alternative"]},),
        v1,
    )
    model_catalog._validate_catalog_model_role_pool(
        "vllm-sr/mom-v2-blend",
        "next",
        ({"recommended_pool": ["local/v2"]},),
        v2,
    )

    with pytest.raises(ModelCatalogError, match=r"vllm-sr/mom-v2-blend.*local/v2"):
        model_catalog._validate_catalog_model_role_pool(
            "vllm-sr/mom-v2-blend",
            "next",
            ({"recommended_pool": ["local/v1"]},),
            v2,
        )


def test_role_validation_accepts_connection_free_recipe_templates() -> None:
    document = _asset_document("vllm-sr/mom-v1-blend", "balance", "local/v1")
    document.pop("providers")
    document["recipes"][0]["routing"]["decisions"] = []

    model_catalog._validate_catalog_model_role_pool(
        "vllm-sr/mom-v1-blend",
        "balance",
        ({"recommended_pool": ["operator/assigned-model"]},),
        document,
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
    for bundle in ("mom-v1", "mom-v2"):
        bundle_path = version / bundle
        bundle_path.mkdir()
        for name in MODEL_BUNDLE_FILES:
            content = (
                yaml.safe_dump(
                    _asset_document(
                        f"vllm-sr/{bundle}-blend",
                        bundle.removeprefix("mom-"),
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
        "  - id: mom-v1\n"
        "    bundle: mom-v1\n"
        f"    sha256: {digests['mom-v1']}\n"
        "  - id: mom-v2\n"
        "    bundle: mom-v2\n"
        f"    sha256: {digests['mom-v2']}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sync, "SOURCE", source)
    monkeypatch.setattr(sync, "DESTINATION", destination)

    assert sync.sync() == 0
    assert (destination / "latest/mom-v1/config.yaml").is_file()
    assert (destination / "latest/mom-v2/metadata.yaml").is_file()

    (version / "undeclared.yaml").write_text("version: v0.3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contents differ from the declared assets"):
        sync.check()
