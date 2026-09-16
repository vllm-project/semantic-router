"""Keep public assignment roles aligned with the packaged routing policy."""

from __future__ import annotations

from pathlib import Path

import yaml
from cli import model_catalog
from cli.model_catalog_export import packaged_model_catalog_document


def test_virtual_assignment_roles_match_decisions_that_invoke_backends() -> None:
    document = packaged_model_catalog_document()
    models = {model["id"]: model for model in document["models"]}
    virtual = [model for model in models.values() if model["kind"] == "virtual"]
    source_root = Path(__file__).resolve().parents[3]
    authored = yaml.safe_load(
        (
            source_root / "config/catalog/resources/models/virtual/vllm-sr.yaml"
        ).read_text()
    )
    provenance = {model["id"]: model["verification"] for model in authored}
    asset_root = model_catalog._model_assets_root().joinpath("latest")
    assert len(virtual) == 5
    for model in virtual:
        assert model["generation"] == 1
        assert model["policy_version"] == "2.0.0"
        assert {
            key: value
            for key, value in model["verification"].items()
            if key != "asset_sha256"
        } == provenance[model["id"]]
        bundle = asset_root.joinpath(model["asset"])
        metadata = yaml.safe_load(bundle.joinpath("metadata.yaml").read_text())
        assert model["policy_version"] == metadata["version"]
        asset = yaml.safe_load(bundle.joinpath("config.yaml").read_text())
        recipe = next(
            item for item in asset["recipes"] if item["name"] == model["recipe"]
        )
        backend_decisions = [
            decision
            for decision in recipe["routing"]["decisions"]
            if not any(
                plugin["type"] == "fast_response"
                for plugin in decision.get("plugins", [])
            )
        ]
        assert [role["name"] for role in model["roles"]] == [
            decision["name"] for decision in backend_decisions
        ]
        for role, decision in zip(model["roles"], backend_decisions, strict=True):
            assert role["required"] is True
            assert role["minimum_candidates"] == decision.get("algorithm", {}).get(
                "minimum_candidates", 1
            )
            quality_index = (
                decision.get("algorithm", {})
                .get("multi_factor", {})
                .get("quality", {})
                .get("index")
            )
            for model_id in role["recommended_pool"]:
                card = models[model_id]
                assert card["kind"] == "physical"
                assert card["limits"]["context_window_size"] > 0
                assert card["limits"]["max_output_tokens"] > 0
                if quality_index:
                    assert any(
                        result["model"] == model_id
                        and result["index"] == quality_index
                        and result["status"] == "available"
                        and result["score"] is not None
                        for result in document["index_results"]
                    )
        if model["recipe"] == "vault":
            assert "local_only" not in model["traits"]
            assert all(not role["recommended_pool"] for role in model["roles"])
