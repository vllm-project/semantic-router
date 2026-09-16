"""Backend recommendations are independent of required operator assignments."""

from __future__ import annotations

import pytest
from cli.model_catalog_validation import ModelCatalogError, _parse_roles


@pytest.mark.parametrize(
    "advisory",
    ({}, {"recommended_pool": []}, {"recommended_pool": ["operator/model"]}),
)
def test_optional_advisories_preserve_required_assignment_cardinality(advisory) -> None:
    roles = _parse_roles(
        [
            {
                "name": "review",
                "required": True,
                "minimum_candidates": 2,
                "traits": ["reasoning"],
                **advisory,
            }
        ],
        "vllm-sr/example",
    )
    assert len(roles) == 1
    assert roles[0]["required"] is True
    assert roles[0]["minimum_candidates"] == 2
    assert roles[0]["recommended_pool"] == advisory.get("recommended_pool", [])


@pytest.mark.parametrize(
    "invalid",
    (
        {"minimum_candidates": 0},
        {"minimum_candidates": True},
        {"required": False},
        {"recommended_pool": None},
        {"recommended_pool": ["operator/model", "operator/model"]},
        {"recommended_pool": ["../model"]},
    ),
)
def test_optional_advisories_do_not_relax_role_or_reference_validation(invalid) -> None:
    with pytest.raises(ModelCatalogError):
        _parse_roles(
            [
                {
                    "name": "review",
                    "required": True,
                    "minimum_candidates": 2,
                    "traits": ["reasoning"],
                    **invalid,
                }
            ],
            "vllm-sr/example",
        )
