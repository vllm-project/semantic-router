import pytest
from pydantic import ValidationError

from cli.models import UserConfig


def human_config() -> dict:
    return {
        "version": "v0.4",
        "billing_currency": "USD",
        "models": [
            {
                "name": "local/primary",
                "card": {
                    "description": "Primary local model",
                    "capabilities": ["chat", "tools"],
                    "reasoning": {
                        "type": "reasoning_effort",
                        "efforts": ["medium", "high"],
                    },
                },
                "connections": [
                    {
                        "provider": "vllm",
                        "interface": "chat",
                        "endpoint": "http://model.example/v1",
                        "model": "upstream-primary",
                    }
                ],
                "runtime": {
                    "max_retries": 2,
                    "request_timeout": "30s",
                    "stream_timeout": "2m",
                },
                "pricing": {
                    "input_cost_per_million_tokens": "0.5",
                    "output_cost_per_million_tokens": "1",
                },
            }
        ],
        "recipes": [
            {
                "name": "balance",
                "document": {
                    "decisions": [
                        {
                            "name": "simple",
                            "rules": {"operator": "AND", "conditions": []},
                        }
                    ]
                },
            }
        ],
        "entrypoints": [
            {
                "name": "vllm-sr/auto",
                "aliases": ["auto"],
                "recipe": "balance",
                "assignments": {"simple": {"models": [{"model": "local/primary"}]}},
            }
        ],
        "global": {
            "services": {
                "backend_egress": {
                    "policy_file": "/app/config/backend-egress-policy.yaml"
                }
            }
        },
    }


def test_human_v04_contract_contains_no_compiler_owned_state() -> None:
    config = UserConfig.model_validate(human_config())
    dumped = config.model_dump(mode="json", by_alias=True, exclude_none=True)

    model = dumped["models"][0]
    assert set(model) == {"name", "card", "connections", "runtime", "pricing"}
    assert model["connections"] == [
        {
            "provider": "vllm",
            "interface": "chat",
            "endpoint": "http://model.example/v1",
            "model": "upstream-primary",
            "weight": "1",
        }
    ]
    assert dumped["recipes"][0]["document"]["decisions"][0]["name"] == "simple"
    assert (
        dumped["entrypoints"][0]["assignments"]["simple"]["models"][0]["model"]
        == "local/primary"
    )
    forbidden = {
        "id",
        "revision",
        "provider_catalog_revision",
        "backends",
        "model_id",
        "recipe_id",
    }
    assert not forbidden.intersection(str(dumped).replace("'", '"').split('"'))


def test_recipe_decision_schema_is_model_independent() -> None:
    schema = UserConfig.model_json_schema()
    definitions = schema["$defs"]
    properties = definitions["RecipeDecision"]["properties"]

    assert "id" not in properties
    assert "modelRefs" not in properties
    assert set(definitions["Model"]["properties"]) == {
        "name",
        "card",
        "connections",
        "runtime",
        "pricing",
    }
    assert "interface" in definitions["ModelConnection"]["properties"]
    assert "interface_" not in definitions["ModelConnection"]["properties"]
    assert set(definitions["Entrypoint"]["properties"]) == {
        "name",
        "aliases",
        "recipe",
        "assignments",
        "rules",
    }


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("model", "id", "mdl_primary"),
        ("model", "provider_catalog_revision", "sha256:" + "a" * 64),
        ("model", "backends", []),
        ("recipe", "id", "rcp_balance"),
        ("decision", "id", "dec_simple"),
        ("entrypoint", "id", "ep_auto"),
    ],
)
def test_human_v04_contract_rejects_compiler_owned_fields(
    section: str, field: str, value: object
) -> None:
    payload = human_config()
    targets = {
        "model": payload["models"][0],
        "recipe": payload["recipes"][0],
        "decision": payload["recipes"][0]["document"]["decisions"][0],
        "entrypoint": payload["entrypoints"][0],
    }
    targets[section][field] = value

    with pytest.raises(ValidationError, match=field):
        UserConfig.model_validate(payload)


def test_conditional_entrypoint_keeps_recipe_and_assignments_inside_each_rule() -> None:
    payload = human_config()
    payload["entrypoints"] = [
        {
            "name": "vllm-sr/conditional",
            "rules": [
                {
                    "name": "premium",
                    "matches": [{"claim": {"name": "tier", "exact": "premium"}}],
                    "recipe": "balance",
                    "assignments": {"simple": {"models": [{"model": "local/primary"}]}},
                }
            ],
        }
    ]

    config = UserConfig.model_validate(payload)

    assert config.entrypoints[0].rules[0].recipe == "balance"
    assert config.entrypoints[0].recipe is None
