import pytest
from cli.models import UserConfig
from pydantic import ValidationError


def human_config() -> dict:
    return {
        "version": "v0.3",
        "providers": {
            "defaults": {"default_model": "local/primary"},
            "models": [
                {
                    "name": "local/primary",
                    "provider_model_id": "upstream-primary",
                    "backend_refs": [
                        {
                            "name": "primary",
                            "provider": "vllm",
                            "endpoint": "model.example:8000",
                            "protocol": "http",
                        }
                    ],
                    "control": {
                        "retry": {
                            "count": 2,
                            "on": ["unavailable", "timeout"],
                        },
                        "timeout": {"request": "30s", "stream": "2m"},
                    },
                    "pricing": {
                        "input_cost_per_million_tokens": "0.5",
                        "output_cost_per_million_tokens": "1",
                        "cache_read_cost_per_million_tokens": "0.1",
                        "cache_write_cost_per_million_tokens": "0.7",
                    },
                }
            ],
        },
        "routing": {
            "modelCards": [
                {
                    "name": "local/primary",
                    "description": "Primary local model",
                    "capabilities": ["chat", "tools"],
                }
            ]
        },
        "recipes": [
            {
                "name": "balance",
                "routing": {
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
                "model_names": ["vllm-sr/balance", "balance"],
                "recipe": "balance",
                "assignments": {"simple": {"models": [{"model": "local/primary"}]}},
            }
        ],
        "global": {"billing": {"currency": "USD"}},
    }


def test_current_v03_contract_keeps_connections_and_metadata_separate() -> None:
    config = UserConfig.model_validate(human_config())
    dumped = config.model_dump(mode="json", by_alias=True, exclude_none=True)

    provider_model = dumped["providers"]["models"][0]
    assert set(provider_model) == {
        "name",
        "provider_model_id",
        "backend_refs",
        "control",
        "pricing",
    }
    assert dumped["routing"]["modelCards"][0]["name"] == "local/primary"
    assert dumped["recipes"][0]["routing"]["decisions"][0]["name"] == "simple"
    assert dumped["entrypoints"][0]["model_names"] == [
        "vllm-sr/balance",
        "balance",
    ]


def test_current_v03_schema_exposes_only_additive_model_control() -> None:
    definitions = UserConfig.model_json_schema()["$defs"]

    assert set(definitions["Model"]["properties"]) == {
        "name",
        "reasoning_family",
        "provider_model_id",
        "backend_refs",
        "control",
        "pricing",
        "api_format",
        "external_model_ids",
    }
    assert set(definitions["ModelControl"]["properties"]) == {
        "retry",
        "timeout",
    }
    assert set(definitions["Entrypoint"]["properties"]) == {
        "model_names",
        "recipe",
        "assignments",
    }


@pytest.mark.parametrize(
    ("path", "field", "value"),
    [
        (("providers", "models", 0), "id", "mdl_primary"),
        (("providers", "models", 0), "reliability", {"retry_count": 2}),
        (("providers", "models", 0), "runtime", {"max_retries": 2}),
        (("recipes", 0), "document", {}),
        (("entrypoints", 0), "name", "vllm-sr/balance"),
    ],
)
def test_current_v03_rejects_middle_state_fields(
    path: tuple[object, ...], field: str, value: object
) -> None:
    payload = human_config()
    target: object = payload
    for part in path:
        target = target[part]
    target[field] = value

    with pytest.raises(ValidationError, match=field):
        UserConfig.model_validate(payload)


def test_current_v03_requires_quoted_decimal_prices() -> None:
    payload = human_config()
    payload["providers"]["models"][0]["pricing"]["input_cost_per_million_tokens"] = 0.5

    with pytest.raises(ValidationError, match="string"):
        UserConfig.model_validate(payload)


def test_current_v03_requires_one_global_currency_for_priced_models() -> None:
    payload = human_config()
    del payload["global"]["billing"]

    with pytest.raises(ValidationError, match=r"global\.billing\.currency"):
        UserConfig.model_validate(payload)


def test_current_v03_retry_defaults_to_unavailable() -> None:
    payload = human_config()
    payload["providers"]["models"][0]["control"]["retry"] = {"count": 1}

    parsed = UserConfig.model_validate(payload)

    assert parsed.providers.models[0].control.retry.on == ["unavailable"]


def test_current_v03_rejects_public_mode_and_static_access_policy() -> None:
    payload = human_config()
    payload["global"]["control_plane"] = {"mode": "legacy"}

    with pytest.raises(ValidationError, match=r"global\.control_plane"):
        UserConfig.model_validate(payload)

    payload = human_config()
    payload["global"]["ratelimit"] = {"rpm": 10}
    with pytest.raises(ValidationError, match="static inference access policy"):
        UserConfig.model_validate(payload)
