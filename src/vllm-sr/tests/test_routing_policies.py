import pytest
import yaml
from cli.config_schema import schema_document
from cli.config_schema.views import schema_view
from cli.main import main
from cli.models import Recipe, RequestParamsPluginConfig, UserConfig
from cli.parser import parse_user_config
from cli.validator_recipe_contracts import validate_recipe_contracts
from click.testing import CliRunner
from pydantic import ValidationError


def test_recipe_policy_fields_preserve_false_and_independent_dimensions():
    value = Recipe.model_validate(
        {
            "name": "default",
            "candidate_requirements": {"context": "known_limits"},
        }
    )
    dumped = value.model_dump(exclude_none=True, by_alias=True)
    assert dumped["candidate_requirements"] == {"context": "known_limits"}
    assert Recipe(name="default").candidate_requirements is None
    assert "data_policy" not in dumped


@pytest.mark.parametrize(
    "payload",
    [
        {"candidate_requirements": {"context": "bounded"}},
        {"candidate_requirements": {"capabilities": "inferred"}},
        {"candidate_requirements": {"unknown": True}},
        {"data_policy": {"replay": False}},
        {"strategy": "priority", "data_policy": {"replay": False}},
        {"data_policy": {"replay": False, "export": False}},
    ],
)
def test_recipe_policy_fields_reject_invalid_contract(payload):
    with pytest.raises(ValidationError):
        Recipe.model_validate({"name": "default", **payload})


def test_personal_data_capture_override_does_not_require_a_pii_signal():
    config = UserConfig.model_validate(
        {
            "version": "v0.5",
            "global": {
                "plugins": [
                    {
                        "type": "router_replay",
                        "configuration": {"capture_personal_data": False},
                    }
                ]
            },
            "recipes": [
                {
                    "name": "default",
                    "decisions": [
                        {
                            "name": "everything",
                            "priority": 1,
                            "plugins": [
                                {
                                    "type": "router_replay",
                                    "configuration": {"capture_personal_data": False},
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )
    assert validate_recipe_contracts(config) == []


def test_recipe_policy_schema_discovery():
    doc = schema_document()
    for path in ("recipes.candidate_requirements", "global.services.router_replay"):
        result = schema_view(doc, view="section", path=path, expanded=True)
        assert result["x-vllm-sr-view"]["path"] == path
    requirements = doc["$defs"]["CandidateRequirements"]["properties"]
    assert requirements["capabilities"]["enum"] == ["declared"]
    assert requirements["context"]["enum"] == ["known_limits"]
    assert doc["$defs"]["MultiFactorSelectionConfig"]["properties"]["latency_metric"][
        "enum"
    ] == ["ttft", "tpot"]


INVALID_OUTPUT_DEFAULTS = [
    0,
    -1,
    1.0,
    1.5,
    4096.0,
    True,
    False,
    "4096",
    "AUTO",
    " auto",
    "auto ",
    "",
]


@pytest.mark.parametrize("value", INVALID_OUTPUT_DEFAULTS)
def test_request_params_default_rejects_invalid_values(value):
    with pytest.raises(ValidationError):
        RequestParamsPluginConfig(default_max_tokens=value)


@pytest.mark.parametrize("value", [1, 4096, "auto"])
def test_request_params_default_is_optional_and_discoverable(value):
    assert RequestParamsPluginConfig().default_max_tokens is None
    cfg = RequestParamsPluginConfig(default_max_tokens=value, max_tokens_limit=8192)
    assert cfg.model_dump(exclude_none=True)["default_max_tokens"] == value
    schema = schema_document()["$defs"]["RequestParamsPluginConfig"]
    field = schema["properties"]["default_max_tokens"]
    assert field["oneOf"] == [
        {"type": "integer", "minimum": 1},
        {"type": "string", "const": "auto"},
    ]


@pytest.mark.parametrize(
    "params",
    [
        ["temperature", "top_p"],
        ["verbosity", "stream_options", "user", "truncation"],
        ["previous_response_id", "conversation", "auto_store"],
        ["reasoning.summary", "reasoning.display"],
    ],
)
def test_request_params_blocked_params_accepts_valid_fields(params):
    cfg = RequestParamsPluginConfig(blocked_params=params)
    assert cfg.blocked_params == params


@pytest.mark.parametrize(
    "invalid_param",
    [
        "logprobs",
        "top_logprobs",
        "model",
        "messages",
        "",
        "   ",
        "unknown_field",
    ],
)
def test_request_params_blocked_params_rejects_invalid_fields(invalid_param):
    with pytest.raises(ValidationError):
        RequestParamsPluginConfig(blocked_params=[invalid_param])


def _request_params_config(value, *, named=True):
    routing = {
        "decisions": [
            {
                "name": "default-route",
                "priority": 1,
                "modelRefs": [{"model": "local-model"}],
                "plugins": [
                    {
                        "type": "request_params",
                        "configuration": {"default_max_tokens": value},
                    }
                ],
            }
        ]
    }
    document = {
        "version": "v0.5",
        "providers": {
            "models": [
                {
                    "name": "local-model",
                    "backend_refs": [
                        {
                            "name": "primary",
                            "provider": "vllm",
                            "endpoint": "localhost:8000",
                            "protocol": "http",
                        }
                    ],
                }
            ]
        },
        "recipes": [{"name": "default"}],
    }
    if named:
        document["recipes"] = [{"name": "balanced", **routing}]
        document["entrypoints"] = [
            {"model_names": ["router/balanced"], "recipe": "balanced"}
        ]
    else:
        document["recipes"][0].update(routing)
    return document


@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("value", [1, 4096, "auto"])
def test_request_params_default_cli_validation_and_config_roundtrip(
    tmp_path, named, value
):
    document = _request_params_config(value, named=named)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document))
    runner = CliRunner()
    result = runner.invoke(main, ["config", "validate", "--config", str(path)])
    assert result.exit_code == 0, result.output

    parsed = parse_user_config(str(path), log_summary=False)
    routing = parsed.recipes[0]
    assert routing.decisions[0].plugins[0].configuration == {
        "default_max_tokens": value
    }

    generated = runner.invoke(main, ["config", "router", "--config", str(path)])
    assert generated.exit_code == 0, generated.output
    assert yaml.safe_load(generated.stdout) == document
    roundtrip = tmp_path / "roundtrip.yaml"
    roundtrip.write_text(generated.stdout)
    revalidated = runner.invoke(
        main, ["config", "validate", "--config", str(roundtrip)]
    )
    assert revalidated.exit_code == 0, revalidated.output
    envoy = runner.invoke(main, ["config", "envoy", "--config", str(roundtrip)])
    assert envoy.exit_code == 0, envoy.output
    assert yaml.safe_load(envoy.stdout)["static_resources"]["clusters"]


@pytest.mark.parametrize("value", INVALID_OUTPUT_DEFAULTS)
def test_request_params_default_cli_rejects_invalid_values(tmp_path, value):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_request_params_config(value)))
    result = CliRunner().invoke(main, ["config", "validate", "--config", str(path)])
    assert result.exit_code != 0, result.output
    assert "default_max_tokens" in result.output
