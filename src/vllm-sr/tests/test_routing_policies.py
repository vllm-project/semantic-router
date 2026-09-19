import pytest
import yaml
from cli.config_schema import schema_document
from cli.config_schema.views import schema_view
from cli.main import main
from cli.models import RecipeRouting, RequestParamsPluginConfig, Routing, UserConfig
from cli.parser import parse_user_config
from cli.validator_recipe_contracts import validate_recipe_contracts
from click.testing import CliRunner
from pydantic import ValidationError


@pytest.mark.parametrize("model", [Routing, RecipeRouting])
def test_recipe_policy_fields_preserve_false_and_independent_dimensions(model):
    value = model.model_validate(
        {
            "candidate_requirements": {"context": "known_limits"},
            "data_policy": {"replay": False},
        }
    )
    dumped = value.model_dump(exclude_none=True, by_alias=True)
    assert dumped["candidate_requirements"] == {"context": "known_limits"}
    assert dumped["data_policy"] == {"replay": False}
    assert model().candidate_requirements is None
    assert model().data_policy is None


@pytest.mark.parametrize(
    "payload",
    [
        {"candidate_requirements": {"context": "bounded"}},
        {"candidate_requirements": {"capabilities": "inferred"}},
        {"candidate_requirements": {"unknown": True}},
        {"data_policy": {"replay": "false"}},
        {"data_policy": {"replay": False, "export": False}},
    ],
)
def test_recipe_policy_fields_reject_invalid_contract(payload):
    for model in (Routing, RecipeRouting):
        with pytest.raises(ValidationError):
            model.model_validate(payload)


def test_recipe_policy_schema_discovery():
    doc = schema_document()
    for path in ("routing.candidate_requirements", "recipes.routing.data_policy"):
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
        "version": "v0.3",
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
        "routing": {"modelCards": [{"name": "local-model"}]},
    }
    if named:
        document["recipes"] = [{"name": "balanced", "routing": routing}]
        document["entrypoints"] = [
            {"model_names": ["router/balanced"], "recipe": "balanced"}
        ]
    else:
        document["routing"].update(routing)
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
    routing = parsed.recipes[0].routing if named else parsed.routing
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


@pytest.mark.parametrize(
    "policy",
    [
        {"candidate_requirements": {"context": "known_limits"}},
        {"data_policy": {"replay": False}},
    ],
)
def test_policy_only_default_conflict_matches_router(policy):
    cfg = UserConfig.model_validate(
        {
            "version": "v0.3",
            "routing": policy,
            "recipes": [{"name": "default", "routing": {}}],
        }
    )
    errors = validate_recipe_contracts(cfg)
    assert any(error.field == "recipes.default" for error in errors)
    cfg.routing = Routing()
    assert validate_recipe_contracts(cfg) == []
