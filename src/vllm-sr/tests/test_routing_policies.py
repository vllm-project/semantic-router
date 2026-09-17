import pytest
from cli.config_schema import schema_document
from cli.config_schema.views import schema_view
from cli.models import RecipeRouting, RequestParamsPluginConfig, Routing, UserConfig
from cli.validator_recipe_contracts import validate_recipe_contracts
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


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "4096"])
def test_request_params_default_rejects_nonpositive_or_fractional(value):
    with pytest.raises(ValidationError):
        RequestParamsPluginConfig(default_max_tokens=value)


def test_request_params_default_is_optional_and_discoverable():
    assert RequestParamsPluginConfig().default_max_tokens is None
    cfg = RequestParamsPluginConfig(default_max_tokens=4096, max_tokens_limit=8192)
    assert cfg.model_dump(exclude_none=True)["default_max_tokens"] == 4096
    schema = schema_document()["$defs"]["RequestParamsPluginConfig"]
    field = schema["properties"]["default_max_tokens"]
    assert field["type"] == "integer"
    assert field["minimum"] == 1


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
