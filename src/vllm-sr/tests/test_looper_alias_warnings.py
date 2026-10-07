"""A direct Looper alias that is also a model name warns, as the Router does."""

import copy

import yaml
from cli.main import main
from cli.models import UserConfig
from cli.validator import validate_user_config
from cli.validator_recipe_contracts import looper_alias_collision_warnings
from click.testing import CliRunner

FLOW_FIELD = "global.integrations.looper.flow.model_names"

# The shape that broke the response-api-redis profile (#4651): its backend
# model is also a Flow alias, so its requests evaluate only the workflows
# decision.
_CAPTURED_MODEL_CONFIG = {
    "version": "v0.3",
    "providers": {
        "defaults": {"model": "gpt-oss"},
        "models": [
            {
                "name": "gpt-oss",
                "backend_refs": [{"endpoint": "127.0.0.1:8000", "provider": "vllm"}],
            },
            {
                "name": "base",
                "backend_refs": [{"endpoint": "127.0.0.1:8001", "provider": "vllm"}],
            },
        ],
    },
    "routing": {
        "modelCards": [
            {"name": "gpt-oss"},
            {"name": "base", "loras": [{"name": "base-sql"}]},
        ],
        "signals": {
            "keywords": [
                {"name": "plan_keywords", "operator": "OR", "keywords": ["plan"]}
            ]
        },
        "decisions": [
            {
                "name": "workflow_route",
                "priority": 20,
                "rules": {
                    "operator": "AND",
                    "conditions": [{"type": "keyword", "name": "plan_keywords"}],
                },
                "modelRefs": [{"model": "gpt-oss"}],
                "algorithm": {
                    "type": "workflows",
                    "workflows": {
                        "mode": "static",
                        "roles": [{"name": "worker", "models": ["gpt-oss"]}],
                    },
                },
            },
            {
                "name": "default_route",
                "priority": 10,
                "rules": {"operator": "AND", "conditions": []},
                "modelRefs": [{"model": "gpt-oss"}],
            },
        ],
    },
    "global": {"integrations": {"looper": {"flow": {"model_names": ["gpt-oss"]}}}},
}


def _config(looper=None, recipes=None):
    document = copy.deepcopy(_CAPTURED_MODEL_CONFIG)
    if looper is not None:
        document["global"]["integrations"]["looper"] = looper
    if recipes is not None:
        document["recipes"] = recipes
    return document


def test_flow_alias_naming_a_served_model_warns_without_failing_validation():
    config = UserConfig.model_validate(_config())

    assert validate_user_config(config, log_summary=False) == []
    warnings = looper_alias_collision_warnings(config)

    assert len(warnings) == 1
    warning = warnings[0]
    assert warning.field == FLOW_FIELD
    assert warning.message == (
        "Flow alias 'gpt-oss' is also a model that providers.models serves and "
        "decisions 'workflow_route', 'default_route' route to; requests for it "
        "evaluate only workflows decisions, so the model cannot be requested "
        "directly and a request that matches none of them fails with no_route"
    )
    assert "vllm-sr/flow" in warning.hint


def test_distinct_looper_aliases_do_not_warn():
    config = UserConfig.model_validate(
        _config(looper={"flow": {"model_names": ["vllm-sr/flow", "team/flow"]}})
    )

    assert looper_alias_collision_warnings(config) == []


def test_every_family_and_model_reference_is_checked():
    config = UserConfig.model_validate(
        _config(
            looper={
                "remom": {"model_names": ["base-sql"]},
                "fusion": {"model_names": ["private-model"]},
            },
            recipes=[
                {
                    "name": "privacy",
                    "routing": {
                        "decisions": [
                            {
                                "name": "private_route",
                                "priority": 1,
                                "rules": {"operator": "AND", "conditions": []},
                                "modelRefs": [{"model": "private-model"}],
                            }
                        ]
                    },
                }
            ],
        )
    )

    warnings = looper_alias_collision_warnings(config)

    assert [warning.field for warning in warnings] == [
        "global.integrations.looper.remom.model_names",
        "global.integrations.looper.fusion.model_names",
    ]
    assert warnings[0].message.startswith(
        "ReMoM alias 'base-sql' is also a model that providers.models serves;"
    )
    assert warnings[1].message.startswith(
        "Fusion alias 'private-model' is also a model that decisions "
        "'privacy::private_route' route to;"
    )


def test_only_the_family_that_captures_a_name_warns():
    config = UserConfig.model_validate(
        _config(
            looper={
                "fusion": {"model_names": ["gpt-oss"]},
                "flow": {"model_names": ["gpt-oss"]},
            }
        )
    )

    warnings = looper_alias_collision_warnings(config)

    assert [warning.field for warning in warnings] == [
        "global.integrations.looper.fusion.model_names"
    ]


def test_config_validate_reports_the_warning(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_config()))

    result = CliRunner().invoke(main, ["config", "validate", "--config", str(path)])

    assert result.exit_code == 0, result.output
    assert "Configuration is valid" in result.output
    assert f"Warning: [{FLOW_FIELD}] Flow alias 'gpt-oss'" in result.output
    assert "Hint: Give the alias a name that no model uses" in result.output
