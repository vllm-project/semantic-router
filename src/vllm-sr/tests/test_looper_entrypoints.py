"""Algorithms are exposed through explicit recipe entrypoints."""

import copy

import pytest
import yaml
from cli.main import main
from cli.models import UserConfig
from cli.parser import ConfigParseError, parse_user_config_data
from cli.validator import validate_user_config
from click.testing import CliRunner

FLOW_FIELD = "global.integrations.looper.flow.model_names"

# A backend model formerly captured by an implicit Flow alias. That config
# now fails explicitly; authors must choose a distinct recipe entrypoint.
_CAPTURED_MODEL_CONFIG = {
    "version": "v0.5",
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
                "loras": [{"name": "base-sql"}],
            },
        ],
    },
    "recipes": [
        {
            "name": "default",
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
        }
    ],
    "global": {"integrations": {"looper": {"flow": {"model_names": ["gpt-oss"]}}}},
}


@pytest.mark.parametrize("family", ["flow", "remom", "fusion"])
def test_implicit_algorithm_alias_is_rejected(family):
    document = copy.deepcopy(_CAPTURED_MODEL_CONFIG)
    document["global"]["integrations"]["looper"] = {
        family: {"model_names": ["gpt-oss"]}
    }
    with pytest.raises(ConfigParseError) as refused:
        parse_user_config_data(document, log_summary=False)
    algorithm = "workflows" if family == "flow" else family
    assert str(refused.value) == (
        f"global.integrations.looper.{family}.model_names: removed_field: the model "
        "names of a Looper algorithm are removed; run `vllm-sr config migrate` (an "
        "entrypoint whose model_names name a recipe with a decision that runs the "
        f"{algorithm} algorithm)"
    )


def test_workflow_recipe_uses_an_explicit_distinct_entrypoint():
    document = copy.deepcopy(_CAPTURED_MODEL_CONFIG)
    document["global"] = {}
    default = document["recipes"][0]
    document["recipes"] = [
        default,
        {
            "name": "workflow",
            "signals": default["signals"],
            "decisions": [default["decisions"].pop(0)],
        },
    ]
    document["entrypoints"] = [
        {"model_names": ["vllm-sr/auto"], "recipe": "default", "default": True},
        {"model_names": ["team/workflow"], "recipe": "workflow"},
    ]
    assert (
        validate_user_config(UserConfig.model_validate(document), log_summary=False)
        == []
    )


def test_config_validate_rejects_the_removed_field(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_CAPTURED_MODEL_CONFIG))
    result = CliRunner().invoke(main, ["config", "validate", "--config", str(path)])
    assert result.exit_code != 0
    assert f"{FLOW_FIELD}: removed_field" in " ".join(result.output.split())
