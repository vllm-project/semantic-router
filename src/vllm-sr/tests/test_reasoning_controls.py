"""Regressions for reasoning controls discovered through remote CLI startup."""

import copy

import pytest
import yaml
from cli.algorithms import ModelRef
from cli.main import main as cli
from cli.models import UserConfig
from cli.validator_reasoning import validate_reasoning_controls
from click.testing import CliRunner


def _config(ref, *, named=True, family="qwen3"):
    routing = {
        "decisions": [
            {
                "name": "default",
                "priority": 1,
                "modelRefs": [{"model": "local-model", **ref}],
            }
        ]
    }
    document = {
        "version": "v0.3",
        "providers": {
            "models": [
                {
                    "name": "local-model",
                    "reasoning": {"family": family},
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
        document["recipes"] = [{"name": "balance", "routing": routing}]
        document["entrypoints"] = [{"model_names": ["my/balance"], "recipe": "balance"}]
    else:
        document["routing"].update(routing)
    return document


@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize(
    "ref, expected",
    [
        (
            {"use_reasoning": True, "reasoning_effort": "high"},
            "mode-only reasoning family",
        ),
        ({"reasoning_mode": "enabled"}, "use_reasoning conflicts"),
        (
            {"use_reasoning": False, "reasoning_mode": "enabled"},
            "use_reasoning conflicts",
        ),
        ({"use_reasoning": True, "reasoning_mode": "adaptive"}, "not supported"),
        (
            {"use_reasoning": False, "reasoning_effort": "low"},
            "while reasoning is disabled",
        ),
        (
            {"use_reasoning": True, "reasoning_effort": " high "},
            "surrounding whitespace",
        ),
    ],
)
def test_config_validate_rejects_controls_before_runtime(
    tmp_path, named, ref, expected
):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_config(ref, named=named)))
    result = CliRunner().invoke(cli, ["config", "validate", "--config", str(path)])
    assert result.exit_code != 0, result.output
    assert expected in result.output
    if named:
        assert "balance" in result.output


@pytest.mark.parametrize(
    "ref",
    [
        {},
        {"use_reasoning": False},
        {"use_reasoning": True, "reasoning_mode": "enabled"},
        {"use_reasoning": False, "reasoning_mode": "disabled"},
    ],
)
def test_qwen_mode_contract_and_serialization_preserve_authored_choice(ref):
    document = _config(ref)
    original = copy.deepcopy(document)
    parsed = UserConfig.model_validate(document)
    assert validate_reasoning_controls(parsed) == []
    serialized = parsed.model_dump(by_alias=True, exclude_none=True)
    model_ref = serialized["recipes"][0]["routing"]["decisions"][0]["modelRefs"][0]
    assert ("use_reasoning" in model_ref) == ("use_reasoning" in ref)
    if "use_reasoning" in ref:
        assert model_ref["use_reasoning"] == ref["use_reasoning"]
    assert document == original


def test_omitted_model_ref_reasoning_remains_unspecified():
    assert ModelRef(model="local-model").use_reasoning is None
    assert "use_reasoning" not in ModelRef(model="local-model").model_dump(
        exclude_none=True
    )


def test_catalog_bound_model_inherits_always_on_family():
    document = _config({})
    model = document["providers"]["models"][0]
    model.pop("reasoning")
    model["catalog"] = "anthropic/claude-fable-5.1"
    errors = validate_reasoning_controls(UserConfig.model_validate(document))
    assert any("always-on" in error.message for error in errors)


@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("catalog_bound", [False, True])
def test_catalog_named_custom_model_keeps_its_reasoning_contract(
    tmp_path, named, catalog_bound
):
    document = _config({"use_reasoning": False}, named=named)
    model_name = "openai/gpt-oss-20b"
    model = document["providers"]["models"][0]
    model.pop("reasoning")
    model["name"] = model_name
    if catalog_bound:
        model["catalog"] = model_name
    document["routing"]["modelCards"] = [{"name": model_name}]
    routing = document["recipes"][0]["routing"] if named else document["routing"]
    routing["decisions"][0]["modelRefs"][0]["model"] = model_name
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document))

    result = CliRunner().invoke(cli, ["config", "validate", "--config", str(path)])

    if catalog_bound:
        assert result.exit_code != 0, result.output
        assert "always-on" in result.output
    else:
        assert result.exit_code == 0, result.output
        assert "Configuration is valid" in result.output


def test_familyless_custom_effort_remains_passthrough_but_mode_requires_family():
    document = _config({"use_reasoning": True, "reasoning_effort": "custom"})
    document["providers"]["models"][0].pop("reasoning")
    assert validate_reasoning_controls(UserConfig.model_validate(document)) == []
    document["recipes"][0]["routing"]["decisions"][0]["modelRefs"][0][
        "reasoning_mode"
    ] = "enabled"
    assert (
        "requires a reasoning family"
        in validate_reasoning_controls(UserConfig.model_validate(document))[0].message
    )
