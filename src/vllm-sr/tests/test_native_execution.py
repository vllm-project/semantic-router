"""Native API configuration survives CLI parsing without Chat-only defaults."""

from copy import deepcopy

import pytest
import yaml
from cli.models import UserConfig
from cli.parser import parse_user_config
from cli.validator import validate_user_config
from cli.validator_native import validate_native_execution
from cli.validator_recipe_contracts import effective_entrypoints
from pydantic import ValidationError


@pytest.fixture
def native_document():
    return yaml.safe_load(
        """
version: v0.3
listeners:
  - name: native
    port: 8801
    systemone:
      models: [vllm-sr/auto, kai, strong]
providers:
  models:
    - {name: kai, api_format: systemone, deployment: local-kai}
    - name: strong
      api_format: systemone
      provider_model_id: vllm-sr/Decision-2.0-Nox-4B
      backend_refs:
        - provider: systemone-compatible
          base_url: http://localhost:8900/v1
          api_key_env: NATIVE_TEST_TOKEN
global:
  model_catalog:
    deployments:
      local-kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        device: cpu
entrypoints:
  - {api: systemone, model_names: [vllm-sr/auto], recipe: native}
recipes:
  - name: native
    routing:
      budget: {deadline: 3s, max_calls: 3}
      decisions:
        - name: classify
          rules: {}
          modelRefs: [{model: kai}, {model: strong}]
          algorithm:
            type: cascade
            quality:
              type: uncalibrated
              acceptance:
                rules:
                  - {question_type: choice, field: top_probability, predicate: {gte: 0.9}}
            stages:
              - {name: fast, kind: native, model: kai}
              - {name: strong, kind: native, model: strong}
"""
    )


def test_native_round_trip_and_listener_scope(native_document, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(native_document))
    config = parse_user_config(str(path), log_summary=False)
    assert validate_user_config(config, log_summary=False) == []
    dumped = config.model_dump(by_alias=True, exclude_none=True, exclude_unset=True)
    assert dumped == native_document
    assert config.recipes[0].routing.decisions[0].algorithm.on_error is None
    entries = effective_entrypoints(config)
    assert [(entry.api or "chat", entry.model_names) for entry in entries] == [
        ("chat", ["vllm-sr/auto"]),
        ("systemone", ["vllm-sr/auto"]),
    ]


def test_native_policy_and_calibration_survive_projection(native_document):
    document = deepcopy(native_document)
    algorithm = document["recipes"][0]["routing"]["decisions"][0]["algorithm"]
    algorithm["type"] = "policy"
    algorithm["policy"] = {
        "source": "./policy.json",
        "sha256": "a" * 64,
        "cost_weight": 0.001,
    }
    algorithm["quality"] = {
        "type": "calibrated",
        "calibration": "suite",
        "loss": "bundle_error",
        "max_risk": 0.1,
    }
    document["evaluation"] = {
        "calibrations": [
            {"name": "suite", "source": "./evidence.json", "sha256": "b" * 64}
        ]
    }
    config = UserConfig.model_validate(document)
    assert validate_native_execution(config) == []
    assert (
        config.model_dump(by_alias=True, exclude_none=True, exclude_unset=True)
        == document
    )


@pytest.mark.parametrize("field, value", [("models", ["kai"]), ("unknown_field", True)])
def test_native_stage_uses_generated_closed_contract(native_document, field, value):
    stage = native_document["recipes"][0]["routing"]["decisions"][0]["algorithm"][
        "stages"
    ][0]
    stage[field] = value
    with pytest.raises(ValidationError, match="Additional properties"):
        UserConfig.model_validate(native_document)


@pytest.mark.parametrize(
    "mutation", ["wrong_deployment", "wrong_stage", "chat_default", "missing_budget"]
)
def test_native_invalid_resource_references_are_rejected(native_document, mutation):
    if mutation == "wrong_deployment":
        native_document["providers"]["models"][0]["deployment"] = "missing"
    elif mutation == "wrong_stage":
        native_document["recipes"][0]["routing"]["decisions"][0]["algorithm"]["stages"][
            1
        ]["model"] = "missing"
    elif mutation == "chat_default":
        native_document["providers"]["defaults"] = {"model": "kai"}
    else:
        del native_document["recipes"][0]["routing"]["budget"]
    config = UserConfig.model_validate(native_document)
    assert validate_native_execution(config)


def test_native_policy_judge_is_terminal(native_document):
    document = deepcopy(native_document)
    document["providers"]["models"].append(
        {
            "name": "reviewer",
            "api_format": "openai",
            "backend_refs": [
                {
                    "provider": "openai-compatible",
                    "base_url": "http://localhost:8901/v1",
                }
            ],
        }
    )
    decision = document["recipes"][0]["routing"]["decisions"][0]
    decision["modelRefs"].append({"model": "reviewer"})
    algorithm = decision["algorithm"]
    algorithm["type"] = "policy"
    algorithm["policy"] = {"source": "./policy.json", "sha256": "a" * 64}
    judge = {
        "name": "judge",
        "model": "reviewer",
        "kind": "judge",
        "generation": {"max_output_tokens": 128},
    }
    algorithm["stages"].append(judge)
    assert validate_native_execution(UserConfig.model_validate(document)) == []
    algorithm["stages"].insert(1, algorithm["stages"].pop())
    assert any(
        "terminal judge" in str(error)
        for error in validate_native_execution(UserConfig.model_validate(document))
    )


@pytest.mark.parametrize("signal", ["authz", "metadata", "conversation", "reask", "kb"])
def test_native_rejects_unavailable_request_context(native_document, signal):
    decision = native_document["recipes"][0]["routing"]["decisions"][0]
    decision["rules"] = {"type": signal, "name": "private"}
    errors = validate_native_execution(UserConfig.model_validate(native_document))
    assert any("native request context" in str(error) for error in errors)


def test_native_rejects_unavailable_projection_input(native_document):
    native_document["recipes"][0]["routing"]["projections"] = {
        "scores": [
            {
                "name": "derived",
                "method": "weighted_sum",
                "inputs": [{"type": "conversation", "name": "history", "weight": 1}],
            }
        ]
    }
    errors = validate_native_execution(UserConfig.model_validate(native_document))
    assert any("native request context" in str(error) for error in errors)


@pytest.mark.parametrize(
    "control",
    [
        "fallback",
        "global_fallback",
        "candidate_requirements",
        "reliability",
        "output_contract",
        "adaptations",
        "weight",
        "use_reasoning",
    ],
)
def test_native_rejects_ignored_chat_controls(native_document, control):
    routing = native_document["recipes"][0]["routing"]
    decision = routing["decisions"][0]
    if control == "fallback":
        routing["fallback"] = {"enabled": True}
    elif control == "global_fallback":
        native_document["global"]["router"] = {"fallback": {"enabled": True}}
    elif control == "candidate_requirements":
        routing[control] = {"context": "known_limits"}
    elif control == "reliability":
        decision[control] = {"total_timeout": "1ms", "retry_count": 0}
    elif control == "output_contract":
        decision[control] = "text"
    elif control == "adaptations":
        decision[control] = {"mode": "observe"}
    else:
        decision["modelRefs"][0][control] = 2 if control == "weight" else False
    assert validate_native_execution(UserConfig.model_validate(native_document))


@pytest.mark.parametrize("field", ["candidateIterations", "emits"])
def test_native_rejects_chat_directives_before_pydantic_drops_them(
    native_document, field
):
    native_document["recipes"][0]["routing"]["decisions"][0][field] = [
        {"kind": "retention"}
    ]
    with pytest.raises(ValidationError, match="unsupported for native execution"):
        UserConfig.model_validate(native_document)


@pytest.mark.parametrize(
    "variant", ["contrastive", "examples", "external", "http", "mcp", "explicit"]
)
def test_native_signal_backend_lifecycle(native_document, variant):
    profile = native_document["recipes"][0]["routing"]
    profile["signals"] = {
        "preferences": [
            {"name": "brief", "description": "Brief"},
            {"name": "detailed", "description": "Detailed"},
        ]
    }
    profile["decisions"][0]["rules"] = {
        "operator": "AND",
        "conditions": [{"type": "preference", "name": "brief"}],
    }
    catalog = native_document["global"]["model_catalog"]
    if variant in {"contrastive", "explicit"}:
        catalog["modules"] = {"classifier": {"preference": {"use_contrastive": True}}}
    elif variant == "examples":
        profile["signals"]["preferences"][0]["examples"] = ["be brief"]
    elif variant == "external":
        catalog["external"] = [{"model_role": "preference"}]
    elif variant == "mcp":
        catalog["modules"] = {"classifier": {"mcp": {"enabled": True}}}
        profile["decisions"][0]["rules"]["conditions"][0]["type"] = "domain"
    if variant in {"http", "explicit"}:
        profile["model_bindings"] = {
            "preference": {
                "deployment": "local-kai" if variant == "explicit" else "remote",
                "contract": "decision.v1",
                "adapter": "decision",
            }
        }
    errors = validate_native_execution(UserConfig.model_validate(native_document))
    assert bool(errors) == (variant != "explicit")


def test_native_stage_cannot_shadow_abstention(native_document):
    native_document["recipes"][0]["routing"]["decisions"][0]["algorithm"]["stages"][0][
        "name"
    ] = "abstain"
    assert validate_native_execution(UserConfig.model_validate(native_document))
