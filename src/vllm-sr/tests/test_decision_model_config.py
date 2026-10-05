"""Decision signals, the decision selector and model_runtime deployments in the CLI contract."""

import copy

import pytest
from cli.models import UserConfig
from cli.validator import validate_user_config
from cli.validator_decision_model import model_runtime_deployment_error
from pydantic import ValidationError

REVISION = "881bee413681d80ebeac86afcda8b4138dae516e"

BASE = {
    "version": "v0.3",
    "listeners": [{"name": "http", "address": "0.0.0.0", "port": 8899}],
    "providers": {
        "defaults": {"model": "small"},
        "models": [
            {"name": "small", "backend_refs": [{"endpoint": "small:8000"}]},
            {"name": "large", "backend_refs": [{"endpoint": "large:8000"}]},
        ],
    },
    "global": {
        "model_catalog": {
            "deployments": {
                "decision-kai": {
                    "provider": "model_runtime",
                    "artifact": "vllm-sr/Decision-2.0-Kai-0.6B",
                    "revision": REVISION,
                }
            }
        }
    },
    "routing": {
        "modelCards": [{"name": "small"}, {"name": "large"}],
        "signals": {
            "decision": [
                {
                    "name": "request_kind",
                    "deployment": "decision-kai",
                    "question": {
                        "type": "choice",
                        "instructions": "What kind of request is this?",
                        "choices": [{"key": "code"}, {"key": "chat"}],
                    },
                },
                {
                    "name": "needs_reasoning",
                    "deployment": "decision-kai",
                    "question": {"type": "noul", "instructions": "Reasoning?"},
                    "predicate": {"gte": 0.7},
                },
            ]
        },
        "decisions": [
            {
                "name": "code",
                "priority": 10,
                "rules": {
                    "operator": "AND",
                    "conditions": [
                        {"type": "decision", "name": "request_kind", "label": "code"},
                        {
                            "type": "decision",
                            "name": "needs_reasoning",
                            "on_error": "no_match",
                        },
                    ],
                },
                "modelRefs": [{"model": "small"}, {"model": "large"}],
                "algorithm": {
                    "type": "decision",
                    "decision": {
                        "deployment": "decision-kai",
                        "instructions": "Which model should answer?",
                        "candidates": {"large": "Strong reasoning"},
                    },
                },
            }
        ],
    },
}


def _config(mutate=None) -> dict:
    document = copy.deepcopy(BASE)
    if mutate:
        mutate(document)
    return document


def _errors(document: dict) -> list[str]:
    config = UserConfig.model_validate(document)
    return [error.message for error in validate_user_config(config, log_summary=False)]


def test_valid_decision_model_config_passes():
    assert _errors(_config()) == []


@pytest.mark.parametrize(
    "question, message",
    [
        ({"type": "choice", "instructions": "x", "choices": [{"key": "a"}]}, "2..255"),
        ({"type": "score", "instructions": "x", "levels": ["low"]}, "2..10 levels"),
        (
            {"type": "noul", "instructions": "x", "choices": [{"key": "y"}]},
            "false and true",
        ),
        ({"type": "noul", "instructions": "x", "levels": ["a", "b"]}, "only to score"),
        ({"type": "vote", "instructions": "x"}, "choice"),
    ],
)
def test_decision_questions_are_validated(question, message):
    def mutate(document):
        document["routing"]["signals"]["decision"][1]["question"] = question

    with pytest.raises(ValidationError, match=message):
        UserConfig.model_validate(_config(mutate))


def test_score_question_requires_a_predicate():
    def mutate(document):
        document["routing"]["signals"]["decision"][1] = {
            "name": "difficulty",
            "deployment": "decision-kai",
            "question": {"type": "score", "instructions": "x", "levels": ["a", "b"]},
        }

    with pytest.raises(ValidationError, match="requires a predicate"):
        UserConfig.model_validate(_config(mutate))


def test_condition_labels_follow_the_question_type():
    def undeclared_choice(document):
        conditions = document["routing"]["decisions"][0]["rules"]["conditions"]
        conditions[0]["label"] = "math"

    def labeled_noul(document):
        conditions = document["routing"]["decisions"][0]["rules"]["conditions"]
        conditions[1]["label"] = "true"

    assert any(
        "declared choice key" in error for error in _errors(_config(undeclared_choice))
    )
    assert any("takes no label" in error for error in _errors(_config(labeled_noul)))


def test_references_must_name_model_runtime_deployments():
    def unknown(document):
        document["routing"]["signals"]["decision"][0]["deployment"] = "missing"

    def wrong_provider(document):
        document["global"]["model_catalog"]["deployments"]["decision-kai"] = {
            "provider": "candle",
            "artifact": "models/kai",
        }

    assert any("is not declared" in error for error in _errors(_config(unknown)))
    assert any(
        "must use provider model_runtime" in error
        for error in _errors(_config(wrong_provider))
    )


def test_selector_candidates_must_be_model_refs():
    def mutate(document):
        selector = document["routing"]["decisions"][0]["algorithm"]["decision"]
        selector["candidates"] = {"other": "x"}

    assert any(
        "not one of the decision's modelRefs" in e for e in _errors(_config(mutate))
    )


def test_decision_type_requires_its_configuration():
    def mutate(document):
        document["routing"]["decisions"][0]["algorithm"] = {"type": "decision"}

    with pytest.raises(ValidationError, match="requires decision configuration"):
        UserConfig.model_validate(_config(mutate))


@pytest.mark.parametrize(
    "deployment, message",
    [
        ({"artifact": "vllm-sr/Decision-2.0-Kai-0.6B"}, None),
        ({"endpoint": "unix:///run/vllm-sr/runtime.sock"}, None),
        ({"endpoint": "http://runtime:8100", "device": "rocm:1"}, None),
        ({"artifact": "/models/kai", "profile": "batching"}, None),
        ({"artifact": "./models/kai"}, "Hub repository ID or an absolute"),
        ({"artifact": "/models/kai", "revision": REVISION}, "only to Hub"),
        ({"artifact": "vllm-sr/x", "revision": "main"}, "40-hex"),
        ({"artifact": "vllm-sr/x", "device": "metal"}, "device must be"),
        ({"artifact": "vllm-sr/x", "profile": "fast"}, "profile must be"),
        ({"artifact": "vllm-sr/x", "precision": "fp16"}, "precision native"),
        (
            {"artifact": "vllm-sr/x", "input": {"overflow": "truncate"}},
            "never truncate",
        ),
        ({"endpoint": "tcp://runtime:8100"}, "unix://, http:// or https://"),
        ({"endpoint": "unix://relative.sock"}, "absolute socket path"),
        ({}, "requires artifact"),
    ],
)
def test_model_runtime_deployment_rules(deployment, message):
    error = model_runtime_deployment_error({"provider": "model_runtime", **deployment})
    if message is None:
        assert error is None
    else:
        assert error is not None and message in error


def test_other_providers_reject_runtime_fields_and_bindings():
    def runtime_field(document):
        document["global"]["model_catalog"]["deployments"]["local"] = {
            "provider": "candle",
            "artifact": "models/x",
            "profile": "exact",
        }

    def binding(document):
        document["global"]["model_catalog"]["bindings"] = {
            "domain_classifier": {
                "deployment": "decision-kai",
                "contract": "label_distribution.v1",
                "adapter": "sequence_classification",
            }
        }

    assert any(
        "apply only to model_runtime" in e for e in _errors(_config(runtime_field))
    )
    assert any("not task bindings" in e for e in _errors(_config(binding)))
