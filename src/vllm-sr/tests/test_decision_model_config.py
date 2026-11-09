"""Decision signals, the decision selector and model_runtime deployments in the CLI contract.

The Router validates the deployments that answer them.
"""

import copy

import pytest
from cli.models import UserConfig
from cli.validator import collect_validation_warnings, validate_user_config
from pydantic import ValidationError

REVISION = "cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764"

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


SET_QUESTION = {
    "type": "set",
    "instructions": "Which topics does the request mention?",
    "labels": [{"key": "billing", "description": "payments"}, {"key": "shipping"}],
    "threshold": 0.4,
}
SPAN_QUESTION = {
    "type": "span",
    "instructions": "Which spans name a place?",
    "labels": [{"key": "city"}],
    "head": "broad",
}


def _with_vela2(document):
    document["global"]["model_catalog"]["deployments"]["vela2"] = {
        "provider": "model_runtime",
        "artifact": "vllm-sr/Vela-2.0-0.3B",
    }
    document["routing"]["signals"]["decision"] += [
        {"name": "topics", "deployment": "vela2", "question": SET_QUESTION},
        {"name": "places", "deployment": "vela2", "question": SPAN_QUESTION},
    ]
    document["routing"]["decisions"][0]["rules"]["conditions"] += [
        {"type": "decision", "name": "topics", "label": "billing"},
        {"type": "decision", "name": "places", "label": "city"},
    ]


def test_set_and_span_questions_route_on_a_declared_label():
    assert _errors(_config(_with_vela2)) == []


@pytest.mark.parametrize(
    "question, message",
    [
        ({**SET_QUESTION, "labels": []}, "1..255 labels"),
        ({**SET_QUESTION, "labels": [{"key": "a"}, {"key": "a"}]}, "unique"),
        ({**SET_QUESTION, "choices": [{"key": "a"}, {"key": "b"}]}, "takes labels"),
        ({**SET_QUESTION, "threshold": 1.5}, "less than or equal to 1"),
        ({**SET_QUESTION, "head": "router"}, "only to span"),
        ({**SPAN_QUESTION, "head": "wide"}, "router"),
        ({**SPAN_QUESTION, "levels": ["a", "b"]}, "takes labels"),
        (
            {"type": "choice", "instructions": "x", "labels": [{"key": "a"}]},
            "only to set and span",
        ),
        ({"type": "noul", "instructions": "x", "threshold": 0.5}, "only to set"),
    ],
)
def test_set_and_span_questions_are_validated(question, message):
    def mutate(document):
        document["routing"]["signals"]["decision"][1]["question"] = question

    with pytest.raises(ValidationError, match=message):
        UserConfig.model_validate(_config(mutate))


def test_set_and_span_conditions_name_a_declared_label():
    def undeclared(document):
        _with_vela2(document)
        document["routing"]["decisions"][0]["rules"]["conditions"][2]["label"] = "x"

    def unlabelled(document):
        _with_vela2(document)
        del document["routing"]["decisions"][0]["rules"]["conditions"][3]["label"]

    assert any("declared label" in error for error in _errors(_config(undeclared)))
    assert any("declared label" in error for error in _errors(_config(unlabelled)))


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


def test_selector_candidates_must_be_model_refs():
    def mutate(document):
        selector = document["routing"]["decisions"][0]["algorithm"]["decision"]
        selector["candidates"] = {"other": "x"}

    assert any(
        "not one of the decision's modelRefs" in e for e in _errors(_config(mutate))
    )


def test_selector_candidate_count_applies_once_models_are_assigned():
    def model_free(document):
        decision = document["routing"]["decisions"][0]
        del decision["modelRefs"]
        del decision["algorithm"]["decision"]["candidates"]

    def one_model(document):
        document["routing"]["decisions"][0]["modelRefs"] = [{"model": "small"}]

    assert _errors(_config(model_free)) == []
    assert any("modelRefs to choose from" in e for e in _errors(_config(one_model)))


def test_selector_without_deployment_asks_the_decision_model():
    def omit(document):
        del document["routing"]["decisions"][0]["algorithm"]["decision"]["deployment"]

    assert _errors(_config(omit)) == []

    def alternate(document):
        omit(document)
        document["global"]["model_catalog"]["system"] = {
            "decision_model": {"deployment": "decision-kai"}
        }

    assert _errors(_config(alternate)) == []

    def blank(document):
        document["routing"]["decisions"][0]["algorithm"]["decision"]["deployment"] = " "

    with pytest.raises(ValidationError, match="omit it to ask the decision model"):
        UserConfig.model_validate(_config(blank))


def test_decision_type_requires_its_configuration():
    def mutate(document):
        document["routing"]["decisions"][0]["algorithm"] = {"type": "decision"}

    with pytest.raises(ValidationError, match="requires decision configuration"):
        UserConfig.model_validate(_config(mutate))


@pytest.mark.parametrize(
    "deployment, warned",
    [
        ({"artifact": "vllm-sr/d3-FP8"}, True),
        ({"artifact": "vllm-sr/d3-FP8", "revision": REVISION}, False),
        ({"artifact": "VLLM-SR/decision-2.0-kai-0.6b"}, False),
        ({"artifact": "vllm-sr/d3-turbo"}, False),
        ({"artifact": "vllm-sr/d3-ultra"}, False),
        ({"artifact": "/models/d3-FP8"}, False),
        ({"artifact": "vllm-sr/d3-FP8", "endpoint": "http://runtime:8100"}, False),
        (
            {
                "artifact": "vllm-sr/d3-FP8",
                "replicas": [{"endpoint": "http://runtime:8100"}, {"device": "rocm:0"}],
            },
            True,
        ),
    ],
)
def test_unpinned_hub_deployments_are_warned(deployment, warned):
    document = copy.deepcopy(BASE)
    document["global"]["model_catalog"]["deployments"]["decision-kai"] = {
        "provider": "model_runtime",
        **deployment,
    }
    config = UserConfig.model_validate(document)
    assert _errors(document) == []
    warnings = [
        str(warning)
        for warning in collect_validation_warnings(config)
        if "decision-kai" in str(warning)
    ]
    assert bool(warnings) == warned
    if warned:
        assert "vllm-sr/d3-FP8 is not a built-in model" in warnings[0]


def test_model_runtime_deployments_serve_task_bindings():
    def binding(document):
        document["global"]["model_catalog"]["deployments"]["vela-domain"] = {
            "provider": "model_runtime",
            "artifact": "vllm-sr/Vela-1.0-Encoder-307M-Domain",
            "device": "cpu",
            "input": {"max_tokens": 512, "overflow": "truncate"},
        }
        document["global"]["model_catalog"]["bindings"] = {"domain": "vela-domain"}

    assert _errors(_config(binding)) == []


def test_systemone_listener_bounds_the_request_deadline():
    document = copy.deepcopy(BASE)
    systemone = {"models": ["vllm-sr/Decision-2.0-Kai-0.6B"], "max_deadline_ms": 120000}
    document["listeners"][0]["systemone"] = systemone
    assert _errors(document) == []
    for limit in (0, -1, "10m"):
        systemone["max_deadline_ms"] = limit
        with pytest.raises(ValidationError, match="max_deadline_ms"):
            UserConfig.model_validate(document)


def test_decision_window_input_is_a_supported_scan_budget():
    document = copy.deepcopy(BASE)
    document["global"]["model_catalog"]["deployments"]["decision-kai"]["input"] = {
        "max_tokens": 4096,
        "overflow": "window",
    }
    assert _errors(document) == []


def _ask_the_default_decision_model(document):
    for rule in document["routing"]["signals"]["decision"]:
        del rule["deployment"]
    del document["routing"]["decisions"][0]["algorithm"]["decision"]["deployment"]
    document["global"]["model_catalog"]["system"] = {
        "decision_model": {"deployment": "decision-kai"}
    }


def test_default_decision_model_accepts_a_scan_budget():
    def mutate(document):
        _ask_the_default_decision_model(document)
        document["global"]["model_catalog"]["deployments"]["decision-kai"]["input"] = {
            "max_tokens": 4096,
            "overflow": "window",
        }

    assert _errors(_config(mutate)) == []


def test_systemone_publication_accepts_a_scan_budget():
    def mutate(document):
        document["listeners"][0]["systemone"] = {
            "models": ["vllm-sr/Decision-2.0-Kai-0.6B"]
        }
        document["global"]["model_catalog"]["deployments"]["decision-kai"]["input"] = {
            "max_tokens": 4096,
            "overflow": "window",
        }

    assert _errors(_config(mutate)) == []
