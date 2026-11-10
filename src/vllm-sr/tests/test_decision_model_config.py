"""Systemone signals, the systemone selector and model_runtime deployments in the CLI contract.

The Router validates the deployments that answer them.
"""

import copy

import pytest
from cli.models import UserConfig
from cli.parser import parse_user_config_data
from cli.validator import collect_validation_warnings, validate_user_config
from pydantic import ValidationError

REVISION = "cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764"

BASE = {
    "version": "v0.5",
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
    "recipes": [
        {
            "name": "default",
            "signals": {
                "systemone": [
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
                            {
                                "type": "systemone",
                                "name": "request_kind",
                                "label": "code",
                            },
                            {
                                "type": "systemone",
                                "name": "needs_reasoning",
                                "on_error": "no_match",
                            },
                        ],
                    },
                    "modelRefs": [{"model": "small"}, {"model": "large"}],
                    "algorithm": {
                        "type": "systemone",
                        "systemone": {
                            "deployment": "decision-kai",
                            "instructions": "Which model should answer?",
                            "candidates": {"large": "Strong reasoning"},
                        },
                    },
                }
            ],
        }
    ],
}


def _config(mutate=None) -> dict:
    document = copy.deepcopy(BASE)
    if mutate:
        mutate(document)
    return document


def _errors(document: dict) -> list[str]:
    config = UserConfig.model_validate(document)
    return [error.message for error in validate_user_config(config, log_summary=False)]


def _rules(document: dict) -> list[dict]:
    return document["recipes"][0]["signals"]["systemone"]


def _conditions(document: dict) -> list[dict]:
    return document["recipes"][0]["decisions"][0]["rules"]["conditions"]


def test_valid_systemone_config_passes():
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
def test_systemone_questions_are_validated(question, message):
    def mutate(document):
        _rules(document)[1]["question"] = question

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
    _rules(document).extend(
        [
            {"name": "topics", "deployment": "vela2", "question": SET_QUESTION},
            {"name": "places", "deployment": "vela2", "question": SPAN_QUESTION},
        ]
    )
    _conditions(document).extend(
        [
            {"type": "systemone", "name": "topics", "label": "billing"},
            {"type": "systemone", "name": "places", "label": "city"},
        ]
    )


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
        _rules(document)[1]["question"] = question

    with pytest.raises(ValidationError, match=message):
        UserConfig.model_validate(_config(mutate))


def test_set_and_span_rules_match_themselves_and_name_counted_labels():
    def unlabelled(document):
        _with_vela2(document)
        del _conditions(document)[3]["label"]

    assert _errors(_config(unlabelled)) == []

    def undeclared(document):
        _with_vela2(document)
        _conditions(document)[2]["label"] = "x"

    def ignored(document):
        _with_vela2(document)
        _rules(document)[2]["ignore_labels"] = ["billing"]

    assert any("not an option" in error for error in _errors(_config(undeclared)))
    assert any("never count" in error for error in _errors(_config(ignored)))


@pytest.mark.parametrize(
    "rule, message",
    [
        ({"labels": ["true"]}, "a noul rule takes none"),
        ({"ignore_labels": ["x"]}, "only to set and span rules"),
        ({"input": {"previous_turns": 33}}, "less than or equal to 32"),
        ({"input": {"answer": True}}, "request-stage rule does not have"),
        ({"input": {"context": True}, "stage": "request"}, "request-stage rule"),
        ({"on_unscanned": "match"}, "only to a rule with coverage: full"),
        ({"stage": "routing"}, "request"),
        ({"coverage": "whole"}, "truncate"),
    ],
)
def test_systemone_rule_fields_are_validated(rule, message):
    def mutate(document):
        _rules(document)[1].update(rule)

    with pytest.raises(ValidationError, match=message):
        UserConfig.model_validate(_config(mutate))


def test_systemone_rule_fields_within_their_limits_pass():
    def mutate(document):
        _with_vela2(document)
        _rules(document)[1].update(
            {
                "stage": "response",
                "input": {"answer": True, "context": True, "previous_turns": 32},
                "coverage": "full",
                "on_unscanned": "no_match",
            }
        )
        _rules(document)[0]["labels"] = ["code"]
        _rules(document)[2]["ignore_labels"] = ["shipping"]
        del _conditions(document)[1]

    assert _errors(_config(mutate)) == []


@pytest.mark.parametrize(
    "labels, message",
    [
        (["chat", "chat"], "twice"),
        (["math"], "not an option"),
    ],
)
def test_rule_labels_name_options(labels, message):
    def mutate(document):
        _rules(document)[0]["labels"] = labels

    with pytest.raises(ValidationError, match=message):
        UserConfig.model_validate(_config(mutate))


def test_score_question_requires_a_predicate():
    def mutate(document):
        _rules(document)[1] = {
            "name": "difficulty",
            "deployment": "decision-kai",
            "question": {"type": "score", "instructions": "x", "levels": ["a", "b"]},
        }

    with pytest.raises(ValidationError, match="requires a predicate"):
        UserConfig.model_validate(_config(mutate))


def test_condition_labels_follow_the_question_type():
    def undeclared_choice(document):
        _conditions(document)[0]["label"] = "math"

    def labeled_noul(document):
        _conditions(document)[1]["label"] = "true"

    def unlabelled_choice(document):
        del _conditions(document)[0]["label"]

    def unlabelled_choice_with_labels(document):
        unlabelled_choice(document)
        _rules(document)[0]["labels"] = ["code"]

    assert any(
        "not an option" in error for error in _errors(_config(undeclared_choice))
    )
    assert any("takes no label" in error for error in _errors(_config(labeled_noul)))
    assert any(
        "choice rule without labels" in error
        for error in _errors(_config(unlabelled_choice))
    )
    assert _errors(_config(unlabelled_choice_with_labels)) == []


def test_selector_candidates_must_be_model_refs():
    def mutate(document):
        selector = document["recipes"][0]["decisions"][0]["algorithm"]["systemone"]
        selector["candidates"] = {"other": "x"}

    assert any(
        "not one of the decision's modelRefs" in e for e in _errors(_config(mutate))
    )


def test_selector_candidate_count_applies_once_models_are_assigned():
    def model_free(document):
        decision = document["recipes"][0]["decisions"][0]
        del decision["modelRefs"]
        del decision["algorithm"]["systemone"]["candidates"]

    def one_model(document):
        document["recipes"][0]["decisions"][0]["modelRefs"] = [{"model": "small"}]

    assert _errors(_config(model_free)) == []
    assert any("modelRefs to choose from" in e for e in _errors(_config(one_model)))


def test_selector_without_deployment_asks_the_decision_model():
    def omit(document):
        del document["recipes"][0]["decisions"][0]["algorithm"]["systemone"][
            "deployment"
        ]

    assert _errors(_config(omit)) == []

    def alternate(document):
        omit(document)
        document["global"]["model_catalog"]["system"] = {
            "decision_model": {"deployment": "decision-kai"}
        }

    assert _errors(_config(alternate)) == []

    def blank(document):
        document["recipes"][0]["decisions"][0]["algorithm"]["systemone"][
            "deployment"
        ] = " "

    with pytest.raises(ValidationError, match="omit it to ask the decision model"):
        UserConfig.model_validate(_config(blank))


def test_systemone_type_requires_its_configuration():
    def mutate(document):
        document["recipes"][0]["decisions"][0]["algorithm"] = {"type": "systemone"}

    with pytest.raises(ValidationError, match="requires systemone configuration"):
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
    for rule in _rules(document):
        del rule["deployment"]
    del document["recipes"][0]["decisions"][0]["algorithm"]["systemone"]["deployment"]
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


def test_the_decision_forms_are_read_as_systemone(tmp_path):
    """The CLI reads the decision signal, conditions and selector as the
    Router does: as systemone."""

    def written_as_decision(document):
        routing = document["recipes"][0]
        routing["signals"]["decision"] = routing["signals"].pop("systemone")
        for condition in _conditions(document):
            condition["type"] = "decision"
        selector = routing["decisions"][0]["algorithm"]
        routing["decisions"][0]["algorithm"] = {
            "type": "decision",
            "decision": selector["systemone"],
        }

    written = _config(written_as_decision)
    before = copy.deepcopy(written)
    config = parse_user_config_data(written)
    canonical = parse_user_config_data(_config())

    assert written == before
    assert validate_user_config(config, log_summary=False) == []
    assert config.model_dump() == canonical.model_dump()
