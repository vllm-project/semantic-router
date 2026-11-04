"""A systemone deployment, a remote System One API, as a decision model in the CLI contract."""

import copy

import pytest
from cli.commands import runtime_serve_config
from cli.commands.runtime_mode_config import apply_instance_options
from cli.models import UserConfig
from cli.validator import validate_user_config

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
                "hosted-d3": {
                    "provider": "systemone",
                    "endpoint": "https://systemone.example.com/v1/systemone",
                    "model": "vllm-sr/d3-ultra",
                    "auth": {"api_key_env": "HOSTED_D3_TEST_KEY"},
                    "timeout_ms": 3000,
                },
                "gateway-d3": {
                    "provider": "systemone",
                    "endpoint": "https://gateway.example.com/Unified/v1/systemone",
                    "model": "vLLM-SR-d3-27B",
                    "question_types": ["choice", "noul", "score"],
                    "auth": {
                        "header": "Ocp-Apim-Subscription-Key",
                        "prefix": "",
                        "api_key_file": "/run/secrets/gateway-key",
                    },
                },
            },
            "system": {"decision_model": {"deployment": "hosted-d3"}},
        }
    },
    "routing": {
        "modelCards": [{"name": "small"}, {"name": "large"}],
        "signals": {
            "decision": [
                {
                    "name": "support_team",
                    "deployment": "gateway-d3",
                    "question": {
                        "type": "choice",
                        "instructions": "Which team should handle this ticket?",
                        "choices": [{"key": "billing"}, {"key": "engineering"}],
                    },
                },
                {
                    "name": "needs_reasoning",
                    "question": {"type": "noul", "instructions": "Reasoning?"},
                },
            ]
        },
        "decisions": [
            {
                "name": "engineering",
                "priority": 10,
                "rules": {
                    "operator": "AND",
                    "conditions": [
                        {
                            "type": "decision",
                            "name": "support_team",
                            "label": "engineering",
                            "on_error": "no_match",
                        },
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
                    "decision": {"instructions": "Which model should answer?"},
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


def _deployment(name):
    return lambda document: document["global"]["model_catalog"]["deployments"][name]


def test_systemone_deployments_answer_questions_selectors_and_the_default():
    assert _errors(_config()) == []

    def named_selector(document):
        selector = document["routing"]["decisions"][0]["algorithm"]["decision"]
        selector["deployment"] = "gateway-d3"

    assert _errors(_config(named_selector)) == []


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: _deployment("hosted-d3")(d).update({"device": "cpu"}), "not device"),
        (
            lambda d: _deployment("hosted-d3")(d).update({"max_num_seqs": 4}),
            "not max_num_seqs",
        ),
        (
            lambda d: _deployment("hosted-d3")(d).update(
                {"endpoint": "systemone.example.com/v1/systemone"}
            ),
            "endpoint must be the http:// or https:// URL",
        ),
        (
            lambda d: _deployment("hosted-d3")(d).update(
                {"endpoint": "https://user:secret@systemone.example.com/v1/systemone"}
            ),
            "endpoint must not carry credentials",
        ),
        (lambda d: _deployment("hosted-d3")(d).pop("model"), "model must name"),
        (
            lambda d: _deployment("hosted-d3")(d).update(
                {"auth": {"header": "Authorization"}}
            ),
            "exactly one of api_key_env and api_key_file",
        ),
        (
            lambda d: _deployment("gateway-d3")(d)["auth"].update(
                {"api_key_file": "secrets/key"}
            ),
            "auth.api_key_file must be an absolute path",
        ),
        (
            lambda d: _deployment("gateway-d3")(d).update(
                {"question_types": ["choice", "box"]}
            ),
            "question_types lists distinct types",
        ),
        (
            lambda d: _deployment("gateway-d3")(d).update({"presets": ["pii", "pii"]}),
            "presets lists distinct ready-made questions",
        ),
        (
            lambda d: _deployment("hosted-d3")(d).update({"timeout_ms": 90000}),
            "timeout_ms must be within",
        ),
    ],
)
def test_systemone_deployments_take_only_their_own_fields(mutate, message):
    assert any(message in error for error in _errors(_config(mutate)))


def test_a_systemone_deployment_declares_its_presets():
    document = _config(
        lambda d: _deployment("hosted-d3")(d).update({"presets": ["pii", "halu"]})
    )
    assert not _errors(document)


def test_a_task_never_binds_a_systemone_deployment():
    def bind(document):
        document["global"]["model_catalog"]["bindings"] = {
            "domain_classifier": {
                "deployment": "hosted-d3",
                "contract": "label_distribution.v1",
                "adapter": "vela",
            }
        }

    assert any(
        "answers decision signals and selectors only" in error
        for error in _errors(_config(bind))
    )


def test_serving_a_systemone_decision_model_starts_no_local_one():
    document = _config()
    before = copy.deepcopy(document)
    apply_instance_options(document, platform="rocm")
    assert (
        document["global"]["model_catalog"]["deployments"]
        == before["global"]["model_catalog"]["deployments"]
    )
    runtime_serve_config._check_served_decision_model(
        document, "cpu", local_host=True
    )
    with pytest.raises(ValueError, match="is a systemone deployment"):
        apply_instance_options(
            copy.deepcopy(before),
            model_options={"artifact": "vllm-sr/vela3-fast"},
            platform="cpu",
        )
