import importlib
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

UserConfig = importlib.import_module("cli.models").UserConfig


def _decision(**overrides):
    decision = {
        "name": "vision_request",
        "priority": 1000,
        "rules": {
            "operator": "AND",
            "conditions": [{"type": "conversation", "name": "image_input"}],
        },
        "modelRefs": [{"model": "vision-model"}],
    }
    decision.update(overrides)
    return decision


def test_keyword_signal_include_history_parses_and_defaults_off():
    config = UserConfig(
        version="0.3",
        routing={
            "signals": {
                "keywords": [
                    {
                        "name": "local_only_markers",
                        "operator": "OR",
                        "keywords": ["local model only"],
                        "include_history": True,
                    },
                    {"name": "urgent", "operator": "OR", "keywords": ["urgent"]},
                ]
            },
            "decisions": [_decision()],
        },
    )

    opted_in, default = config.routing.signals.keywords
    assert opted_in.include_history is True
    assert default.include_history is False


def test_decision_without_description_parses():
    config = UserConfig(version="0.3", routing={"decisions": [_decision()]})

    assert config.decisions[0].description is None


def test_decision_with_description_still_parses():
    config = UserConfig(
        version="0.3",
        routing={"decisions": [_decision(description="Routes vision requests.")]},
    )

    assert config.decisions[0].description == "Routes vision requests."


def test_decision_route_action_requires_jailbreak_condition():
    with pytest.raises(ValueError, match="jailbreak condition"):
        UserConfig(
            version="0.3",
            routing={
                "decisions": [
                    _decision(action={"type": "route", "destination": "safe-model"})
                ]
            },
        )


def test_decision_route_action_with_jailbreak_condition_parses():
    decision = _decision(
        rules={
            "operator": "AND",
            "conditions": [{"type": "jailbreak", "name": "prompt_injection"}],
        },
        action={"type": "route", "destination": "safe-model"},
    )
    config = UserConfig(version="0.3", routing={"decisions": [decision]})

    assert config.decisions[0].action.destination == "safe-model"


def test_decision_route_action_accepts_root_leaf_jailbreak_rule():
    decision = _decision(
        rules={"type": "jailbreak", "name": "prompt_injection"},
        action={"type": "route", "destination": "safe-model"},
    )
    config = UserConfig(version="0.3", routing={"decisions": [decision]})

    assert config.decisions[0].action.destination == "safe-model"
    assert config.decisions[0].rules.conditions[0].type == "jailbreak"


def test_decision_route_action_accepts_nested_jailbreak_rule():
    decision = _decision(
        rules={
            "operator": "AND",
            "conditions": [
                {
                    "operator": "OR",
                    "conditions": [{"type": "jailbreak", "name": "prompt_injection"}],
                }
            ],
        },
        action={"type": "route", "destination": "safe-model"},
    )
    config = UserConfig(version="0.3", routing={"decisions": [decision]})

    assert config.decisions[0].action.destination == "safe-model"


def test_decision_route_action_rejects_unknown_type():
    with pytest.raises(ValueError):
        _ = UserConfig(
            version="0.3",
            routing={
                "decisions": [
                    _decision(action={"type": "block", "destination": "safe-model"})
                ]
            },
        )


def test_custom_evaluation_and_model_evidence_round_trip():
    config = UserConfig.model_validate(
        {
            "version": "0.3",
            "evaluation": {
                "benchmarks": [
                    {
                        "id": "acme/legal-reasoning@1.0.0",
                        "display_name": "Acme Legal Reasoning",
                        "domain": "reasoning",
                        "default_profile": "default",
                        "profiles": [
                            {
                                "id": "default",
                                "display_name": "Default",
                                "description": "Frozen public test split.",
                            }
                        ],
                        "metrics": [
                            {
                                "id": "accuracy",
                                "unit": "percent",
                                "direction": "higher_is_better",
                                "range": [0, 100],
                            }
                        ],
                    }
                ],
                "indices": [
                    {
                        "id": "acme/legal@1.0.0",
                        "display_name": "Acme Legal",
                        "aggregation": "weighted_mean",
                        "scale": [0, 100],
                        "missing": {"policy": "require_all"},
                        "components": [
                            {
                                "benchmark": "acme/legal-reasoning@1.0.0",
                                "metric": "accuracy",
                                "weight": 1,
                                "normalization": {
                                    "type": "linear_clamp",
                                    "min": 0,
                                    "max": 100,
                                },
                            }
                        ],
                    }
                ],
                "records": [
                    {
                        "model": "acme-model",
                        "benchmark": "acme/legal-reasoning@1.0.0",
                        "metrics": {"accuracy": 82},
                    }
                ],
            },
            "routing": {"modelCards": [{"name": "acme-model"}]},
        }
    )

    dumped = config.model_dump(by_alias=True, exclude_none=True)
    assert dumped["evaluation"]["indices"][0]["id"] == "acme/legal@1.0.0"
    assert dumped["evaluation"]["records"][0]["model"] == "acme-model"
    assert dumped["evaluation"]["records"][0]["metrics"]["accuracy"] == 82


def test_custom_evaluation_requires_versioned_namespace():
    with pytest.raises(ValueError):
        UserConfig.model_validate(
            {
                "version": "0.3",
                "evaluation": {
                    "benchmarks": [
                        {
                            "id": "legal-reasoning",
                            "display_name": "Legal Reasoning",
                            "domain": "reasoning",
                            "default_profile": "default",
                            "profiles": [
                                {
                                    "id": "default",
                                    "display_name": "Default",
                                    "description": "Frozen split.",
                                }
                            ],
                            "metrics": [
                                {
                                    "id": "accuracy",
                                    "unit": "percent",
                                    "direction": "higher_is_better",
                                    "range": [0, 100],
                                }
                            ],
                        }
                    ]
                },
            }
        )


def _backend_ref(**overrides):
    backend_ref = importlib.import_module("cli.models").BackendRef
    ref = {"name": "gpu-a", "endpoint": "10.0.0.1:8000", "provider": "vllm"}
    ref.update(overrides)
    return backend_ref.model_validate(ref)


def test_backend_ref_accepts_engine_and_metrics():
    ref = _backend_ref(
        engine="tensorrt-llm", metrics={"port": 8002, "path": "/metrics"}
    )

    assert ref.engine == "tensorrt-llm"
    assert ref.metrics.port == 8002
    assert ref.model_dump(exclude_none=True)["metrics"] == {
        "port": 8002,
        "path": "/metrics",
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"engine": "triton"},
        {"metrics": {"port": 9000}},
        {"engine": "vllm", "metrics": {"port": 70000}},
        {"engine": "vllm", "metrics": {"path": "metrics"}},
        {"engine": "vllm", "metrics": {"unknown": 1}},
    ],
)
def test_backend_ref_rejects_invalid_telemetry(overrides):
    with pytest.raises(ValueError):
        _backend_ref(**overrides)
