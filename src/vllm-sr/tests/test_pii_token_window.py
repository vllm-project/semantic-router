"""Synthetic config-only PII window admission and preservation checks."""

from copy import deepcopy

import pytest
from cli.models import UserConfig
from cli.validator_model_runtime import validate_model_runtime_references


def config(pii, deployment=None, *, named_recipe=False):
    global_config = {"model_catalog": {"modules": {"classifier": {"pii": pii}}}}
    profile = {}
    if deployment is not None:
        global_config["model_catalog"]["deployments"] = {"synthetic-pii": deployment}
        profile = {
            "model_bindings": {
                "pii_classifier": {
                    "deployment": "synthetic-pii",
                    "contract": "token_spans.v1",
                    "adapter": "token_classifier",
                }
            }
        }
    raw = {"version": "v0.3", "global": global_config, "routing": profile}
    if named_recipe:
        raw["routing"] = {}
        raw["recipes"] = [{"name": "synthetic", "routing": profile}]
    return UserConfig.model_validate(raw)


@pytest.mark.parametrize(
    "pii,refused",
    [
        ({}, False),
        ({"window": None}, False),
        ({"window": {"size": 512}}, False),
        (
            {
                "use_mmbert_32k": True,
                "max_sequence_length": 8192,
                "window": {"size": 512, "overlap": 64},
            },
            False,
        ),
        ({"use_mmbert_32k": False}, False),
        ({"use_mmbert_32k": False, "window": {"size": 128}}, True),
        ({"backend": {}, "window": {"size": 128}}, True),
        ({"window": {}}, True),
        ({"window": {"size": 0}}, True),
        ({"window": {"size": 513}}, True),
        ({"window": {"size": True}}, True),
        ({"window": {"size": 128.5}}, True),
        ({"window": {"size": 128, "overlap": -1}}, True),
        ({"window": {"size": 128, "overlap": 128}}, True),
        ({"window": {"size": 128, "overlap": False}}, True),
        ({"window": [], "max_sequence_length": 1024}, True),
        ({"window": {"size": 128}, "max_sequence_length": -1}, True),
    ],
)
def test_pii_local_window(pii, refused):
    value = config(deepcopy(pii))
    before = value.model_dump(by_alias=True)
    errors = validate_model_runtime_references(value)
    assert bool(errors) == refused, errors
    assert value.model_dump(by_alias=True) == before
    assert before["global"]["model_catalog"]["modules"]["classifier"]["pii"] == pii


@pytest.mark.parametrize("provider", ["candle", "ort"])
@pytest.mark.parametrize("named_recipe", [False, True])
def test_pii_bound_window_uses_deployment_budget(provider, named_recipe):
    value = config(
        {"max_sequence_length": 512, "window": {"size": 1024, "overlap": 128}},
        {
            "provider": provider,
            "artifact": "models/synthetic-pii",
            "device": "cpu",
            "input": {"max_tokens": 8192, "overflow": "window"},
        },
        named_recipe=named_recipe,
    )
    # An unrelated unbound default profile still validates the module locally.
    # Give it its own sufficient budget when testing a named recipe.
    if named_recipe:
        value.global_["model_catalog"]["modules"]["classifier"]["pii"][
            "max_sequence_length"
        ] = 2048
    assert not validate_model_runtime_references(value)


@pytest.mark.parametrize(
    "window,budget,provider",
    [
        (None, {"max_tokens": 1024, "overflow": "window"}, "candle"),
        ({"size": 128}, {"max_tokens": 1024, "overflow": "reject"}, "candle"),
        ({"size": 128}, {"max_tokens": 0, "overflow": "window"}, "candle"),
        ({"size": 1024}, {"max_tokens": 512, "overflow": "window"}, "ort"),
        ({"size": 128}, {"max_tokens": 0, "overflow": "reject"}, "http"),
    ],
)
def test_pii_bound_window_refuses_incompatible_contract(window, budget, provider):
    deployment = {
        "provider": provider,
        "artifact": "models/synthetic-pii",
        "input": budget,
    }
    if provider == "http":
        deployment.pop("artifact")
        deployment["external_model"] = "synthetic-remote"
    value = config({"window": window}, deployment)
    errors = validate_model_runtime_references(value)
    assert any("PII window" in str(error) for error in errors), errors
