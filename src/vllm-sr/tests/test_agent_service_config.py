from __future__ import annotations

import pytest
from cli.models import UserConfig
from pydantic import ValidationError


def _config(endpoint: object = None, *, include_agent: bool = True) -> dict:
    services: dict[str, object] = {}
    if include_agent:
        services["agent"] = {"public_inference_endpoint": endpoint}
    return {"version": "v0.3", "global": {"services": services}}


def test_agent_public_front_door_is_independent_of_store_capabilities() -> None:
    config = _config("https://inference.example.test/v1/chat/completions")

    parsed = UserConfig.model_validate(config)

    assert parsed.global_["services"]["agent"] == config["global"]["services"]["agent"]
    UserConfig.model_validate(_config(include_agent=False))


@pytest.mark.parametrize(
    "endpoint",
    [
        " http://inference.example.test/v1/chat/completions",
        "grpc://inference.example.test/v1/chat/completions",
        "http://inference.example.test",
        "http://inference.example.test/v1/chat/completions/",
        "http://user@inference.example.test/v1/chat/completions",
        "http://inference.example.test/v1/chat/completions?tenant=one",
        "http://inference.example.test/v1/chat/completions#fragment",
        7,
    ],
)
def test_agent_rejects_ambiguous_public_front_doors(endpoint: object) -> None:
    with pytest.raises(ValidationError, match="public_inference_endpoint"):
        UserConfig.model_validate(_config(endpoint))
