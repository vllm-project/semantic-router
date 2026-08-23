from __future__ import annotations

import pytest
from pydantic import ValidationError

from cli.models import UserConfig


def _config(*, mode: str, endpoint: object = None, include_agent: bool = True) -> dict:
    services: dict[str, object] = {
        "backend_egress": {"policy_file": "/app/config/backend-egress-policy.yaml"}
    }
    if include_agent:
        services["agent"] = {"public_inference_endpoint": endpoint}
    return {
        "version": "v0.4",
        "global": {
            "control_plane": {"mode": mode},
            "services": services,
        },
    }


def test_managed_agent_requires_one_explicit_public_front_door() -> None:
    config = _config(
        mode="managed",
        endpoint="https://inference.example.test/v1/chat/completions",
    )

    parsed = UserConfig.model_validate(config)

    assert parsed.global_["services"]["agent"] == config["global"]["services"]["agent"]
    with pytest.raises(ValidationError, match="public_inference_endpoint"):
        UserConfig.model_validate(_config(mode="managed", include_agent=False))


def test_standalone_rejects_managed_agent_service() -> None:
    UserConfig.model_validate(_config(mode="standalone", include_agent=False))

    with pytest.raises(ValidationError, match="managed-only"):
        UserConfig.model_validate(
            _config(
                mode="standalone",
                endpoint="http://inference.example.test/v1/chat/completions",
            )
        )


@pytest.mark.parametrize(
    "endpoint",
    [
        None,
        "",
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
def test_managed_agent_rejects_ambiguous_public_front_doors(endpoint: object) -> None:
    with pytest.raises(ValidationError, match="public_inference_endpoint"):
        UserConfig.model_validate(_config(mode="managed", endpoint=endpoint))
