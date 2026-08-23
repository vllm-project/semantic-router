"""Tests for latency-aware algorithm validation behavior."""

import os
import tempfile

import yaml

from cli.parser import parse_user_config
from cli.validator import validate_user_config


def _parse_config(*, conditions: list[dict], latency_aware: dict):
    data = {
        "version": "v0.4",
        "listeners": [{"name": "http-8888", "address": "0.0.0.0", "port": 8888}],
        "models": [
            {
                "name": "test_model",
                "card": {
                    "description": "Test model",
                    "capabilities": ["chat"],
                },
                "connections": [
                    {
                        "provider": "vllm",
                        "endpoint": "http://localhost:8000/v1",
                        "model": "test_model",
                    }
                ],
            }
        ],
        "recipes": [
            {
                "name": "test_recipe",
                "document": {
                    "signals": {
                        "domains": [{"name": "math", "description": "Math domain"}]
                    },
                    "decisions": [
                        {
                            "name": "math_decision",
                            "description": "Math decision",
                            "priority": 100,
                            "rules": {
                                "operator": "AND",
                                "conditions": conditions,
                            },
                            "algorithm": {
                                "type": "latency_aware",
                                "latency_aware": latency_aware,
                            },
                        }
                    ],
                },
            }
        ],
        "entrypoints": [
            {
                "name": "test_entrypoint",
                "recipe": "test_recipe",
                "assignments": {"math_decision": {"models": [{"model": "test_model"}]}},
            }
        ],
        "global": {
            "services": {
                "backend_dispatch": {
                    "bind_address": "127.0.0.1",
                    "port": 8180,
                    "audience": "vllm-sr.backend-dispatch",
                    "capability_ttl": "30s",
                    "max_request_body_bytes": 64 << 20,
                },
                "backend_egress": {
                    "policy_file": "/app/config/backend-egress-policy.yaml"
                },
            }
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(data, f, sort_keys=False)
        temp_path = f.name

    try:
        return parse_user_config(temp_path)
    finally:
        os.unlink(temp_path)


def test_validate_rejects_unsupported_signal_condition():
    config = _parse_config(
        conditions=[
            {"type": "domain", "name": "math"},
            {"type": "latency", "name": "low_latency"},
        ],
        latency_aware={"tpot_percentile": 20},
    )
    errors = validate_user_config(config)

    assert any(
        "uses unsupported signal type 'latency'" in str(error) for error in errors
    )


def test_validate_accepts_latency_aware_configuration():
    config = _parse_config(
        conditions=[{"type": "domain", "name": "math"}],
        latency_aware={"tpot_percentile": 20, "ttft_percentile": 20},
    )
    errors = validate_user_config(config)

    assert errors == []
