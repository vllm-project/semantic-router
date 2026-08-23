import os
import tempfile

import yaml

from cli.parser import parse_user_config
from cli.validator import validate_user_config


def _parse_config(hybrid: dict):
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
                        "domains": [
                            {"name": "general", "description": "General domain"}
                        ]
                    },
                    "decisions": [
                        {
                            "name": "hybrid_route",
                            "description": "Hybrid route",
                            "priority": 100,
                            "rules": {
                                "operator": "AND",
                                "conditions": [{"type": "domain", "name": "general"}],
                            },
                            "algorithm": {"type": "hybrid", "hybrid": hybrid},
                        }
                    ],
                },
            }
        ],
        "entrypoints": [
            {
                "name": "test_entrypoint",
                "recipe": "test_recipe",
                "assignments": {"hybrid_route": {"models": [{"model": "test_model"}]}},
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
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
        temp_path = handle.name

    try:
        return parse_user_config(temp_path)
    finally:
        os.unlink(temp_path)


def test_validate_user_config_rejects_all_zero_hybrid_weights():
    config = _parse_config(
        {
            "experience_weight": 0,
            "router_dc_weight": 0,
            "automix_weight": 0,
            "cost_weight": 0,
        }
    )
    errors = validate_user_config(config)

    assert any("all zero" in e.message for e in errors)


def test_validate_user_config_accepts_partial_hybrid_weights():
    config = _parse_config({"experience_weight": 0.6, "router_dc_weight": 0.4})
    errors = validate_user_config(config)

    assert errors == []
