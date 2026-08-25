"""Shared canonical fixture builders for configuration loader tests."""

from pathlib import Path

import yaml


def write_minimal_config(path: Path) -> None:
    """Write the smallest complete current-v0.3 runtime document."""

    path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "listeners": [
                    {"name": "http-8899", "address": "0.0.0.0", "port": 8899}
                ],
                "providers": {
                    "models": [
                        {
                            "name": "demo-model",
                            "provider_model_id": "demo-model",
                            "backend_refs": [
                                {
                                    "provider": "openai-compatible",
                                    "base_url": "http://127.0.0.1:8000/v1",
                                    "weight": 1,
                                }
                            ],
                        }
                    ]
                },
                "routing": {
                    "modelCards": [
                        {
                            "name": "demo-model",
                            "description": "Model used by parser tests.",
                            "capabilities": ["chat"],
                        }
                    ]
                },
                "recipes": [
                    {
                        "name": "default",
                        "routing": {
                            "decisions": [
                                {
                                    "name": "default-route",
                                    "description": "fallback",
                                    "priority": 100,
                                    "rules": {"operator": "AND", "conditions": []},
                                }
                            ]
                        },
                    }
                ],
                "entrypoints": [
                    {
                        "model_names": ["vllm-sr/default"],
                        "recipe": "default",
                        "assignments": {
                            "default-route": {"models": [{"model": "demo-model"}]}
                        },
                    }
                ],
                "global": {
                    "services": {
                        "backend_egress": {
                            "policy_file": "/app/config/backend-egress-policy.yaml"
                        }
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
