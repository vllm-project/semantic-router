import sys
from pathlib import Path

import yaml
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.config_migration import migrate_config_data  # noqa: E402
from cli.main import main  # noqa: E402


def test_migrate_config_data_splits_legacy_provider_models():
    legacy = {
        "version": "v0.1",
        "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        "signals": {
            "keywords": [
                {"name": "math_terms", "operator": "OR", "keywords": ["algebra"]}
            ]
        },
        "decisions": [
            {
                "name": "default-route",
                "description": "fallback",
                "priority": 100,
                "rules": {"operator": "AND", "conditions": []},
                "modelRefs": [{"model": "gpt-4o"}],
            }
        ],
        "providers": {
            "default_model": "gpt-4o",
            "reasoning_families": {
                "openai": {"type": "reasoning_effort", "parameter": "reasoning_effort"}
            },
            "default_reasoning_effort": "high",
            "models": [
                {
                    "name": "gpt-4o",
                    "endpoints": [
                        {
                            "name": "primary",
                            "endpoint": "api.openai.com:443",
                            "protocol": "https",
                            "weight": 100,
                        }
                    ],
                    "access_key": "sk-test",
                    "reasoning_family": "openai",
                    "description": "General reasoning model",
                    "capabilities": ["general", "reasoning"],
                    "modality": "text",
                    "quality_score": 0.95,
                }
            ],
        },
        "memory": {
            "enabled": True,
            "default_retrieval_limit": 3,
        },
    }

    migrated = migrate_config_data(legacy)

    assert migrated["version"] == "v0.3"
    assert migrated["routing"]["signals"]["keywords"][0]["name"] == "math_terms"
    assert migrated["routing"]["decisions"][0]["name"] == "default-route"
    assert migrated["routing"]["modelCards"] == [
        {
            "name": "gpt-4o",
            "reasoning_family_ref": "openai",
            "description": "General reasoning model",
            "capabilities": ["general", "reasoning"],
            "quality_score": 0.95,
            "modality": "text",
        }
    ]
    assert migrated["providers"]["models"] == [
        {
            "name": "gpt-4o",
            "backend_refs": [
                {
                    "name": "primary",
                    "endpoint": "api.openai.com:443",
                    "protocol": "https",
                    "weight": 100,
                    "api_key": "sk-test",
                }
            ],
        }
    ]
    assert migrated["global"]["memory"]["enabled"] is True


def test_cli_config_migrate_writes_canonical_yaml(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.1",
                "listeners": [
                    {"name": "http-8899", "address": "0.0.0.0", "port": 8899}
                ],
                "providers": {
                    "default_model": "gpt-4o-mini",
                    "models": [
                        {
                            "name": "gpt-4o-mini",
                            "endpoints": [
                                {
                                    "name": "primary",
                                    "endpoint": "host.docker.internal:8000",
                                    "protocol": "http",
                                    "weight": 100,
                                }
                            ],
                        }
                    ],
                },
                "decisions": [
                    {
                        "name": "default-route",
                        "description": "fallback",
                        "priority": 100,
                        "rules": {"operator": "AND", "conditions": []},
                        "modelRefs": [{"model": "gpt-4o-mini"}],
                    }
                ],
            },
            sort_keys=False,
        )
    )

    runner = CliRunner()
    result = runner.invoke(main, ["config", "migrate", "--config", str(config_path)])

    assert result.exit_code == 0

    migrated_path = tmp_path / "config.migrated.yaml"
    migrated = yaml.safe_load(migrated_path.read_text())

    assert migrated["version"] == "v0.3"
    assert migrated["providers"]["default_model"] == "gpt-4o-mini"
    assert migrated["routing"]["modelCards"][0]["name"] == "gpt-4o-mini"
    assert migrated["providers"]["models"][0]["backend_refs"] == [
        {
            "name": "primary",
            "endpoint": "host.docker.internal:8000",
            "protocol": "http",
            "weight": 100,
        }
    ]
