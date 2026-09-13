from cli.config_migration import migrate_config_data
from cli.models import UserConfig


def test_inline_reasoning_activation_parameter_survives_migration_and_schema():
    source = {
        "version": "v0.3",
        "providers": {
            "models": [
                {
                    "name": "private-model",
                    "reasoning": {
                        "type": "reasoning_effort",
                        "parameter": "reasoning_effort",
                        "activation_parameter": "enable_thinking",
                        "levels": ["disabled", "low", "high"],
                        "default": "low",
                        "disabled": "disabled",
                    },
                    "backend_refs": [
                        {
                            "endpoint": "127.0.0.1:8000",
                            "provider": "vllm",
                        }
                    ],
                }
            ]
        },
        "routing": {},
    }

    migrated = migrate_config_data(source)
    assert migrate_config_data(migrated) == migrated

    config = UserConfig.model_validate(migrated)
    reasoning = config.providers.models[0].reasoning
    assert reasoning is not None
    assert reasoning.activation_parameter == "enable_thinking"
    dumped = config.model_dump(mode="python", by_alias=True, exclude_none=True)
    assert (
        dumped["providers"]["models"][0]["reasoning"]["activation_parameter"]
        == "enable_thinking"
    )
