from cli.models import UserConfig


def test_inline_reasoning_activation_parameter_survives_the_schema():
    source = {
        "version": "v0.5",
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
                        {"endpoint": "127.0.0.1:8000", "provider": "vllm"}
                    ],
                }
            ]
        },
        "recipes": [{"name": "default"}],
    }

    config = UserConfig.model_validate(source)
    reasoning = config.providers.models[0].reasoning
    assert reasoning is not None
    assert reasoning.activation_parameter == "enable_thinking"
    dumped = config.model_dump(mode="python", by_alias=True, exclude_none=True)
    assert (
        dumped["providers"]["models"][0]["reasoning"]["activation_parameter"]
        == "enable_thinking"
    )
