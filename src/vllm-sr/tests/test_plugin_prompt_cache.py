import pytest
from cli.models import PluginConfig, PromptCachePluginConfig
from pydantic import ValidationError


def test_prompt_cache_plugin_parses_complete_contract() -> None:
    config = PromptCachePluginConfig(
        enabled=True,
        ttl="1h",
        targets=["instructions", "tools"],
        on_unsupported="reject",
    )

    assert config.model_dump() == {
        "enabled": True,
        "ttl": "1h",
        "targets": ["instructions", "tools"],
        "on_unsupported": "reject",
    }
    plugin = PluginConfig(type="prompt_cache", configuration=config.model_dump())
    assert plugin.type.value == "prompt_cache"


def test_prompt_cache_plugin_defaults_disabled() -> None:
    config = PromptCachePluginConfig()

    assert config.model_dump() == {
        "enabled": False,
        "ttl": "5m",
        "targets": ["instructions", "tools"],
        "on_unsupported": "skip",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ttl", "10m"),
        ("targets", ["messages"]),
        ("targets", []),
        ("targets", ["tools", "tools"]),
        ("on_unsupported", "ignore"),
        ("unknown", True),
    ],
)
def test_prompt_cache_plugin_rejects_invalid_contract(
    field: str,
    value: object,
) -> None:
    payload = {"enabled": True, field: value}
    with pytest.raises(ValidationError):
        PromptCachePluginConfig(**payload)
