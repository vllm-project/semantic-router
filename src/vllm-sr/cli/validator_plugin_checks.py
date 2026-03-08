"""Plugin configuration validation helpers."""

from pydantic import ValidationError as PydanticValidationError

from cli.models import (
    FastResponsePluginConfig,
    HallucinationPluginConfig,
    HeaderMutationPluginConfig,
    MemoryPluginConfig,
    PluginType,
    RAGPluginConfig,
    RouterReplayPluginConfig,
    SemanticCachePluginConfig,
    SystemPromptPluginConfig,
    UserConfig,
)
from cli.validator_common import ValidationError

_PLUGIN_CONFIG_MODELS = {
    PluginType.SEMANTIC_CACHE.value: SemanticCachePluginConfig,
    PluginType.FAST_RESPONSE.value: FastResponsePluginConfig,
    PluginType.SYSTEM_PROMPT.value: SystemPromptPluginConfig,
    PluginType.HEADER_MUTATION.value: HeaderMutationPluginConfig,
    PluginType.HALLUCINATION.value: HallucinationPluginConfig,
    PluginType.ROUTER_REPLAY.value: RouterReplayPluginConfig,
    PluginType.MEMORY.value: MemoryPluginConfig,
    PluginType.RAG.value: RAGPluginConfig,
}


def validate_plugin_configurations(config: UserConfig) -> list[ValidationError]:
    """Validate plugin configurations match their plugin types."""
    errors: list[ValidationError] = []

    for decision in config.decisions:
        for idx, plugin in enumerate(decision.plugins or []):
            plugin_type = _plugin_type_value(plugin.type)
            config_model = _PLUGIN_CONFIG_MODELS.get(plugin_type)
            if config_model is None:
                continue
            plugin_error = _validate_plugin_configuration(
                decision.name,
                idx,
                plugin_type,
                config_model,
                plugin.configuration,
            )
            if plugin_error is not None:
                errors.append(plugin_error)

    return errors


def _plugin_type_value(plugin_type) -> str:
    if hasattr(plugin_type, "value"):
        return plugin_type.value
    return str(plugin_type)


def _validate_plugin_configuration(
    decision_name: str,
    plugin_index: int,
    plugin_type: str,
    config_model,
    plugin_config: dict,
) -> ValidationError | None:
    try:
        config_model(**plugin_config)
    except PydanticValidationError as exc:
        details = ", ".join(_plugin_error_messages(exc))
        return ValidationError(
            f"Decision '{decision_name}' plugin #{plugin_index + 1} ({plugin_type}) "
            f"has invalid configuration: {details}",
            field=f"decisions.{decision_name}.plugins[{plugin_index}]",
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        return ValidationError(
            f"Decision '{decision_name}' plugin #{plugin_index + 1} ({plugin_type}) "
            f"configuration validation failed: {exc}",
            field=f"decisions.{decision_name}.plugins[{plugin_index}]",
        )
    return None


def _plugin_error_messages(exc: PydanticValidationError) -> list[str]:
    return [
        f"{' -> '.join(str(x) for x in error['loc'])}: {error['msg']}"
        for error in exc.errors()
    ]
