"""Configuration validator for vLLM Semantic Router."""

from typing import Dict, Any, List
from cli.models import (
    UserConfig,
    PluginType,
    SemanticCachePluginConfig,
    JailbreakPluginConfig,
    PIIPluginConfig,
    SystemPromptPluginConfig,
    HeaderMutationPluginConfig,
    HallucinationPluginConfig,
    RouterReplayPluginConfig,
)
from pydantic import ValidationError as PydanticValidationError
from cli.utils import getLogger
from cli.consts import EXTERNAL_API_MODEL_FORMATS

log = getLogger(__name__)


class ValidationError:
    """Validation error."""

    def __init__(self, message: str, field: str = None):
        self.message = message
        self.field = field

    def __str__(self):
        if self.field:
            return f"[{self.field}] {self.message}"
        return self.message


def validate_signal_references(config: UserConfig) -> List[ValidationError]:
    """
    Validate that all signal references in decisions exist.

    Args:
        config: User configuration

    Returns:
        list: List of validation errors
    """
    errors = []

    # Build signal name index
    signal_names = set()
    if config.signals:
        for signal in config.signals.keywords:
            signal_names.add(signal.name)
        for signal in config.signals.embeddings:
            signal_names.add(signal.name)
        if config.signals.fact_check:
            for signal in config.signals.fact_check:
                signal_names.add(signal.name)
        if config.signals.context:
            for signal in config.signals.context:
                signal_names.add(signal.name)

    # Check decision conditions
    for decision in config.decisions:
        for condition in decision.rules.conditions:
            if condition.type in ["keyword", "embedding", "fact_check", "context"]:
                if condition.name not in signal_names:
                    errors.append(
                        ValidationError(
                            f"Decision '{decision.name}' references unknown signal '{condition.name}'",
                            field=f"decisions.{decision.name}.rules.conditions",
                        )
                    )

    return errors


def validate_domain_references(config: UserConfig) -> List[ValidationError]:
    """
    Validate that all domain references in decisions exist.

    Args:
        config: User configuration

    Returns:
        list: List of validation errors
    """
    errors = []

    # Build domain name index
    domain_names = set()
    if config.signals and config.signals.domains:
        for domain in config.signals.domains:
            domain_names.add(domain.name)

    # If no domains defined, collect from decisions (will be auto-generated)
    if not domain_names:
        for decision in config.decisions:
            for condition in decision.rules.conditions:
                if condition.type == "domain":
                    domain_names.add(condition.name)

    # Check decision conditions
    for decision in config.decisions:
        for condition in decision.rules.conditions:
            if condition.type == "domain":
                if not domain_names:
                    errors.append(
                        ValidationError(
                            f"Decision '{decision.name}' references domain '{condition.name}' but no domains are defined",
                            field=f"decisions.{decision.name}.rules.conditions",
                        )
                    )

    return errors


def validate_model_references(config: UserConfig) -> List[ValidationError]:
    """
    Validate that all model references in decisions exist.

    Args:
        config: User configuration

    Returns:
        list: List of validation errors
    """
    errors = []

    # Build model name index
    model_names = {model.name for model in config.providers.models}

    # Check decision model references
    for decision in config.decisions:
        for model_ref in decision.modelRefs:
            if model_ref.model not in model_names:
                errors.append(
                    ValidationError(
                        f"Decision '{decision.name}' references unknown model '{model_ref.model}'",
                        field=f"decisions.{decision.name}.modelRefs",
                    )
                )

    # Check default model
    if config.providers.default_model not in model_names:
        errors.append(
            ValidationError(
                f"Default model '{config.providers.default_model}' not found in models",
                field="providers.default_model",
            )
        )

    return errors


def validate_merged_config(merged_config: Dict[str, Any]) -> List[ValidationError]:
    """
    Validate the merged router configuration.

    Args:
        merged_config: Merged configuration dictionary

    Returns:
        list: List of validation errors
    """
    errors = []

    # Validate required fields
    required_fields = [
        "vllm_endpoints",
        "model_config",
        "default_model",
        "decisions",
        "categories",
    ]
    for field in required_fields:
        if field not in merged_config:
            errors.append(
                ValidationError(f"Missing required field: {field}", field=field)
            )

    # Validate endpoints
    if "vllm_endpoints" in merged_config:
        endpoints = merged_config["vllm_endpoints"]

        # Check if all models use external API backends (no vLLM endpoints needed)
        all_external_api = False
        if "model_config" in merged_config and merged_config["model_config"]:
            all_external_api = all(
                model_cfg.get("api_format") in EXTERNAL_API_MODEL_FORMATS
                for model_cfg in merged_config["model_config"].values()
                if isinstance(model_cfg, dict)
            )

        if not endpoints and not all_external_api:
            errors.append(
                ValidationError("No vLLM endpoints configured", field="vllm_endpoints")
            )

        # Check for duplicate endpoint names
        endpoint_names = set()
        for endpoint in endpoints:
            if endpoint["name"] in endpoint_names:
                errors.append(
                    ValidationError(
                        f"Duplicate endpoint name: {endpoint['name']}",
                        field="vllm_endpoints",
                    )
                )
            endpoint_names.add(endpoint["name"])

    # Validate categories
    if "categories" in merged_config:
        categories = merged_config["categories"]
        if not categories:
            errors.append(
                ValidationError(
                    "No categories configured or auto-generated", field="categories"
                )
            )

    return errors


def validate_plugin_configurations(config: UserConfig) -> List[ValidationError]:
    """
    Validate plugin configurations match their plugin types.

    Args:
        config: User configuration

    Returns:
        list: List of validation errors
    """
    errors = []

    # Map plugin types to their configuration models
    config_models = {
        PluginType.SEMANTIC_CACHE.value: SemanticCachePluginConfig,
        PluginType.JAILBREAK.value: JailbreakPluginConfig,
        PluginType.PII.value: PIIPluginConfig,
        PluginType.SYSTEM_PROMPT.value: SystemPromptPluginConfig,
        PluginType.HEADER_MUTATION.value: HeaderMutationPluginConfig,
        PluginType.HALLUCINATION.value: HallucinationPluginConfig,
        PluginType.ROUTER_REPLAY.value: RouterReplayPluginConfig,
    }

    for decision in config.decisions:
        if not decision.plugins:
            continue

        for idx, plugin in enumerate(decision.plugins):
            # plugin.type is now a PluginType enum, get its string value
            plugin_type = (
                plugin.type.value if hasattr(plugin.type, "value") else str(plugin.type)
            )
            plugin_config = plugin.configuration

            # Get the appropriate config model for this plugin type
            config_model = config_models.get(plugin_type)
            if config_model:
                try:
                    # Validate configuration against the plugin-specific model
                    config_model(**plugin_config)
                except PydanticValidationError as e:
                    error_messages = []
                    for error in e.errors():
                        field = " -> ".join(str(x) for x in error["loc"])
                        msg = error["msg"]
                        error_messages.append(f"{field}: {msg}")
                    errors.append(
                        ValidationError(
                            f"Decision '{decision.name}' plugin #{idx + 1} ({plugin_type}) has invalid configuration: {', '.join(error_messages)}",
                            field=f"decisions.{decision.name}.plugins[{idx}]",
                        )
                    )
                except Exception as e:
                    errors.append(
                        ValidationError(
                            f"Decision '{decision.name}' plugin #{idx + 1} ({plugin_type}) configuration validation failed: {e}",
                            field=f"decisions.{decision.name}.plugins[{idx}]",
                        )
                    )

    return errors


def validate_user_config(config: UserConfig) -> List[ValidationError]:
    """
    Validate user configuration.

    Args:
        config: User configuration

    Returns:
        list: List of validation errors
    """
    log.info("Validating user configuration...")

    errors = []

    # Validate signal references
    errors.extend(validate_signal_references(config))

    # Validate domain references
    errors.extend(validate_domain_references(config))

    # Validate model references
    errors.extend(validate_model_references(config))

    # Validate plugin configurations
    errors.extend(validate_plugin_configurations(config))

    if errors:
        log.warning(f"Found {len(errors)} validation error(s)")
        for error in errors:
            log.warning(f"  • {error}")
    else:
        log.info("✓ Configuration validation passed")

    return errors


def print_validation_errors(errors: List[ValidationError]):
    """
    Print validation errors in a user-friendly format.

    Args:
        errors: List of validation errors
    """
    if not errors:
        return

    print("\n❌ Configuration validation failed:\n")
    for i, error in enumerate(errors, 1):
        print(f"{i}. {error}")
    print()
