"""Configuration validator for vLLM Semantic Router."""

from typing import Dict, Any, List
from cli.models import UserConfig
from cli.utils import getLogger

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

    # Check decision conditions
    for decision in config.decisions:
        for condition in decision.rules.conditions:
            if condition.type in ["keyword", "embedding", "fact_check"]:
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
        if not endpoints:
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
