"""Configuration validator orchestration for vLLM Semantic Router."""

from cli.models import UserConfig
from cli.utils import getLogger
from cli.validator_algorithm_checks import (
    validate_algorithm_configurations,
    validate_algorithm_one_of,
    validate_latency_aware_algorithm_config,
    validate_latency_compatibility,
)
from cli.validator_common import ValidationError
from cli.validator_merged import validate_merged_config as _validate_merged_config
from cli.validator_plugin_checks import validate_plugin_configurations
from cli.validator_signal_checks import (
    validate_domain_references,
    validate_model_references,
    validate_signal_references,
)

log = getLogger(__name__)


def validate_user_config(config: UserConfig) -> list[ValidationError]:
    """Validate user configuration."""
    log.info("Validating user configuration...")

    errors: list[ValidationError] = []
    errors.extend(validate_signal_references(config))
    errors.extend(validate_latency_compatibility(config))
    errors.extend(validate_algorithm_one_of(config))
    errors.extend(validate_latency_aware_algorithm_config(config))
    errors.extend(validate_domain_references(config))
    errors.extend(validate_model_references(config))
    errors.extend(validate_plugin_configurations(config))
    errors.extend(validate_algorithm_configurations(config))

    if errors:
        log.warning(f"Found {len(errors)} validation error(s)")
        for error in errors:
            log.warning(f"  • {error}")
    else:
        log.info("Configuration validation passed")

    return errors


def validate_merged_config(merged_config: dict) -> list[ValidationError]:
    """Validate the merged router configuration."""
    return _validate_merged_config(merged_config)


def print_validation_errors(errors: list[ValidationError]) -> None:
    """Print validation errors in a user-friendly format."""
    if not errors:
        return

    print("\n❌ Configuration validation failed:\n")
    for i, error in enumerate(errors, 1):
        print(f"{i}. {error}")
    print()
