"""Recipe-local decision signal-reference validation."""

from cli.config_contract import (
    CONDITION_TYPE_DOMAIN,
    CONDITION_TYPE_PROJECTION,
    build_projection_reference_index,
    build_signal_reference_index,
    is_signal_condition_type,
    iter_condition_leaves,
    iter_routing_profiles,
    signal_reference_exists,
)
from cli.models import UserConfig
from cli.validation_error import ValidationError


def validate_signal_references(config: UserConfig) -> list[ValidationError]:
    """Resolve every decision leaf inside its owning routing profile."""
    errors: list[ValidationError] = []
    for profile_name, routing in iter_routing_profiles(config):
        errors.extend(_validate_profile_signal_references(profile_name, routing))

    return errors


def _validate_profile_signal_references(profile_name, routing):
    errors: list[ValidationError] = []
    signal_names = build_signal_reference_index(routing.signals)
    projection_names = build_projection_reference_index(routing.projections)

    for decision in routing.decisions:
        field = (
            f"recipes.{profile_name}.routing.decisions."
            f"{decision.name}.rules.conditions"
        )
        decision_context = f"Decision '{decision.name}' in recipe '{profile_name}'"
        for condition in iter_condition_leaves(decision.rules.conditions):
            message = _condition_reference_error(
                condition,
                signal_names,
                projection_names,
            )
            if message:
                errors.append(
                    ValidationError(f"{decision_context} {message}", field=field)
                )

    return errors


def _condition_reference_error(condition, signal_names, projection_names):
    condition_type = (condition.type or "").strip().lower()
    if condition_type == CONDITION_TYPE_PROJECTION:
        if condition.name in projection_names:
            return None
        return f"references unknown projection '{condition.name}'"
    if condition_type == CONDITION_TYPE_DOMAIN:
        # Domains support auto-generated declarations and have a dedicated
        # validator that owns their reference contract.
        return None
    if not is_signal_condition_type(condition.type):
        return f"uses unsupported signal type '{condition.type}'"
    if signal_reference_exists(signal_names, condition.type, condition.name):
        return None
    return f"references unknown signal '{condition.name}'"
