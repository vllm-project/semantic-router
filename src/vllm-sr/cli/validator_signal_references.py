"""Recipe-local decision signal-reference validation."""

from cli.config_contract import (
    build_projection_reference_index,
    build_signal_reference_index,
    is_signal_condition_type,
    iter_routing_profiles,
    signal_reference_exists,
)
from cli.models import UserConfig
from cli.validation_error import ValidationError


def _iter_condition_nodes(conditions):
    if not conditions:
        return
    for condition in conditions:
        yield condition
        if getattr(condition, "conditions", None):
            yield from _iter_condition_nodes(condition.conditions)


def validate_signal_references(config: UserConfig) -> list[ValidationError]:
    """Resolve every decision leaf inside its owning routing profile."""
    errors: list[ValidationError] = []
    for profile_name, routing in iter_routing_profiles(config):
        signal_names = build_signal_reference_index(routing.signals)
        projection_names = build_projection_reference_index(routing.projections)

        for decision in routing.decisions:
            field = (
                f"recipes.{profile_name}.routing.decisions."
                f"{decision.name}.rules.conditions"
            )
            for condition in _iter_condition_nodes(decision.rules.conditions):
                condition_type = (condition.type or "").strip().lower()
                if condition_type == "projection":
                    if condition.name not in projection_names:
                        errors.append(
                            ValidationError(
                                f"Decision '{decision.name}' in recipe "
                                f"'{profile_name}' references unknown projection "
                                f"'{condition.name}'",
                                field=field,
                            )
                        )
                    continue
                if condition_type == "domain":
                    # Domain references also support auto-generated declarations
                    # and are validated by validator_recipe_contracts.
                    continue
                if not is_signal_condition_type(condition.type):
                    errors.append(
                        ValidationError(
                            f"Decision '{decision.name}' in recipe '{profile_name}' "
                            f"uses unsupported signal type '{condition.type}'",
                            field=field,
                        )
                    )
                    continue
                if not signal_reference_exists(
                    signal_names, condition.type, condition.name
                ):
                    errors.append(
                        ValidationError(
                            f"Decision '{decision.name}' in recipe '{profile_name}' "
                            f"references unknown signal '{condition.name}'",
                            field=field,
                        )
                    )

    return errors
