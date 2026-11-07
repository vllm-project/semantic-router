"""Classifier conditions route on the labels their rule declares.

The Router validates the model that answers a classifier rule and the score
predicates that model supports.
"""

from cli.config_contract import iter_condition_leaves, iter_routing_profiles
from cli.models import UserConfig
from cli.validation_error import ValidationError


def validate_classifier_contracts(
    config: UserConfig,
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    for profile_name, routing in iter_routing_profiles(config):
        profile_field = (
            "routing"
            if profile_name == "default"
            else f"recipes.{profile_name}.routing"
        )
        rules = {rule.name: rule for rule in routing.signals.classifiers or []}
        for decision in routing.decisions:
            for condition in iter_condition_leaves(decision.rules.conditions):
                if (condition.type or "").strip().lower() != "classifier":
                    continue
                rule = rules.get(condition.name or "")
                if rule is None or condition.label in rule.labels:
                    continue
                errors.append(
                    ValidationError(
                        f"Decision '{decision.name}' classifier label '{condition.label}' is not declared by signal '{rule.name}'",
                        field=f"{profile_field}.decisions.{decision.name}.rules.conditions",
                    )
                )
    return errors
