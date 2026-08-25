"""Prompt-selection dependency validation."""

from cli.config_contract import iter_routing_profiles
from cli.models import UserConfig
from cli.validation_error import ValidationError


def validate_prompt_dependencies(
    config: UserConfig,
) -> list[ValidationError]:
    """Validate process dependencies without reintroducing a helper Model field.

    Prompt selection consumes the effective Model set from either an
    Entrypoint assignment or the Recipe's additive v0.3 ``modelRefs`` default.
    """

    errors: list[ValidationError] = []
    looper_endpoint = _looper_endpoint(config.global_)
    for profile_name, routing in iter_routing_profiles(config):
        field_prefix = (
            "routing.decisions"
            if profile_name == "default"
            else f"recipes.{profile_name}.routing.decisions"
        )
        for decision in routing.decisions:
            algorithm = decision.algorithm
            if algorithm is None or algorithm.type != "prompt":
                continue
            if not looper_endpoint:
                errors.append(
                    ValidationError(
                        f"Decision '{decision.name}' prompt selection requires global.integrations.looper.endpoint",
                        field=f"{field_prefix}.{decision.name}.algorithm.prompt",
                    )
                )
    return errors


def _looper_endpoint(global_config) -> str | None:
    if not isinstance(global_config, dict):
        return None
    integrations = global_config.get("integrations")
    if not isinstance(integrations, dict):
        return None
    looper = integrations.get("looper")
    if not isinstance(looper, dict):
        return None
    endpoint = looper.get("endpoint")
    return endpoint if isinstance(endpoint, str) else None
