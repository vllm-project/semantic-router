"""Prompt-selection dependency validation."""

from cli.models import UserConfig
from cli.validation_error import ValidationError


def validate_prompt_dependencies(
    config: UserConfig,
) -> list[ValidationError]:
    """Validate process dependencies without reintroducing a helper Model field.

    v0.4 prompt selection consumes the Models assigned to the decision by its
    Entrypoint. Recipe documents therefore contain instructions and timeout
    only; assignment cardinality is validated with the Entrypoint contract.
    """

    errors: list[ValidationError] = []
    looper_endpoint = _looper_endpoint(config.global_)
    for recipe in config.recipes:
        field_prefix = f"recipes.{recipe.name}.document.decisions"
        for decision in recipe.document.decisions:
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
