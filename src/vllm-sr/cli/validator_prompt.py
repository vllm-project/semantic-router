"""Prompt-selection dependency validation."""

from cli.models import UserConfig
from cli.validation_error import ValidationError


def validate_prompt_dependencies(
    config: UserConfig,
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    model_by_name = {model.name: model for model in config.providers.models}
    looper_endpoint = _looper_endpoint(config.global_)
    profiles = [("decisions", config.routing)]
    profiles.extend(
        (f"recipes.{recipe.name}.decisions", recipe.routing)
        for recipe in config.recipes
    )
    for field_prefix, routing in profiles:
        for decision in routing.decisions:
            algorithm = decision.algorithm
            if algorithm is None or algorithm.type != "prompt":
                continue
            helper = algorithm.prompt.model if algorithm.prompt else ""
            helper_model = model_by_name.get(helper)
            errors.extend(
                _prompt_model_errors(
                    decision.name,
                    helper,
                    helper_model,
                    field_prefix,
                )
            )
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


def _prompt_model_errors(
    decision_name: str,
    helper: str,
    helper_model,
    field_prefix: str,
) -> list[ValidationError]:
    field = f"{field_prefix}.{decision_name}.algorithm.prompt.model"
    if helper_model is None:
        return [
            ValidationError(
                f"Decision '{decision_name}' prompt helper model '{helper}' is not declared in providers.models",
                field=field,
            )
        ]
    if (helper_model.api_format or "").strip().lower() == "anthropic":
        return [
            ValidationError(
                f"Decision '{decision_name}' prompt helper must use an OpenAI-compatible API format",
                field=field,
            )
        ]
    return []
