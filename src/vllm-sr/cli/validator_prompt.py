"""Prompt-selection dependency validation."""

from cli.models import UserConfig
from cli.validation_error import ValidationError


def validate_prompt_dependencies(
    config: UserConfig,
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    model_by_name = {model.name: model for model in config.providers.models}
    model_cards = {model.name: model for model in config.routing.model_cards}
    auto_model_names = _auto_model_names(config)
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
            helper_card = model_cards.get(helper)
            errors.extend(
                _prompt_model_errors(
                    decision.name,
                    helper,
                    helper_model,
                    helper_card,
                    field_prefix,
                    auto_model_names,
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
    helper_card,
    field_prefix: str,
    auto_model_names: set[str],
) -> list[ValidationError]:
    field = f"{field_prefix}.{decision_name}.algorithm.prompt.model"
    if helper_model is None:
        return [
            ValidationError(
                f"Decision '{decision_name}' prompt helper model '{helper}' is not declared in providers.models",
                field=field,
            )
        ]
    if helper in auto_model_names:
        return [
            ValidationError(
                f"Decision '{decision_name}' prompt helper model must be a concrete non-auto provider model",
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
    if helper_card is not None and (
        (helper_card.modality or "").strip().lower()
        not in {"", "text", "ar", "both", "multimodal"}
    ):
        return [
            ValidationError(
                f"Decision '{decision_name}' prompt helper must use text modality",
                field=field,
            )
        ]
    return []


def _auto_model_names(config: UserConfig) -> set[str]:
    global_config = config.global_ if isinstance(config.global_, dict) else {}
    router = global_config.get("router")
    if not isinstance(router, dict):
        router = {}
    raw_names = router.get("auto_model_names")
    if not isinstance(raw_names, list):
        raw_names = []
    configured = {str(name).strip() for name in raw_names if str(name).strip()}
    if configured:
        return configured
    return {
        "vllm-sr/auto",
        "auto",
        str(router.get("auto_model_name") or "MoM").strip(),
    }
