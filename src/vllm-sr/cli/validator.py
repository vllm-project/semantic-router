"""Configuration validator for vLLM Semantic Router."""

from typing import Any, List
from cli.config_contract import iter_routing_profiles
from cli.models import (
    UserConfig,
    PluginType,
    ResponseCachePluginConfig,
    FastResponsePluginConfig,
    RequestParamsPluginConfig,
    ResponseJailbreakPluginConfig,
    ToolsPluginConfig,
    ToolSelectionPluginConfig,
    SystemPromptPluginConfig,
    HeaderMutationPluginConfig,
    HallucinationPluginConfig,
    RouterReplayPluginConfig,
    MemoryPluginConfig,
    RAGPluginConfig,
)
from cli.terminal import echo, error as terminal_error
from pydantic import ValidationError as PydanticValidationError
from cli.utils import get_logger
from cli.validation_error import ValidationError
from cli.validator_classifier import validate_classifier_contracts
from cli.validator_latency import (
    validate_latency_aware_algorithm_config,
)
from cli.validator_prompt import validate_prompt_dependencies
from cli.validator_projection_embedding import (
    validate_embedding_modality_compatibility,
    validate_projection_score_dependencies,
)
from cli.validator_recipe_contracts import (
    validate_domain_references,
    validate_recipe_contracts,
)
from cli.validator_workflows import (
    validate_static_workflow_roles,
    validate_workflow_final_model,
)
from cli.validator_signal_references import validate_signal_references

log = get_logger(__name__)

EXPECTED_ALGORITHM_BLOCK_BY_TYPE = {
    "confidence": "confidence",
    "ratings": "ratings",
    "remom": "remom",
    "fusion": "fusion",
    "workflows": "workflows",
    "router_dc": "router_dc",
    "automix": "automix",
    "hybrid": "hybrid",
    "latency_aware": "latency_aware",
    "multi_factor": "multi_factor",
    "prompt": "prompt",
}

ALGORITHM_CONFIG_BLOCKS = (
    "confidence",
    "ratings",
    "remom",
    "fusion",
    "workflows",
    "router_dc",
    "automix",
    "hybrid",
    "latency_aware",
    "multi_factor",
    "prompt",
)


def _iter_profile_decisions(config: UserConfig):
    for _, routing in iter_routing_profiles(config):
        yield from routing.decisions


VALID_ALGORITHM_TYPES = {
    "confidence",
    "ratings",
    "remom",
    "fusion",
    "workflows",
    "static",
    "router_dc",
    "automix",
    "hybrid",
    "knn",
    "kmeans",
    "svm",
    "mlp",
    "multi_factor",
    "latency_aware",
    "prompt",
}

MIGRATED_LEARNING_ALGORITHM_TARGETS = {
    "elo": "global.router.learning.adaptation",
    "rl_driven": "global.router.learning.adaptation",
    "gmtrouter": "global.router.learning.adaptation",
    "bandit": "global.router.learning.adaptation",
    "personalization": "global.router.learning.adaptation",
}


def _routing_profiles(config: UserConfig):
    profiles = [("decisions", config.routing)]
    profiles.extend(
        (f"recipes.{recipe.name}.decisions", recipe.routing)
        for recipe in config.recipes
    )
    return profiles


def _all_decisions(config: UserConfig):
    for field_prefix, routing in _routing_profiles(config):
        for decision in routing.decisions:
            yield field_prefix, decision


def _iter_condition_nodes(conditions):
    """Depth-first traversal over recursive condition trees."""
    if not conditions:
        return
    for condition in conditions:
        yield condition
        if getattr(condition, "conditions", None):
            yield from _iter_condition_nodes(condition.conditions)


def _iter_merged_condition_nodes(conditions):
    """Depth-first traversal over merged router condition dicts."""
    if not conditions:
        return
    for condition in conditions:
        if not isinstance(condition, dict):
            continue
        yield condition
        if condition.get("conditions"):
            yield from _iter_merged_condition_nodes(condition["conditions"])


def configured_algorithm_blocks(algorithm: Any) -> List[str]:
    return [
        block_name
        for block_name in ALGORITHM_CONFIG_BLOCKS
        if getattr(algorithm, block_name) is not None
    ]


def validate_migrated_learning_algorithm(
    decision, normalized_type: str, field_prefix: str = "decisions"
):
    algorithm = decision.algorithm
    if (
        normalized_type == "session_aware"
        or getattr(algorithm, "session_aware", None) is not None
    ):
        return ValidationError(
            f"decision '{decision.name}' algorithm.type=session_aware is no longer supported; "
            "remove "
            "algorithm.type=session_aware and configure a normal base algorithm only "
            "when this decision needs one. Enable global.router.learning.protection "
            "for session or conversation protection.",
            field=f"{field_prefix}.{decision.name}.algorithm",
        )
    if normalized_type in MIGRATED_LEARNING_ALGORITHM_TARGETS:
        return ValidationError(
            f"decision '{decision.name}' algorithm.type={normalized_type} has moved to "
            f"{MIGRATED_LEARNING_ALGORITHM_TARGETS[normalized_type]}; remove the learning "
            "algorithm type and choose a request-time base algorithm only when needed",
            field=f"{field_prefix}.{decision.name}.algorithm",
        )
    return None


def validate_migrated_learning_blocks(
    decision, field_prefix: str = "decisions"
) -> List[ValidationError]:
    errors = []
    algorithm = decision.algorithm
    for block_name, target in MIGRATED_LEARNING_ALGORITHM_TARGETS.items():
        if getattr(algorithm, block_name, None) is not None:
            errors.append(
                ValidationError(
                    f"decision '{decision.name}' algorithm.{block_name} has moved to "
                    f"{target}",
                    field=f"{field_prefix}.{decision.name}.algorithm.{block_name}",
                )
            )
    return errors


def validate_algorithm_one_of(config: UserConfig) -> List[ValidationError]:
    errors = []

    for field_prefix, decision in _all_decisions(config):
        if decision.algorithm is None:
            continue

        algorithm = decision.algorithm
        configured_blocks = configured_algorithm_blocks(algorithm)

        display_type = (algorithm.type or "").strip() or "<empty>"
        normalized_type = (algorithm.type or "").strip().lower()

        migrated_error = validate_migrated_learning_algorithm(
            decision,
            normalized_type,
            field_prefix,
        )
        if migrated_error is not None:
            errors.append(migrated_error)
            continue

        migrated_block_errors = validate_migrated_learning_blocks(
            decision,
            field_prefix,
        )
        if migrated_block_errors:
            errors.extend(migrated_block_errors)
            continue

        if len(configured_blocks) > 1:
            errors.append(
                ValidationError(
                    f"decision '{decision.name}' algorithm.type={display_type} cannot be combined with multiple algorithm config blocks: "
                    f"{', '.join(configured_blocks)}",
                    field=f"{field_prefix}.{decision.name}.algorithm",
                )
            )
            continue

        expected_block = EXPECTED_ALGORITHM_BLOCK_BY_TYPE.get(normalized_type)
        if expected_block is None:
            if configured_blocks:
                errors.append(
                    ValidationError(
                        f"decision '{decision.name}' algorithm.type={display_type} cannot be used with algorithm.{configured_blocks[0]} configuration",
                        field=f"{field_prefix}.{decision.name}.algorithm.{configured_blocks[0]}",
                    )
                )
            continue

        if len(configured_blocks) == 1 and configured_blocks[0] != expected_block:
            errors.append(
                ValidationError(
                    f"decision '{decision.name}' algorithm.type={display_type} requires algorithm.{expected_block} configuration; "
                    f"found algorithm.{configured_blocks[0]}",
                    field=f"{field_prefix}.{decision.name}.algorithm.{configured_blocks[0]}",
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

    provider_model_names = {model.name for model in config.providers.models}
    routing_cards = {card.name: card for card in config.routing.model_cards}
    routing_model_names = set(routing_cards.keys())

    for model in config.providers.models:
        if model.name not in routing_model_names:
            errors.append(
                ValidationError(
                    f"Provider model '{model.name}' is missing from routing.modelCards",
                    field=f"providers.models.{model.name}",
                )
            )

    for model in config.providers.models:
        if (
            model.reasoning_family
            and model.reasoning_family not in config.providers.reasoning_families
        ):
            errors.append(
                ValidationError(
                    f"Provider model '{model.name}' references unknown reasoning family '{model.reasoning_family}'",
                    field=f"providers.models.{model.name}.reasoning_family",
                )
            )

    # Check decision model references
    for field_prefix, decision in _all_decisions(config):
        for model_ref in decision.modelRefs:
            if model_ref.model not in provider_model_names:
                errors.append(
                    ValidationError(
                        f"Decision '{decision.name}' references unknown model '{model_ref.model}'",
                        field=f"{field_prefix}.{decision.name}.modelRefs",
                    )
                )
                continue
            if model_ref.model not in routing_model_names:
                errors.append(
                    ValidationError(
                        f"Decision '{decision.name}' references model '{model_ref.model}' without a routing.modelCards entry",
                        field=f"{field_prefix}.{decision.name}.modelRefs",
                    )
                )
                continue
            if model_ref.lora_name:
                declared_loras = {
                    adapter.name
                    for adapter in (routing_cards[model_ref.model].loras or [])
                    if adapter.name
                }
                if not declared_loras:
                    errors.append(
                        ValidationError(
                            f"Decision '{decision.name}' references LoRA '{model_ref.lora_name}' for model '{model_ref.model}', "
                            "but routing.modelCards declares no loras for that model",
                            field=f"{field_prefix}.{decision.name}.modelRefs",
                        )
                    )
                elif model_ref.lora_name not in declared_loras:
                    errors.append(
                        ValidationError(
                            f"Decision '{decision.name}' references unknown LoRA '{model_ref.lora_name}' for model '{model_ref.model}'",
                            field=f"{field_prefix}.{decision.name}.modelRefs",
                        )
                    )

    # Check default model
    if config.providers.default_model not in provider_model_names:
        errors.append(
            ValidationError(
                f"Default model '{config.providers.default_model}' not found in models",
                field="providers.defaults.default_model",
            )
        )
    elif config.providers.default_model not in routing_model_names:
        errors.append(
            ValidationError(
                f"Default model '{config.providers.default_model}' not found in routing.modelCards",
                field="providers.defaults.default_model",
            )
        )

    return errors


def _collect_pydantic_error_messages(exc: PydanticValidationError) -> List[str]:
    messages: List[str] = []
    for error in exc.errors():
        field = " -> ".join(str(x) for x in error["loc"])
        messages.append(f"{field}: {error['msg']}")
    return messages


def _validate_single_plugin_configuration(
    decision_name: str,
    idx: int,
    plugin_type: str,
    plugin_config: dict,
    config_model: type | None,
    field_prefix: str = "decisions",
) -> List[ValidationError]:
    if config_model is None:
        return []
    field = f"{field_prefix}.{decision_name}.plugins[{idx}]"
    try:
        config_model(**plugin_config)
        return []
    except PydanticValidationError as exc:
        joined = ", ".join(_collect_pydantic_error_messages(exc))
        return [
            ValidationError(
                f"Decision '{decision_name}' plugin #{idx + 1} ({plugin_type}) has invalid configuration: {joined}",
                field=field,
            )
        ]
    except Exception as exc:
        return [
            ValidationError(
                f"Decision '{decision_name}' plugin #{idx + 1} ({plugin_type}) configuration validation failed: {exc}",
                field=field,
            )
        ]


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
        PluginType.RESPONSE_CACHE.value: ResponseCachePluginConfig,
        PluginType.FAST_RESPONSE.value: FastResponsePluginConfig,
        PluginType.REQUEST_PARAMS.value: RequestParamsPluginConfig,
        PluginType.RESPONSE_JAILBREAK.value: ResponseJailbreakPluginConfig,
        PluginType.SYSTEM_PROMPT.value: SystemPromptPluginConfig,
        PluginType.HEADER_MUTATION.value: HeaderMutationPluginConfig,
        PluginType.HALLUCINATION.value: HallucinationPluginConfig,
        PluginType.ROUTER_REPLAY.value: RouterReplayPluginConfig,
        PluginType.MEMORY.value: MemoryPluginConfig,
        PluginType.RAG.value: RAGPluginConfig,
        PluginType.TOOLS.value: ToolsPluginConfig,
        PluginType.TOOL_SELECTION.value: ToolSelectionPluginConfig,
    }

    for field_prefix, decision in _all_decisions(config):
        if not decision.plugins:
            continue

        for idx, plugin in enumerate(decision.plugins):
            plugin_type = (
                plugin.type.value if hasattr(plugin.type, "value") else str(plugin.type)
            )
            config_model = config_models.get(plugin_type)
            errors.extend(
                _validate_single_plugin_configuration(
                    decision.name,
                    idx,
                    plugin_type,
                    plugin.configuration,
                    config_model,
                    field_prefix,
                )
            )

    return errors


def _router_dc_missing_description_errors(
    decision, algo, config: UserConfig
) -> List[ValidationError]:
    if (
        algo.type != "router_dc"
        or not algo.router_dc
        or not algo.router_dc.require_descriptions
    ):
        return []
    errs: List[ValidationError] = []
    routing_cards = {card.name: card for card in config.routing.model_cards}
    for model_ref in decision.modelRefs:
        model_card = routing_cards.get(model_ref.model)
        if model_card is None or model_card.description:
            continue
        errs.append(
            ValidationError(
                f"Decision '{decision.name}' uses router_dc with require_descriptions=true, "
                f"but model '{model_ref.model}' has no description",
                field=f"routing.modelCards.{model_ref.model}.description",
            )
        )
    return errs


def _maybe_hybrid_weight_error(
    decision_name: str,
    algo_type: str,
    algo,
    field_prefix: str = "decisions",
) -> ValidationError | None:
    if algo_type != "hybrid" or not algo.hybrid:
        return None
    h = algo.hybrid
    # Per-weight non-negativity is enforced by the pydantic model (ge=0).
    # Weights are normalized at runtime, so they need not sum to 1.0 — but an
    # all-zero set leaves the selector with nothing to normalize, so reject it.
    total = (
        (0.3 if h.experience_weight is None else h.experience_weight)
        + (0.3 if h.router_dc_weight is None else h.router_dc_weight)
        + (0.2 if h.automix_weight is None else h.automix_weight)
        + (0.2 if h.cost_weight is None else h.cost_weight)
    )
    if total <= 0:
        return ValidationError(
            f"Decision '{decision_name}' hybrid weights are all zero; "
            "at least one weight must be positive",
            field=f"{field_prefix}.{decision_name}.algorithm.hybrid",
        )
    return None


def validate_algorithm_configurations(config: UserConfig) -> List[ValidationError]:
    """
    Validate algorithm configurations in decisions.

    Validates both looper algorithms (confidence, ratings, remom, fusion,
    workflows)
    and selection algorithms (static, router_dc, automix, hybrid,
    knn, kmeans, svm, mlp, multi_factor, latency_aware, prompt).

    Args:
        config: User configuration

    Returns:
        list: List of validation errors
    """
    errors = []

    for field_prefix, decision in _all_decisions(config):
        if not decision.algorithm:
            continue

        algo = decision.algorithm
        algo_type = algo.type

        # Validate algorithm type
        if algo_type not in VALID_ALGORITHM_TYPES:
            errors.append(
                ValidationError(
                    f"Decision '{decision.name}' has invalid algorithm type '{algo_type}'. "
                    f"Valid types: {', '.join(sorted(VALID_ALGORITHM_TYPES))}",
                    field=f"{field_prefix}.{decision.name}.algorithm.type",
                )
            )
            continue

        errors.extend(_router_dc_missing_description_errors(decision, algo, config))

        hybrid_err = _maybe_hybrid_weight_error(
            decision.name,
            algo_type,
            algo,
            field_prefix,
        )
        if hybrid_err is not None:
            errors.append(hybrid_err)

        workflows_cfg = getattr(algo, "workflows", None)
        if algo_type == "workflows" and workflows_cfg is not None:
            mode = workflows_cfg.mode or "static"
            planner = workflows_cfg.planner
            planner_model = (
                getattr(planner, "model", None) if planner is not None else None
            )
            if mode == "dynamic" and not planner_model:
                errors.append(
                    ValidationError(
                        f"Decision '{decision.name}' uses workflows mode=dynamic but does not set planner.model",
                        field=f"{field_prefix}.{decision.name}.algorithm.workflows.planner.model",
                    )
                )
            if mode == "dynamic" and workflows_cfg.roles:
                errors.append(
                    ValidationError(
                        f"Decision '{decision.name}' uses workflows mode=dynamic but also sets static roles",
                        field=f"{field_prefix}.{decision.name}.algorithm.workflows.roles",
                    )
                )
            errors.extend(
                validate_workflow_final_model(
                    decision,
                    workflows_cfg,
                    field_prefix,
                )
            )
            if mode == "static":
                errors.extend(
                    validate_static_workflow_roles(
                        decision,
                        workflows_cfg,
                        field_prefix,
                    )
                )

        remom_cfg = getattr(algo, "remom", None)
        if (
            algo_type == "remom"
            and remom_cfg is not None
            and remom_cfg.synthesis_model
            and remom_cfg.synthesis_model
            not in {model_ref.model for model_ref in decision.modelRefs}
        ):
            errors.append(
                ValidationError(
                    f"Decision '{decision.name}' ReMoM synthesis_model "
                    f"'{remom_cfg.synthesis_model}' is not present in modelRefs",
                    field=f"decisions.{decision.name}.algorithm.remom.synthesis_model",
                )
            )

    return errors


def validate_user_config(
    config: UserConfig, *, log_summary: bool = True
) -> List[ValidationError]:
    """
    Validate user configuration.

    Args:
        config: User configuration
        log_summary: Emit the human-readable validation summary. Machine-readable
            callers disable this so stdout remains a valid document.

    Returns:
        list: List of validation errors
    """
    if log_summary:
        log.info("Validating user configuration...")

    errors = []

    errors.extend(validate_recipe_contracts(config))

    # Validate signal references
    errors.extend(validate_signal_references(config))
    errors.extend(validate_algorithm_one_of(config))
    errors.extend(validate_latency_aware_algorithm_config(config))

    # Validate domain references
    errors.extend(validate_domain_references(config))

    # Validate model references
    errors.extend(validate_model_references(config))
    errors.extend(validate_classifier_contracts(config))

    # Validate plugin configurations
    errors.extend(validate_plugin_configurations(config))

    # Validate algorithm configurations
    errors.extend(validate_algorithm_configurations(config))
    errors.extend(validate_prompt_dependencies(config))

    # Validate projection score dependency ordering
    errors.extend(validate_projection_score_dependencies(config))

    # Validate embedding query_modality compatibility with embedding model
    errors.extend(validate_embedding_modality_compatibility(config))

    if errors and log_summary:
        log.warning(f"Found {len(errors)} validation error(s)")
        for error in errors:
            log.warning(f"  • {error}")
    elif log_summary:
        log.info("Configuration validation passed")

    return errors


def print_validation_errors(errors: List[ValidationError]):
    """
    Print validation errors in a user-friendly format.

    Args:
        errors: List of validation errors
    """
    if not errors:
        return

    terminal_error("Configuration validation failed")
    for i, validation_error in enumerate(errors, 1):
        echo(f"  {i}. {validation_error}", err=True)
