"""Configuration validator for vLLM Semantic Router."""

from typing import Any, List

from cli.config_contract import iter_routing_profiles
from cli.managed_envoy_contract import validate_envoy_dispatch_contract
from cli.models import (
    RecipeDistribution,
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
    ImageGenPluginConfig,
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
    "knn": "ml",
    "kmeans": "ml",
    "svm": "ml",
    "mlp": "ml",
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
    "ml",
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
    return [
        (f"recipes.{recipe.name}.document.decisions", recipe.document)
        for recipe in config.recipes
    ]


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


def _collect_pydantic_error_messages(exc: PydanticValidationError) -> List[str]:
    messages: List[str] = []
    for error in exc.errors():
        field = " -> ".join(str(x) for x in error["loc"])
        messages.append(f"{field}: {error['msg']}")
    return messages


def _validate_single_plugin_configuration(
    decision_name: str,
    *,
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
        PluginType.IMAGE_GEN.value: ImageGenPluginConfig,
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
                    idx=idx,
                    plugin_type=plugin_type,
                    plugin_config=plugin.configuration,
                    config_model=config_model,
                    field_prefix=field_prefix,
                )
            )

    return errors


def _router_dc_missing_description_errors(config: UserConfig) -> List[ValidationError]:
    models = {model.name: model for model in config.models}
    recipes = {recipe.name: recipe for recipe in config.recipes}
    errors: List[ValidationError] = []
    for entrypoint_index, entrypoint in enumerate(config.entrypoints):
        routes = (
            [
                (
                    entrypoint.recipe,
                    entrypoint.assignments or {},
                    f"entrypoints.{entrypoint_index}",
                )
            ]
            if not entrypoint.rules
            else [
                (
                    rule.recipe,
                    rule.assignments,
                    f"entrypoints.{entrypoint_index}.rules.{rule_index}",
                )
                for rule_index, rule in enumerate(entrypoint.rules)
            ]
        )
        for recipe_name, assignments, field in routes:
            recipe = recipes.get(recipe_name or "")
            if recipe is None:
                continue
            decisions = {
                decision.name: decision for decision in recipe.document.decisions
            }
            for decision_name, assignment in assignments.items():
                decision = decisions.get(decision_name)
                algorithm = decision.algorithm if decision is not None else None
                if (
                    algorithm is None
                    or algorithm.type != "router_dc"
                    or algorithm.router_dc is None
                    or not algorithm.router_dc.require_descriptions
                ):
                    continue
                for model_ref in assignment.models:
                    model = models.get(model_ref.model)
                    if model is None or model.card.description:
                        continue
                    errors.append(
                        ValidationError(
                            f"Decision '{decision.name}' uses router_dc with "
                            "require_descriptions=true, but assigned Model "
                            f"'{model_ref.model}' has no description",
                            field=f"{field}.assignments.{decision_name}",
                        )
                    )
    return errors


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
    errors = _router_dc_missing_description_errors(config)

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

        hybrid_err = _maybe_hybrid_weight_error(
            decision.name,
            algo_type,
            algo,
            field_prefix,
        )
        if hybrid_err is not None:
            errors.append(hybrid_err)

    return errors


def _validate_routing_contracts(config: UserConfig) -> List[ValidationError]:
    errors = []
    errors.extend(validate_recipe_contracts(config))
    errors.extend(validate_signal_references(config))
    errors.extend(validate_algorithm_one_of(config))
    errors.extend(validate_latency_aware_algorithm_config(config))
    errors.extend(validate_domain_references(config))
    errors.extend(validate_classifier_contracts(config))
    errors.extend(validate_plugin_configurations(config))
    errors.extend(validate_algorithm_configurations(config))
    errors.extend(validate_prompt_dependencies(config))
    errors.extend(validate_projection_score_dependencies(config))
    errors.extend(validate_embedding_modality_compatibility(config))
    return errors


def _report_validation_result(
    errors: List[ValidationError], *, log_summary: bool
) -> None:
    if not log_summary:
        return
    if errors:
        log.warning(f"Found {len(errors)} validation error(s)")
        for error in errors:
            log.warning(f"  • {error}")
    else:
        log.info("Configuration validation passed")


def validate_recipe_distribution(
    distribution: RecipeDistribution, *, log_summary: bool = True
) -> List[ValidationError]:
    """Validate routing semantics for a portable Recipe-only artifact."""

    if log_summary:
        log.info("Validating Recipe distribution...")
    validation_view = UserConfig.model_construct(
        version=distribution.version,
        listeners=[],
        models=[],
        entrypoints=[],
        recipes=distribution.recipes,
        global_=None,
    )
    errors = _validate_routing_contracts(validation_view)
    _report_validation_result(errors, log_summary=log_summary)
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

    errors = validate_envoy_dispatch_contract(config)
    errors.extend(_validate_routing_contracts(config))
    _report_validation_result(errors, log_summary=log_summary)
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
