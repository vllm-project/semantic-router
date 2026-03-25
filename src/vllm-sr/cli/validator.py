"""Configuration validator for vLLM Semantic Router."""

from typing import Dict, Any, List
from cli.config_contract import (
    build_projection_reference_index,
    build_signal_reference_index,
    is_signal_condition_type,
    signal_reference_exists,
)
from cli.models import (
    UserConfig,
    PluginType,
    SemanticCachePluginConfig,
    FastResponsePluginConfig,
    SystemPromptPluginConfig,
    HeaderMutationPluginConfig,
    HallucinationPluginConfig,
    RouterReplayPluginConfig,
    MemoryPluginConfig,
    RAGPluginConfig,
)
from pydantic import ValidationError as PydanticValidationError
from cli.utils import get_logger
from cli.consts import EXTERNAL_API_MODEL_FORMATS

log = get_logger(__name__)


class ValidationError:
    """Validation error."""

    def __init__(self, message: str, field: str = None):
        self.message = message
        self.field = field

    def __str__(self):
        if self.field:
            return f"[{self.field}] {self.message}"
        return self.message


def _is_latency_condition(condition_type: str) -> bool:
    if not condition_type:
        return False
    return condition_type.strip().lower() == "latency"


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


def _is_latency_aware_algorithm(decision) -> bool:
    if not decision.algorithm:
        return False
    return (decision.algorithm.type or "").strip().lower() == "latency_aware"


def validate_latency_compatibility(config: UserConfig) -> List[ValidationError]:
    errors = []
    has_legacy_conditions = any(
        _is_latency_condition(condition.type)
        for decision in config.decisions
        for condition in _iter_condition_nodes(decision.rules.conditions)
    )

    if has_legacy_conditions:
        errors.append(
            ValidationError(
                "legacy latency config is no longer supported; use decision.algorithm.type=latency_aware and remove conditions.type=latency",
                field="decisions.rules.conditions",
            )
        )

    return errors


def validate_latency_aware_algorithm_config(
    config: UserConfig,
) -> List[ValidationError]:
    errors = []
    for decision in config.decisions:
        if not _is_latency_aware_algorithm(decision):
            continue
        latency_cfg = decision.algorithm.latency_aware
        if latency_cfg is None:
            errors.append(
                ValidationError(
                    f"decision '{decision.name}' requires algorithm.latency_aware when algorithm.type=latency_aware",
                    field=f"decisions.{decision.name}.algorithm.latency_aware",
                )
            )
            continue

        has_tpot = (
            latency_cfg.tpot_percentile is not None and latency_cfg.tpot_percentile > 0
        )
        has_ttft = (
            latency_cfg.ttft_percentile is not None and latency_cfg.ttft_percentile > 0
        )
        if not has_tpot and not has_ttft:
            errors.append(
                ValidationError(
                    f"decision '{decision.name}' must set tpot_percentile or ttft_percentile in algorithm.latency_aware",
                    field=f"decisions.{decision.name}.algorithm.latency_aware",
                )
            )
        if has_tpot and not (1 <= latency_cfg.tpot_percentile <= 100):
            errors.append(
                ValidationError(
                    f"decision '{decision.name}' algorithm.latency_aware.tpot_percentile must be between 1 and 100",
                    field=f"decisions.{decision.name}.algorithm.latency_aware.tpot_percentile",
                )
            )
        if has_ttft and not (1 <= latency_cfg.ttft_percentile <= 100):
            errors.append(
                ValidationError(
                    f"decision '{decision.name}' algorithm.latency_aware.ttft_percentile must be between 1 and 100",
                    field=f"decisions.{decision.name}.algorithm.latency_aware.ttft_percentile",
                )
            )
    return errors


def validate_algorithm_one_of(config: UserConfig) -> List[ValidationError]:
    errors = []

    expected_block_by_type = {
        "confidence": "confidence",
        "concurrent": "concurrent",
        "remom": "remom",
        "latency_aware": "latency_aware",
    }

    for decision in config.decisions:
        if decision.algorithm is None:
            continue

        algorithm = decision.algorithm
        configured_blocks = []
        if algorithm.confidence is not None:
            configured_blocks.append("confidence")
        if algorithm.concurrent is not None:
            configured_blocks.append("concurrent")
        if algorithm.remom is not None:
            configured_blocks.append("remom")
        if algorithm.latency_aware is not None:
            configured_blocks.append("latency_aware")

        display_type = (algorithm.type or "").strip() or "<empty>"
        normalized_type = (algorithm.type or "").strip().lower()

        if len(configured_blocks) > 1:
            errors.append(
                ValidationError(
                    f"decision '{decision.name}' algorithm.type={display_type} cannot be combined with multiple algorithm config blocks: "
                    f"{', '.join(configured_blocks)}",
                    field=f"decisions.{decision.name}.algorithm",
                )
            )
            continue

        expected_block = expected_block_by_type.get(normalized_type)
        if expected_block is None:
            if configured_blocks:
                errors.append(
                    ValidationError(
                        f"decision '{decision.name}' algorithm.type={display_type} cannot be used with algorithm.{configured_blocks[0]} configuration",
                        field=f"decisions.{decision.name}.algorithm.{configured_blocks[0]}",
                    )
                )
            continue

        if len(configured_blocks) == 1 and configured_blocks[0] != expected_block:
            errors.append(
                ValidationError(
                    f"decision '{decision.name}' algorithm.type={display_type} requires algorithm.{expected_block} configuration; "
                    f"found algorithm.{configured_blocks[0]}",
                    field=f"decisions.{decision.name}.algorithm.{configured_blocks[0]}",
                )
            )

    return errors


def validate_signal_references(config: UserConfig) -> List[ValidationError]:
    """
    Validate that all signal references in decisions exist.

    Args:
        config: User configuration

    Returns:
        list: List of validation errors
    """
    errors = []

    signal_names = build_signal_reference_index(config.signals)
    projection_names = build_projection_reference_index(config.routing.projections)

    # Check decision conditions
    for decision in config.decisions:
        for condition in _iter_condition_nodes(decision.rules.conditions):
            if (condition.type or "").strip().lower() == "projection":
                if condition.name in projection_names:
                    continue
                errors.append(
                    ValidationError(
                        f"Decision '{decision.name}' references unknown projection '{condition.name}'",
                        field=f"decisions.{decision.name}.rules.conditions",
                    )
                )
                continue
            if not is_signal_condition_type(condition.type):
                continue
            if signal_reference_exists(signal_names, condition.type, condition.name):
                continue
            errors.append(
                ValidationError(
                    f"Decision '{decision.name}' references unknown signal '{condition.name}'",
                    field=f"decisions.{decision.name}.rules.conditions",
                )
            )

    return errors


def validate_domain_references(config: UserConfig) -> List[ValidationError]:
    """
    Validate that all domain references in decisions exist.

    Args:
        config: User configuration

    Returns:
        list: List of validation errors
    """
    errors = []

    # Build domain name index
    domain_names = set()
    if config.signals and config.signals.domains:
        for domain in config.signals.domains:
            domain_names.add(domain.name)

    # If no domains defined, collect from decisions (will be auto-generated)
    if not domain_names:
        for decision in config.decisions:
            for condition in _iter_condition_nodes(decision.rules.conditions):
                if condition.type == "domain":
                    domain_names.add(condition.name)

    # Check decision conditions
    for decision in config.decisions:
        for condition in _iter_condition_nodes(decision.rules.conditions):
            if condition.type == "domain":
                if not domain_names:
                    errors.append(
                        ValidationError(
                            f"Decision '{decision.name}' references domain '{condition.name}' but no domains are defined",
                            field=f"decisions.{decision.name}.rules.conditions",
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
    for decision in config.decisions:
        for model_ref in decision.modelRefs:
            if model_ref.model not in provider_model_names:
                errors.append(
                    ValidationError(
                        f"Decision '{decision.name}' references unknown model '{model_ref.model}'",
                        field=f"decisions.{decision.name}.modelRefs",
                    )
                )
                continue
            if model_ref.model not in routing_model_names:
                errors.append(
                    ValidationError(
                        f"Decision '{decision.name}' references model '{model_ref.model}' without a routing.modelCards entry",
                        field=f"decisions.{decision.name}.modelRefs",
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
                            field=f"decisions.{decision.name}.modelRefs",
                        )
                    )
                elif model_ref.lora_name not in declared_loras:
                    errors.append(
                        ValidationError(
                            f"Decision '{decision.name}' references unknown LoRA '{model_ref.lora_name}' for model '{model_ref.model}'",
                            field=f"decisions.{decision.name}.modelRefs",
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
        PluginType.SEMANTIC_CACHE.value: SemanticCachePluginConfig,
        PluginType.FAST_RESPONSE.value: FastResponsePluginConfig,
        PluginType.SYSTEM_PROMPT.value: SystemPromptPluginConfig,
        PluginType.HEADER_MUTATION.value: HeaderMutationPluginConfig,
        PluginType.HALLUCINATION.value: HallucinationPluginConfig,
        PluginType.ROUTER_REPLAY.value: RouterReplayPluginConfig,
        PluginType.MEMORY.value: MemoryPluginConfig,
        PluginType.RAG.value: RAGPluginConfig,
    }

    for decision in config.decisions:
        if not decision.plugins:
            continue

        for idx, plugin in enumerate(decision.plugins):
            # plugin.type is now a PluginType enum, get its string value
            plugin_type = (
                plugin.type.value if hasattr(plugin.type, "value") else str(plugin.type)
            )
            plugin_config = plugin.configuration

            # Get the appropriate config model for this plugin type
            config_model = config_models.get(plugin_type)
            if config_model:
                try:
                    # Validate configuration against the plugin-specific model
                    config_model(**plugin_config)
                except PydanticValidationError as e:
                    error_messages = []
                    for error in e.errors():
                        field = " -> ".join(str(x) for x in error["loc"])
                        msg = error["msg"]
                        error_messages.append(f"{field}: {msg}")
                    errors.append(
                        ValidationError(
                            f"Decision '{decision.name}' plugin #{idx + 1} ({plugin_type}) has invalid configuration: {', '.join(error_messages)}",
                            field=f"decisions.{decision.name}.plugins[{idx}]",
                        )
                    )
                except Exception as e:
                    errors.append(
                        ValidationError(
                            f"Decision '{decision.name}' plugin #{idx + 1} ({plugin_type}) configuration validation failed: {e}",
                            field=f"decisions.{decision.name}.plugins[{idx}]",
                        )
                    )

    return errors


def validate_algorithm_configurations(config: UserConfig) -> List[ValidationError]:
    """
    Validate algorithm configurations in decisions.

    Validates both looper algorithms (confidence, concurrent, sequential, remom)
    and selection algorithms (static, elo, router_dc, automix, hybrid,
    latency_aware, thompson, gmtrouter, router_r1).

    Args:
        config: User configuration

    Returns:
        list: List of validation errors
    """
    errors = []

    # Valid algorithm types
    looper_types = {"confidence", "concurrent", "sequential", "remom"}
    selection_types = {
        "static",
        "elo",
        "router_dc",
        "automix",
        "hybrid",
        "latency_aware",
        "thompson",
        "gmtrouter",
        "router_r1",
    }
    all_types = looper_types | selection_types

    for decision in config.decisions:
        if not decision.algorithm:
            continue

        algo = decision.algorithm
        algo_type = algo.type

        # Validate algorithm type
        if algo_type not in all_types:
            errors.append(
                ValidationError(
                    f"Decision '{decision.name}' has invalid algorithm type '{algo_type}'. "
                    f"Valid types: {', '.join(sorted(all_types))}",
                    field=f"decisions.{decision.name}.algorithm.type",
                )
            )
            continue

        # Validate selection algorithm has corresponding config
        if algo_type == "elo" and algo.elo is None:
            # elo config is optional (uses defaults)
            pass
        if algo_type == "router_dc":
            # Warn if require_descriptions is true but models lack descriptions
            if algo.router_dc and algo.router_dc.require_descriptions:
                routing_cards = {card.name: card for card in config.routing.model_cards}
                for model_ref in decision.modelRefs:
                    model_card = routing_cards.get(model_ref.model)
                    if model_card is not None and not model_card.description:
                        errors.append(
                            ValidationError(
                                f"Decision '{decision.name}' uses router_dc with require_descriptions=true, "
                                f"but model '{model_ref.model}' has no description",
                                field=f"routing.modelCards.{model_ref.model}.description",
                            )
                        )

        # Validate hybrid weights sum to ~1.0 (with tolerance)
        # Note: Use `is None` check instead of `or` to handle 0.0 weights correctly
        if algo_type == "hybrid" and algo.hybrid:
            h = algo.hybrid
            total = (
                (0.3 if h.elo_weight is None else h.elo_weight)
                + (0.3 if h.router_dc_weight is None else h.router_dc_weight)
                + (0.2 if h.automix_weight is None else h.automix_weight)
                + (0.2 if h.cost_weight is None else h.cost_weight)
            )
            if abs(total - 1.0) > 0.01:
                errors.append(
                    ValidationError(
                        f"Decision '{decision.name}' hybrid weights sum to {total:.2f}, "
                        "should sum to 1.0",
                        field=f"decisions.{decision.name}.algorithm.hybrid",
                    )
                )

    return errors


def validate_user_config(config: UserConfig) -> List[ValidationError]:
    """
    Validate user configuration.

    Args:
        config: User configuration

    Returns:
        list: List of validation errors
    """
    log.info("Validating user configuration...")

    errors = []

    # Validate signal references
    errors.extend(validate_signal_references(config))
    errors.extend(validate_latency_compatibility(config))
    errors.extend(validate_algorithm_one_of(config))
    errors.extend(validate_latency_aware_algorithm_config(config))

    # Validate domain references
    errors.extend(validate_domain_references(config))

    # Validate model references
    errors.extend(validate_model_references(config))

    # Validate plugin configurations
    errors.extend(validate_plugin_configurations(config))

    # Validate algorithm configurations
    errors.extend(validate_algorithm_configurations(config))

    if errors:
        log.warning(f"Found {len(errors)} validation error(s)")
        for error in errors:
            log.warning(f"  • {error}")
    else:
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

    print("\n❌ Configuration validation failed:\n")
    for i, error in enumerate(errors, 1):
        print(f"{i}. {error}")
    print()
