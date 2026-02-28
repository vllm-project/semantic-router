"""Configuration validator for vLLM Semantic Router."""

from typing import Dict, Any, List
from cli.models import (
    UserConfig,
    PluginType,
    SemanticCachePluginConfig,
    JailbreakPluginConfig,
    PIIPluginConfig,
    SystemPromptPluginConfig,
    HeaderMutationPluginConfig,
    HallucinationPluginConfig,
    RouterReplayPluginConfig,
    MemoryPluginConfig,
    RAGPluginConfig,
)
from pydantic import ValidationError as PydanticValidationError
from cli.utils import getLogger
from cli.consts import EXTERNAL_API_MODEL_FORMATS

log = getLogger(__name__)


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
    return condition_type.strip().lower() == "latency"


def _is_latency_aware_algorithm(decision) -> bool:
    if not decision.algorithm:
        return False
    return (decision.algorithm.type or "").strip().lower() == "latency_aware"


def validate_latency_compatibility(config: UserConfig) -> List[ValidationError]:
    errors = []
    has_legacy_conditions = any(
        _is_latency_condition(condition.type)
        for decision in config.decisions
        for condition in decision.rules.conditions
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

    # Build signal name index
    signal_names = set()
    if config.signals:
        for signal in config.signals.keywords:
            signal_names.add(signal.name)
        for signal in config.signals.embeddings:
            signal_names.add(signal.name)
        if config.signals.domains:
            for signal in config.signals.domains:
                signal_names.add(signal.name)
        if config.signals.fact_check:
            for signal in config.signals.fact_check:
                signal_names.add(signal.name)
        if config.signals.user_feedbacks:
            for signal in config.signals.user_feedbacks:
                signal_names.add(signal.name)
        if config.signals.preferences:
            for signal in config.signals.preferences:
                signal_names.add(signal.name)
        if config.signals.language:
            for signal in config.signals.language:
                signal_names.add(signal.name)
        if config.signals.context:
            for signal in config.signals.context:
                signal_names.add(signal.name)
        if config.signals.complexity:
            for signal in config.signals.complexity:
                signal_names.add(signal.name)
        if config.signals.modality:
            for signal in config.signals.modality:
                signal_names.add(signal.name)
        if config.signals.role_bindings:
            for signal in config.signals.role_bindings:
                signal_names.add(signal.name)
        if config.signals.jailbreak:
            for signal in config.signals.jailbreak:
                signal_names.add(signal.name)
        if config.signals.pii:
            for signal in config.signals.pii:
                signal_names.add(signal.name)

    # Check decision conditions
    for decision in config.decisions:
        for condition in decision.rules.conditions:
            if condition.type in [
                "keyword",
                "embedding",
                "domain",
                "fact_check",
                "user_feedback",
                "preference",
                "language",
                "context",
                "complexity",
                "modality",
                "authz",
                "jailbreak",
                "pii",
            ]:
                # Support sub-level references like "math_problem:hard"
                # where "math_problem" is the defined signal and "hard" is a sub-level.
                ref_name = condition.name
                if ":" in ref_name:
                    ref_name = ref_name.split(":")[0]
                if ref_name not in signal_names:
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
            for condition in decision.rules.conditions:
                if condition.type == "domain":
                    domain_names.add(condition.name)

    # Check decision conditions
    for decision in config.decisions:
        for condition in decision.rules.conditions:
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

    # Build model name index
    model_names = {model.name for model in config.providers.models}

    # Check decision model references
    for decision in config.decisions:
        for model_ref in decision.modelRefs:
            if model_ref.model not in model_names:
                errors.append(
                    ValidationError(
                        f"Decision '{decision.name}' references unknown model '{model_ref.model}'",
                        field=f"decisions.{decision.name}.modelRefs",
                    )
                )

    # Check default model
    if config.providers.default_model not in model_names:
        errors.append(
            ValidationError(
                f"Default model '{config.providers.default_model}' not found in models",
                field="providers.default_model",
            )
        )

    return errors


def validate_merged_config(merged_config: Dict[str, Any]) -> List[ValidationError]:
    """
    Validate the merged router configuration.

    Args:
        merged_config: Merged configuration dictionary

    Returns:
        list: List of validation errors
    """
    errors = []

    # Validate required fields
    required_fields = [
        "vllm_endpoints",
        "model_config",
        "default_model",
        "decisions",
        "categories",
    ]
    for field in required_fields:
        if field not in merged_config:
            errors.append(
                ValidationError(f"Missing required field: {field}", field=field)
            )

    # Validate endpoints
    if "vllm_endpoints" in merged_config:
        endpoints = merged_config["vllm_endpoints"]

        # Check if all models use external API backends (no vLLM endpoints needed)
        all_external_api = False
        if "model_config" in merged_config and merged_config["model_config"]:
            all_external_api = all(
                model_cfg.get("api_format") in EXTERNAL_API_MODEL_FORMATS
                for model_cfg in merged_config["model_config"].values()
                if isinstance(model_cfg, dict)
            )

        if not endpoints and not all_external_api:
            errors.append(
                ValidationError("No vLLM endpoints configured", field="vllm_endpoints")
            )

        # Check for duplicate endpoint names
        endpoint_names = set()
        for endpoint in endpoints:
            if endpoint["name"] in endpoint_names:
                errors.append(
                    ValidationError(
                        f"Duplicate endpoint name: {endpoint['name']}",
                        field="vllm_endpoints",
                    )
                )
            endpoint_names.add(endpoint["name"])

    # Validate categories
    if "categories" in merged_config:
        categories = merged_config["categories"]
        if not categories:
            errors.append(
                ValidationError(
                    "No categories configured or auto-generated", field="categories"
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
        PluginType.JAILBREAK.value: JailbreakPluginConfig,
        PluginType.PII.value: PIIPluginConfig,
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
                for model_ref in decision.modelRefs:
                    # Find model config
                    model = next(
                        (
                            m
                            for m in config.providers.models
                            if m.name == model_ref.model
                        ),
                        None,
                    )
                    if model and not model.description:
                        errors.append(
                            ValidationError(
                                f"Decision '{decision.name}' uses router_dc with require_descriptions=true, "
                                f"but model '{model.name}' has no description",
                                field=f"providers.models.{model.name}.description",
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
        log.info("✓ Configuration validation passed")

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
