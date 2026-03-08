"""Signal, domain, and model reference validation helpers."""

from cli.models import UserConfig
from cli.validator_common import ValidationError, iter_condition_nodes

_DIRECT_SIGNAL_ATTRS = (
    "keywords",
    "embeddings",
    "domains",
    "fact_check",
    "user_feedbacks",
    "preferences",
    "language",
    "context",
    "jailbreak",
    "pii",
    "modality",
)
_REFERENCE_SIGNAL_TYPES = {
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
}


def validate_signal_references(config: UserConfig) -> list[ValidationError]:
    """Validate that all signal references in decisions exist."""
    signal_names = _collect_signal_names(config)
    errors: list[ValidationError] = []

    for decision in config.decisions:
        for condition in iter_condition_nodes(decision.rules.conditions):
            if condition.type not in _REFERENCE_SIGNAL_TYPES:
                continue
            if _signal_reference_exists(condition.type, condition.name, signal_names):
                continue
            errors.append(
                ValidationError(
                    f"Decision '{decision.name}' references unknown signal '{condition.name}'",
                    field=f"decisions.{decision.name}.rules.conditions",
                )
            )

    return errors


def validate_domain_references(config: UserConfig) -> list[ValidationError]:
    """Validate that all domain references in decisions exist."""
    domain_names = _collect_domain_names(config)
    if domain_names:
        return []

    errors: list[ValidationError] = []
    for decision in config.decisions:
        for condition in iter_condition_nodes(decision.rules.conditions):
            if condition.type != "domain":
                continue
            errors.append(
                ValidationError(
                    f"Decision '{decision.name}' references domain '{condition.name}' but no domains are defined",
                    field=f"decisions.{decision.name}.rules.conditions",
                )
            )

    return errors


def validate_model_references(config: UserConfig) -> list[ValidationError]:
    """Validate that all model references in decisions exist."""
    errors: list[ValidationError] = []
    model_names = {model.name for model in config.providers.models}

    for decision in config.decisions:
        for model_ref in decision.modelRefs:
            if model_ref.model in model_names:
                continue
            errors.append(
                ValidationError(
                    f"Decision '{decision.name}' references unknown model '{model_ref.model}'",
                    field=f"decisions.{decision.name}.modelRefs",
                )
            )

    if config.providers.default_model not in model_names:
        errors.append(
            ValidationError(
                f"Default model '{config.providers.default_model}' not found in models",
                field="providers.default_model",
            )
        )

    return errors


def _collect_signal_names(config: UserConfig) -> set[str]:
    signal_names: set[str] = set()
    signals = config.signals
    if not signals:
        return signal_names

    for attr_name in _DIRECT_SIGNAL_ATTRS:
        for signal in getattr(signals, attr_name) or []:
            signal_names.add(signal.name)

    for signal in signals.complexity or []:
        signal_names.update(
            {
                f"{signal.name}:easy",
                f"{signal.name}:medium",
                f"{signal.name}:hard",
            }
        )

    for signal in signals.role_bindings or []:
        signal_names.add(signal.role)

    return signal_names


def _signal_reference_exists(
    signal_type: str | None,
    signal_name: str | None,
    signal_names: set[str],
) -> bool:
    if not signal_name:
        return False
    if signal_type == "complexity":
        return signal_name in signal_names
    return _normalize_signal_reference(signal_name) in signal_names


def _normalize_signal_reference(signal_name: str) -> str:
    if ":" not in signal_name:
        return signal_name
    return signal_name.split(":", 1)[0]


def _collect_domain_names(config: UserConfig) -> set[str]:
    signals = config.signals
    explicit_domains = (
        {domain.name for domain in (signals.domains or [])} if signals else set()
    )
    if explicit_domains:
        return explicit_domains

    return {
        condition.name
        for decision in config.decisions
        for condition in iter_condition_nodes(decision.rules.conditions)
        if condition.type == "domain"
    }
