"""Cross-object classifier signal validation."""

from cli.config_contract import iter_condition_leaves, iter_routing_profiles
from cli.models import UserConfig
from cli.validation_error import ValidationError

LOCAL_CLASSIFIER_MIN_CONFIDENCE = 0.5
MAX_NETWORK_PORT = 65535


def validate_classifier_contracts(
    config: UserConfig,
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    external_models = _external_models(config)
    runtime_signature: tuple[str, bool, tuple[str, ...]] | None = None
    for profile_name, routing in iter_routing_profiles(config):
        profile_field = f"recipes.{profile_name}.document"
        rules = {rule.name: rule for rule in routing.signals.classifiers or []}
        local_rules = [rule for rule in rules.values() if rule.type == "local"]
        if local_rules:
            signature = _local_classifier_signature(local_rules[0])
            if runtime_signature is None:
                runtime_signature = signature
            elif runtime_signature != signature:
                errors.append(
                    ValidationError(
                        "recipe-local classifiers must use identical model_path, labels, and use_cpu across the process",
                        field=f"{profile_field}.signals.classifiers",
                    )
                )
        errors.extend(
            _validate_profile_classifier_rules(
                rules,
                external_models,
                profile_field,
            )
        )
        errors.extend(
            _validate_profile_classifier_decisions(
                routing.decisions,
                rules,
                profile_field,
            )
        )
    return errors


def _local_classifier_signature(rule) -> tuple[str, bool, tuple[str, ...]]:
    return (
        rule.model_path or "",
        bool(rule.use_cpu),
        tuple(rule.labels),
    )


def _validate_profile_classifier_rules(
    rules: dict,
    external_models: dict[str, dict],
    profile_field: str,
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    local_rules = [rule for rule in rules.values() if rule.type == "local"]
    if len(local_rules) > 1:
        errors.append(
            ValidationError(
                "only one local classifier signal is supported per recipe",
                field=f"{profile_field}.signals.classifiers",
            )
        )
    for rule in rules.values():
        if rule.type != "llm":
            continue
        external = external_models.get(rule.model or "")
        field = f"{profile_field}.signals.classifiers.{rule.name}.model"
        if external is None:
            errors.append(
                ValidationError(
                    f"LLM classifier '{rule.name}' references unknown external model '{rule.model}'",
                    field=field,
                )
            )
        elif external.get("model_role") != "classification":
            errors.append(
                ValidationError(
                    f"LLM classifier '{rule.name}' model must have model_role=classification",
                    field=field,
                )
            )
        else:
            errors.extend(
                _external_model_endpoint_errors(
                    rule.name,
                    external,
                    field,
                )
            )
    return errors


def _validate_profile_classifier_decisions(
    decisions,
    rules: dict,
    profile_field: str,
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    for decision in decisions:
        for condition in iter_condition_leaves(decision.rules.conditions):
            if (condition.type or "").strip().lower() != "classifier":
                continue
            rule = rules.get(condition.name or "")
            if rule is None:
                continue
            field = f"{profile_field}.decisions.{decision.name}.rules.conditions"
            if condition.label not in rule.labels:
                errors.append(
                    ValidationError(
                        f"Decision '{decision.name}' classifier label '{condition.label}' is not declared by signal '{rule.name}'",
                        field=field,
                    )
                )
            if rule.type == "local" and not _valid_local_predicate(condition.predicate):
                errors.append(
                    ValidationError(
                        f"Decision '{decision.name}' local classifier condition supports only predicate.gte >= 0.5",
                        field=field,
                    )
                )
    return errors


def _external_models(config: UserConfig) -> dict[str, dict]:
    model_catalog = (config.global_ or {}).get("model_catalog") or {}
    external = model_catalog.get("external") or []
    return {
        item.get("name"): item
        for item in external
        if isinstance(item, dict) and item.get("name")
    }


def _external_model_endpoint_errors(
    rule_name: str,
    external: dict,
    field: str,
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    endpoint = external.get("llm_endpoint") or {}
    if not str(external.get("llm_model_name") or "").strip():
        errors.append(
            ValidationError(
                f"LLM classifier '{rule_name}' external model requires llm_model_name",
                field=field,
            )
        )
    address = str(endpoint.get("address") or "").strip()
    port = endpoint.get("port")
    if not address or not isinstance(port, int) or not 1 <= port <= MAX_NETWORK_PORT:
        errors.append(
            ValidationError(
                f"LLM classifier '{rule_name}' external model requires a valid llm_endpoint address and port",
                field=field,
            )
        )
    protocol = str(endpoint.get("protocol") or "").strip().lower()
    if protocol not in {"", "http", "https"}:
        errors.append(
            ValidationError(
                f"LLM classifier '{rule_name}' llm_endpoint.protocol must be http or https",
                field=field,
            )
        )
    return errors


def _valid_local_predicate(predicate) -> bool:
    return (
        predicate is not None
        and predicate.gte is not None
        and predicate.gte >= LOCAL_CLASSIFIER_MIN_CONFIDENCE
        and predicate.gt is None
        and predicate.lt is None
        and predicate.lte is None
    )
