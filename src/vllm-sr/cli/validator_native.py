"""Cross-resource checks for native System One execution.

Payload structure is validated by the generated Router schema. This owner only
checks the relationships the CLI must resolve before forwarding configuration.
"""

from cli.config_contract import iter_condition_leaves, iter_routing_profiles
from cli.decision_model import decision_model_deployment
from cli.durations import parse_duration
from cli.model_runtime_defaults import (
    effective_model_bindings,
    effective_model_deployments,
)
from cli.models import UserConfig
from cli.validation_error import ValidationError
from cli.validator_decision_model import public_model_name

NATIVE_ALGORITHMS = {"cascade", "policy"}
_UNAVAILABLE_NATIVE_CONTEXT = {
    "authz",
    "metadata",
    "conversation",
    "reask",
    "user_feedback",
    "input_modality",
    "kb",
    "kb_metric",
}


def _is_native(decision) -> bool:
    return (
        decision.algorithm is not None and decision.algorithm.type in NATIVE_ALGORITHMS
    )


def validate_native_execution(config: UserConfig) -> list[ValidationError]:
    errors = []
    models = {model.name: model for model in config.providers.models}
    deployments = effective_model_deployments(config)
    native = {name for name, model in models.items() if model.api_format == "systemone"}
    default = config.providers.defaults.model
    if default in native:
        errors.append(
            ValidationError(
                "providers.defaults.model must be a Chat provider",
                field="providers.defaults.model",
            )
        )
    for name, model in models.items():
        field = f"providers.models.{name}"
        if model.deployment:
            if (
                model.api_format != "systemone"
                or model.backend_refs
                or model.provider_model_id
                or model.external_model_ids
            ):
                errors.append(
                    ValidationError(
                        "deployment requires api_format: systemone without backend_refs or provider identity overrides",
                        field=field,
                    )
                )
            if deployments.get(model.deployment, {}).get("provider") != "model_runtime":
                errors.append(
                    ValidationError(
                        "deployment must reference a model_runtime deployment",
                        field=field + ".deployment",
                    )
                )
        elif name in native and not model.backend_refs:
            errors.append(
                ValidationError(
                    "systemone requires deployment or backend_refs", field=field
                )
            )
        if name in native:
            for key, deployment in deployments.items():
                if public_model_name(deployment) == name and model.deployment != key:
                    errors.append(
                        ValidationError(
                            "native alias conflicts with a different deployment's public model",
                            field=field,
                        )
                    )
    native_recipes = {
        entry.recipe for entry in config.entrypoints if entry.api == "systemone"
    }
    chat_recipes = {
        entry.recipe for entry in config.entrypoints if entry.api != "systemone"
    } | {"default"}
    calibrations = (
        {item["name"] for item in config.evaluation.calibrations}
        if config.evaluation
        else set()
    )
    for name, profile in iter_routing_profiles(config):
        field = "routing" if name == "default" else f"recipes.{name}.routing"
        if name in native_recipes and name in chat_recipes:
            errors.append(
                ValidationError(
                    "a native recipe cannot also be a Chat entrypoint", field=field
                )
            )
        has_native = any(_is_native(decision) for decision in profile.decisions)
        if name in native_recipes or has_native:
            errors.extend(_native_profile_errors(config, profile, field))
        if (name in native_recipes or has_native) and profile.budget is None:
            errors.append(
                ValidationError(
                    "native System One routing requires routing.budget", field=field
                )
            )
        if profile.budget is not None:
            if not has_native and name not in native_recipes:
                errors.append(
                    ValidationError(
                        "routing.budget is supported only for native System One recipes",
                        field=field,
                    )
                )
            try:
                if parse_duration(profile.budget["deadline"]) <= 0:
                    raise ValueError("deadline must be positive")
            except ValueError as error:
                errors.append(
                    ValidationError(str(error), field=field + ".budget.deadline")
                )
        for decision in profile.decisions:
            path = field + f".decisions.{decision.name}"
            if not _is_native(decision):
                if name in native_recipes:
                    errors.append(
                        ValidationError(
                            "System One requires cascade or policy", field=path
                        )
                    )
                if any(ref.model in native for ref in decision.modelRefs):
                    errors.append(
                        ValidationError(
                            "Chat execution cannot use native System One provider aliases",
                            field=path,
                        )
                    )
                continue
            if name in chat_recipes:
                errors.append(
                    ValidationError(
                        "native algorithms require an isolated System One recipe",
                        field=path,
                    )
                )
            if (
                decision.plugins
                or decision.action is not None
                or decision.fallback is not None
                or decision.reliability is not None
                or decision.output_contract
                or decision.output_contract_spec is not None
                or (
                    decision.adaptations is not None
                    and decision.adaptations.model_dump(exclude_none=True)
                )
            ):
                errors.append(
                    ValidationError(
                        "Chat plugins, actions, fallback, reliability, output contracts and adaptations are unsupported for native execution",
                        field=path,
                    )
                )
            errors.extend(_native_decision_errors(decision, models, calibrations, path))
    return errors


def _native_profile_errors(config, profile, field):
    errors = []
    if profile.candidate_requirements is not None:
        errors.append(
            ValidationError(
                "native execution does not support candidate_requirements",
                field=field + ".candidate_requirements",
            )
        )
    router = (config.global_ or {}).get("router") or {}
    fallback = router.get("fallback") if isinstance(router, dict) else None
    enabled = fallback.get("enabled", False) if isinstance(fallback, dict) else False
    if profile.fallback is not None and profile.fallback.enabled is not None:
        enabled = profile.fallback.enabled
    if enabled:
        errors.append(
            ValidationError(
                "native execution uses stages and routing.budget; Chat routing.fallback must be disabled",
                field=field + ".fallback",
            )
        )
    references = [
        leaf.type
        for decision in profile.decisions
        for leaf in iter_condition_leaves(decision.rules.conditions)
    ]
    references.extend(
        item.type for score in profile.projections.scores or [] for item in score.inputs
    )
    errors.extend(
        _native_signal_backend_errors(config, profile, set(references), field)
    )
    for kind in references:
        if (kind or "").strip().lower() in _UNAVAILABLE_NATIVE_CONTEXT:
            errors.append(
                ValidationError(
                    f"signal {kind!r} does not yet support native request context and call accounting",
                    field=field,
                )
            )
    return errors


def _native_signal_backend_errors(config, profile, references, field):
    catalog = (config.global_ or {}).get("model_catalog") or {}
    classifier = (catalog.get("modules") or {}).get("classifier") or {}
    errors = []
    if "domain" in references and (classifier.get("mcp") or {}).get("enabled"):
        errors.append(
            ValidationError(
                "native domain signals do not support the MCP classifier; use a model deployment binding",
                field=field,
            )
        )
    if "preference" not in references:
        return errors
    binding = effective_model_bindings(config, profile).get("preference")
    if binding is None or binding.contract != "decision.v1":
        preference = classifier.get("preference") or {}
        external = any(
            item.get("model_role") == "preference"
            for item in catalog.get("external") or []
        )
        prototypes = preference.get("use_contrastive")
        if prototypes is None:
            prototypes = not external and (
                preference.get("embedding_model")
                or any(rule.examples for rule in profile.signals.preferences or [])
            )
        if prototypes or external:
            errors.append(
                ValidationError(
                    "native preference requires a decision task deployment; contrastive and external preference adapters do not support native request accounting",
                    field=field,
                )
            )
            return errors
    if binding is not None:
        deployment = effective_model_deployments(config).get(binding.deployment) or {}
        if (
            deployment.get("provider") != "model_runtime"
            or binding.head
            or binding.operating_point is not None
        ):
            errors.append(
                ValidationError(
                    "native preference requires a model_runtime decision task binding without a classifier head or operating point",
                    field=field,
                )
            )
    else:
        try:
            decision_model_deployment({"global": config.global_ or {}})
        except ValueError as error:
            errors.append(ValidationError(str(error), field=field))
    return errors


def _native_decision_errors(decision, models, calibrations, path):
    errors = []
    algorithm = decision.algorithm
    refs = [ref.model for ref in decision.modelRefs]
    if any(
        ref.weight not in (None, 0)
        or ref.use_reasoning is not None
        or ref.reasoning_mode
        or ref.reasoning_effort
        or ref.reasoning_description
        for ref in decision.modelRefs
    ):
        errors.append(
            ValidationError(
                "native modelRefs declare model aliases only; weighting and Chat reasoning controls are unsupported",
                field=path + ".modelRefs",
            )
        )
    if (
        not refs
        or len(set(refs)) != len(refs)
        or any(ref.lora_name for ref in decision.modelRefs)
    ):
        errors.append(
            ValidationError(
                "native modelRefs must contain distinct concrete aliases without LoRA overrides",
                field=path + ".modelRefs",
            )
        )
    names = set()
    for index, stage in enumerate(algorithm.stages or []):
        field = path + f".algorithm.stages.{index}"
        if (
            algorithm.type == "policy"
            and stage["kind"] == "judge"
            and index != len(algorithm.stages) - 1
        ):
            errors.append(
                ValidationError(
                    "policy permits one terminal judge after all native stages",
                    field=field,
                )
            )
        if (
            not stage["name"].strip()
            or stage["name"] == "abstain"
            or stage["name"] in names
            or stage["name"] != stage["name"].strip()
        ):
            errors.append(
                ValidationError(
                    "stage names must be distinct and non-empty without surrounding spaces",
                    field=field,
                )
            )
        names.add(stage["name"])
        if index == 0 and (stage["kind"] != "native" or not stage.get("enabled", True)):
            errors.append(
                ValidationError("first stage must be enabled and native", field=field)
            )
        model = models.get(stage["model"])
        expected = "systemone" if stage["kind"] == "native" else "openai"
        if stage["model"] not in refs or model is None:
            errors.append(
                ValidationError(
                    "stage model must name a declared provider modelRef", field=field
                )
            )
        elif (model.api_format or "openai") != expected:
            errors.append(
                ValidationError(f"stage requires api_format: {expected}", field=field)
            )
        if stage["kind"] == "judge" and not stage.get("generation"):
            errors.append(
                ValidationError(
                    "judge requires generation.max_output_tokens", field=field
                )
            )
        if stage["kind"] == "native" and (
            stage.get("generation") or stage.get("instructions")
        ):
            errors.append(
                ValidationError(
                    "native stage cannot declare LLM generation or instructions",
                    field=field,
                )
            )
        if stage.get("timeout"):
            try:
                if parse_duration(stage["timeout"]) <= 0:
                    raise ValueError("stage timeout must be positive")
            except ValueError as error:
                errors.append(ValidationError(str(error), field=field))
    quality = algorithm.quality
    if quality["type"] == "calibrated":
        if (
            quality.get("calibration") not in calibrations
            or quality.get("loss") != "bundle_error"
            or "max_risk" not in quality
            or quality.get("acceptance") is not None
        ):
            errors.append(
                ValidationError(
                    "calibrated quality requires a declared calibration, loss: bundle_error and max_risk without acceptance",
                    field=path + ".algorithm.quality",
                )
            )
    elif not quality.get("acceptance") or any(
        key in quality for key in ("calibration", "loss", "max_risk")
    ):
        errors.append(
            ValidationError(
                "uncalibrated quality requires acceptance without calibration, loss or max_risk",
                field=path + ".algorithm.quality",
            )
        )
    return errors
