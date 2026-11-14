"""Cross-resource checks for native System One execution.

Payload structure is validated by the generated Router schema. This owner only
checks the relationships the CLI must resolve before forwarding configuration;
the Router validates the deployments and bindings native execution uses.
"""

from cli.config_contract import iter_condition_leaves, iter_routing_profiles
from cli.durations import parse_duration
from cli.models import UserConfig
from cli.validation_error import ValidationError

_UNAVAILABLE_NATIVE_CONTEXT = {
    "authz",
    "metadata",
    "conversation",
    "reask",
    "user_feedback",
    "input_modality",
}


def _is_native(decision) -> bool:
    algorithm = decision.algorithm
    return (
        algorithm is not None
        and algorithm.type == "escalate"
        and bool((algorithm.escalate or {}).get("stages"))
    )


def validate_native_execution(config: UserConfig) -> list[ValidationError]:
    errors = []
    models = {model.name: model for model in config.providers.models}
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
        elif name in native and not model.backend_refs:
            errors.append(
                ValidationError(
                    "systemone requires deployment or backend_refs", field=field
                )
            )
    native_recipes = {
        entry.recipe for entry in config.entrypoints if entry.api == "systemone"
    }
    chat_recipes = {
        entry.recipe for entry in config.entrypoints if entry.api != "systemone"
    } | {config.default_recipe.name}
    calibrations = (
        {item["name"] for item in config.evaluation.calibrations}
        if config.evaluation
        else set()
    )
    for name, profile in iter_routing_profiles(config):
        field = f"recipes.{name}"
        if name in native_recipes and name in chat_recipes:
            errors.append(
                ValidationError(
                    "a native recipe cannot also be a Chat entrypoint", field=field
                )
            )
        has_native = any(_is_native(decision) for decision in profile.decisions)
        if name in native_recipes or has_native:
            errors.extend(_native_profile_errors(config, profile, field))
        for decision in profile.decisions:
            path = field + f".decisions.{decision.name}"
            if not _is_native(decision):
                if name in native_recipes:
                    errors.append(
                        ValidationError(
                            "System One requires an escalate algorithm with stages",
                            field=path,
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
                        "escalate stages run System One answers and require an isolated System One recipe",
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
                "native execution uses stages and algorithm.budget; Chat fallback must be disabled",
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
    for kind in references:
        if (kind or "").strip().lower() in _UNAVAILABLE_NATIVE_CONTEXT:
            errors.append(
                ValidationError(
                    f"signal {kind!r} does not yet support native request context and call accounting",
                    field=field,
                )
            )
    return errors


def _native_decision_errors(decision, models, calibrations, path):
    errors = []
    algorithm = decision.algorithm
    refs = [ref.model for ref in decision.modelRefs]
    try:
        if parse_duration(algorithm.budget["deadline"]) <= 0:
            raise ValueError("deadline must be positive")
    except ValueError as error:
        errors.append(
            ValidationError(str(error), field=path + ".algorithm.budget.deadline")
        )
    if algorithm.on_error not in (None, "", "fail"):
        errors.append(
            ValidationError(
                "System One escalate fails when no stage is accepted; on_error must be fail",
                field=path + ".algorithm.on_error",
            )
        )
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
    errors.extend(
        _native_stage_errors(algorithm.escalate["stages"], models, refs, path)
    )
    errors.extend(
        _native_quality_errors(algorithm.escalate.get("quality"), calibrations, path)
    )
    return errors


def _native_stage_errors(stages, models, refs, path):
    errors = []
    names = set()
    for index, stage in enumerate(stages):
        field = path + f".algorithm.escalate.stages.{index}"
        name = stage.get("name") or stage["model"]
        kind = stage.get("kind") or "native"
        if name == "abstain" or name in names or name != name.strip():
            errors.append(
                ValidationError(
                    "stage names must be distinct and non-empty without surrounding spaces",
                    field=field,
                )
            )
        names.add(name)
        if index == 0 and (kind != "native" or not stage.get("enabled", True)):
            errors.append(
                ValidationError("first stage must be enabled and native", field=field)
            )
        model = models.get(stage["model"])
        expected = "systemone" if kind == "native" else "chat_completions"
        if stage["model"] not in refs or model is None:
            errors.append(
                ValidationError(
                    "stage model must name a declared provider modelRef", field=field
                )
            )
        elif (model.api_format or "chat_completions") != expected:
            errors.append(
                ValidationError(f"stage requires api_format: {expected}", field=field)
            )
        if kind == "judge" and not stage.get("generation"):
            errors.append(
                ValidationError(
                    "judge requires generation.max_output_tokens", field=field
                )
            )
        if kind == "native" and (stage.get("generation") or stage.get("instructions")):
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
    return errors


def _native_quality_errors(quality, calibrations, path):
    field = path + ".algorithm.escalate.quality"
    if quality is None:
        return []
    if quality.get("type") == "calibrated":
        if (
            quality.get("calibration") not in calibrations
            or quality.get("loss") != "bundle_error"
            or "max_risk" not in quality
            or quality.get("accept") is not None
        ):
            return [
                ValidationError(
                    "calibrated quality requires a declared calibration, loss: bundle_error and max_risk without accept",
                    field=field,
                )
            ]
        return []
    if any(key in quality for key in ("calibration", "loss", "max_risk")):
        return [
            ValidationError(
                "uncalibrated quality cannot declare calibration, loss or max_risk",
                field=field,
            )
        ]
    return []
