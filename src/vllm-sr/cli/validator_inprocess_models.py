"""Backends for the models a decision calls in process.

The Router makes a Chat executor's model calls and context recovery's
itself, through each model's providers.models[].backend_refs,
in either gateway mode. It refuses at load a decision that calls a model with
no backend (pkg/config/validator_looper_backends.go); this mirrors that check
with the same message.
"""

import json
from collections.abc import Iterator
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from cli.config_contract import iter_routing_profiles
from cli.config_schema import routing_surface_catalog
from cli.models import RouterReplayPluginConfig, UserConfig
from cli.validation_error import ValidationError

EXECUTOR_ALGORITHM_TYPES = frozenset(
    surface["type"]
    for surface in routing_surface_catalog()["algorithms"]
    if surface.get("kind") == "executor"
)


def validate_inprocess_model_backends(config: UserConfig) -> list[ValidationError]:
    served = _served_models(config)
    errors: list[ValidationError] = []
    delegated = _delegated_models(config)
    global_replay = _layer_replay(
        None, (config.global_ or {}).get("plugins", []), "global", errors
    )
    errors.extend(_shadow_backend_errors(global_replay, served, delegated, "global"))
    for field_prefix, routing in _routing_profiles(config):
        recipe_replay = _layer_replay(
            global_replay,
            routing.plugins,
            field_prefix.removesuffix(".decisions"),
            errors,
        )
        errors.extend(
            _shadow_backend_errors(
                recipe_replay,
                served,
                delegated,
                field_prefix.removesuffix(".decisions"),
            )
        )
        for decision in routing.decisions:
            # Native decisions do not inherit Chat plugin defaults.
            native = (
                decision.algorithm is not None
                and decision.algorithm.type == "escalate"
                and not _executes_chat(decision.algorithm)
            )
            errors.extend(
                _shadow_backend_errors(
                    _layer_replay(
                        None if native else recipe_replay,
                        decision.plugins,
                        f"{field_prefix}.{decision.name}",
                        errors,
                    ),
                    served,
                    delegated,
                    f"{field_prefix}.{decision.name}",
                    decision,
                )
            )
            missing = [
                model for model in _called_models(decision) if model not in served
            ]
            for model in dict.fromkeys(missing):
                errors.append(
                    ValidationError(
                        f"decision {_quote(decision.name)}: the Router calls model "
                        f"{_quote(model)} in process, but it has no backend; "
                        "give it providers.models[].backend_refs",
                        field=f"{field_prefix}.{decision.name}",
                    )
                )
    return errors


def _layer_replay(
    base: Any, plugins: Any, owner: str, errors: list[ValidationError]
) -> Any:
    """Merge the replay plugin by field, as v0.5 global/recipe/decision layers do.

    Objects merge recursively; arrays and scalars replace, including an explicit
    shadow: null that clears the inherited observations. Never mutate authorship.
    Each layer may declare replay only once; duplicates must not hide invalid
    fields or model references in the first entry.
    """
    seen = False
    for index, plugin in enumerate(plugins or []):
        raw = plugin if isinstance(plugin, dict) else plugin.model_dump()
        if raw.get("type") == "router_replay":
            if seen:
                errors.append(
                    ValidationError(
                        "duplicate router_replay plugin in the same layer",
                        field=f"{owner}.plugins[{index}]",
                    )
                )
                continue
            seen = True
            override = raw.get("configuration")
            if override is not None:
                base = _merge_replay_fields(base, override)
    return base


def _merge_replay_fields(base: Any, override: Any) -> Any:
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override
    merged = dict(base)
    for key, value in override.items():
        merged[key] = _merge_replay_fields(base.get(key), value)
    return merged


def _shadow_backend_errors(
    configuration: Any,
    served: set[str],
    delegated: dict[str, str],
    owner: str,
    decision: Any = None,
) -> list[ValidationError]:
    if configuration is None:
        return []
    field = f"{owner}.plugins[router_replay].configuration"
    try:
        replay = RouterReplayPluginConfig.model_validate(configuration)
    except PydanticValidationError as exc:
        return [ValidationError(str(exc), field=field)]
    if replay.shadow is None:
        return []
    errors: list[ValidationError] = []
    if (
        decision is not None
        and decision.algorithm is not None
        and decision.algorithm.type in EXECUTOR_ALGORITHM_TYPES
    ):
        errors.append(
            ValidationError(
                f"decision {_quote(decision.name)}: router_replay shadow is not "
                f"supported on executor algorithm {_quote(decision.algorithm.type)}; "
                "the shadow runs only on single-model provider dispatch",
                field=f"{field}.shadow",
            )
        )
    # Disabled replay and sample_rate zero still validate declared backends.
    for model in replay.shadow.models:
        if model not in served:
            errors.append(
                ValidationError(
                    f"router_replay shadow model {_quote(model)} has no configured backend",
                    field=f"{field}.shadow.models",
                )
            )
        elif model in delegated:
            errors.append(
                ValidationError(
                    "auth.credential: delegated, but router_replay.shadow copies "
                    f"({owner}) call it as the Router; a Router-own call never "
                    "carries a user's credential, give it a static key",
                    field=delegated[model],
                )
            )
    return errors


def _delegated_models(config: UserConfig) -> dict[str, str]:
    delegated = {}
    for model in config.providers.models:
        for index, backend in enumerate(model.backend_refs or []):
            if backend.auth is not None and backend.auth.credential == "delegated":
                path = f"providers.models[{model.name}].backend_refs[{index}]"
                for name in [model.name, *(lora.name for lora in model.loras or [])]:
                    delegated[name] = path
                break
    return delegated


def _routing_profiles(config: UserConfig) -> list[tuple[str, Any]]:
    return [
        (f"recipes.{name}.decisions", recipe)
        for name, recipe in iter_routing_profiles(config)
    ]


def _served_models(config: UserConfig) -> set[str]:
    """Models with a backend, and the LoRA adapters their model cards declare."""
    served = {model.name for model in config.providers.models if model.backend_refs}
    for model in config.providers.models:
        if model.name in served:
            served.update(lora.name for lora in model.loras or [])
    return served


def _called_models(decision: Any) -> Iterator[str]:
    """The models a decision calls in process, beyond the one it routes to."""
    refs = [ref.model for ref in decision.modelRefs]
    algorithm = decision.algorithm
    if algorithm is not None and _executes_chat(algorithm):
        yield from refs
        fusion = algorithm.fusion
        if fusion is not None:
            yield from fusion.analysis_models or []
            if fusion.model:
                yield fusion.model
        planner = algorithm.workflows.planner if algorithm.workflows else None
        if planner is not None and (planner.model or "").strip():
            yield planner.model.strip()
    if _recovers_context(decision):
        yield from refs


def _recovers_context(decision: Any) -> bool:
    for plugin in decision.plugins or []:
        if str(getattr(plugin.type, "value", plugin.type)) != "context":
            continue
        configuration = plugin.configuration or {}
        compress = configuration.get("compress")
        recovery = compress.get("recovery") if isinstance(compress, dict) else None
        if (
            configuration.get("enabled")
            and isinstance(recovery, dict)
            and recovery.get("enabled")
        ):
            return True
    return False


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _executes_chat(algorithm: Any) -> bool:
    """A Chat executor; escalate's System One stages call native providers."""
    if algorithm.type not in EXECUTOR_ALGORITHM_TYPES:
        return False
    return not (
        algorithm.type == "escalate" and (algorithm.escalate or {}).get("stages")
    )
