"""Translate v0.3 routing profiles into model-free Recipes and Entrypoints."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from cli.config_upgrade_v03_support import (
    MigrationContext,
    as_list,
    as_mapping,
    canonical_decimal,
    reject_unknown_fields,
)

_ROUTING_FIELDS = frozenset(
    {"modelCards", "signals", "projections", "decisions", "strategy"}
)
_RECIPE_FIELDS = frozenset({"name", "description", "routing"})
_ENTRYPOINT_FIELDS = frozenset({"model_names", "recipe"})
_MODEL_REF_FIELDS = frozenset(
    {
        "model",
        "use_reasoning",
        "reasoning_description",
        "reasoning_effort",
        "lora_name",
        "weight",
    }
)
_MODEL_SELECTION_PATHS = (
    ("candidateIterations", "*", "models"),
    ("algorithm", "remom", "synthesis_model"),
    ("algorithm", "fusion", "model"),
    ("algorithm", "fusion", "analysis_models"),
    ("algorithm", "fusion", "analysis_overrides"),
    ("algorithm", "workflows", "planner", "model"),
    ("algorithm", "workflows", "final", "model"),
    ("algorithm", "workflows", "roles", "*", "models"),
    ("algorithm", "prompt", "model"),
)


@dataclass(frozen=True)
class RoutingUpgradeResult:
    """Translated routing resources and source-derived global bootstrap."""

    recipes: list[dict[str, Any]]
    entrypoints: list[dict[str, Any]]
    global_config: dict[str, Any]
    reasoning_efforts: dict[str, set[str]]


@dataclass(frozen=True)
class _Profile:
    recipe: dict[str, Any]
    assignments: dict[str, dict[str, Any]]


def translate_v03_routing(
    source: dict[str, Any],
    default_model: str | None,
    context: MigrationContext,
) -> RoutingUpgradeResult:
    """Translate the default routing profile, named Recipes, and aliases."""

    global_config, auto_aliases, inherited_strategy = _translate_global(
        source.get("global"), context
    )
    reasoning_efforts: dict[str, set[str]] = {}
    profiles: dict[str, _Profile] = {}

    top_routing = as_mapping(source.get("routing"), "routing", context)
    reject_unknown_fields(top_routing, _ROUTING_FIELDS, "routing", context)
    source_entrypoints = as_list(source.get("entrypoints"), "entrypoints", context)
    references_default = any(
        isinstance(entrypoint, dict) and entrypoint.get("recipe") == "default"
        for entrypoint in source_entrypoints
    )
    top_has_behavior = any(
        top_routing.get(field) not in (None, "", [], {})
        for field in ("signals", "projections", "decisions", "strategy")
    )
    if top_has_behavior or references_default or auto_aliases or default_model:
        profiles["default"] = _translate_profile(
            name="default",
            description="Migrated default routing recipe.",
            routing=top_routing,
            path="routing",
            default_model=default_model,
            inherited_strategy=inherited_strategy,
            reasoning_efforts=reasoning_efforts,
            context=context,
        )

    for index, raw_recipe in enumerate(
        as_list(source.get("recipes"), "recipes", context)
    ):
        path = f"recipes[{index}]"
        recipe = as_mapping(raw_recipe, path, context)
        reject_unknown_fields(recipe, _RECIPE_FIELDS, path, context)
        name = _required_name(recipe.get("name"), f"{path}.name", context)
        if not name:
            continue
        if name in profiles:
            context.add(
                "duplicate_recipe",
                f"{path}.name",
                f"Recipe {name!r} collides with another routing profile",
                "give every Recipe a unique name",
            )
            continue
        routing = as_mapping(recipe.get("routing"), f"{path}.routing", context)
        reject_unknown_fields(
            routing, _ROUTING_FIELDS - {"modelCards"}, f"{path}.routing", context
        )
        profiles[name] = _translate_profile(
            name=name,
            description=recipe.get("description"),
            routing=routing,
            path=f"{path}.routing",
            default_model=default_model,
            inherited_strategy=inherited_strategy,
            reasoning_efforts=reasoning_efforts,
            context=context,
        )

    entrypoints, referenced_profiles = _translate_entrypoints(
        source_entrypoints,
        profiles,
        auto_aliases,
        context,
    )
    for name, profile in profiles.items():
        if name not in referenced_profiles and profile.assignments:
            context.add(
                "unbound_recipe_models",
                f"recipes.{name}",
                "v0.3 stores Model references inside this Recipe, but no Entrypoint references it",
                "add a v0.3 entrypoint for this Recipe or remove its modelRefs before migrating",
            )

    return RoutingUpgradeResult(
        recipes=[profile.recipe for profile in profiles.values()],
        entrypoints=entrypoints,
        global_config=global_config,
        reasoning_efforts=reasoning_efforts,
    )


def _translate_global(
    value: Any, context: MigrationContext
) -> tuple[dict[str, Any], list[str], str | None]:
    global_config = deepcopy(as_mapping(value, "global", context))
    auto_aliases: list[str] = []
    inherited_strategy: str | None = None

    if "modules" in global_config:
        context.add(
            "removed_global_modules",
            "global.modules",
            "the v0.3 global.modules layout has no v0.4 runtime field",
            "move each supported module into its documented global.model_catalog or service contract",
        )

    control_plane = global_config.get("control_plane")
    if isinstance(control_plane, dict) and control_plane.get("mode") == "managed":
        context.add(
            "managed_state_not_portable",
            "global.control_plane.mode",
            "the YAML converter cannot embed managed desired state in a v0.4 bootstrap",
            "export/import managed resources through the versioned Management API instead",
        )

    router_value = global_config.get("router")
    if router_value is not None:
        router = as_mapping(router_value, "global.router", context)
        if isinstance(router_value, dict):
            global_config["router"] = router

        auto_alias = router.pop("auto_model_name", None)
        if auto_alias not in (None, ""):
            if isinstance(auto_alias, str) and auto_alias.strip():
                auto_aliases.append(auto_alias.strip())
            else:
                context.add(
                    "invalid_auto_alias",
                    "global.router.auto_model_name",
                    "expected a non-empty string",
                    "use one request-facing model name",
                )
        raw_aliases = router.pop("auto_model_names", None)
        for index, alias in enumerate(
            as_list(raw_aliases, "global.router.auto_model_names", context)
        ):
            if isinstance(alias, str) and alias.strip():
                auto_aliases.append(alias.strip())
            else:
                context.add(
                    "invalid_auto_alias",
                    f"global.router.auto_model_names[{index}]",
                    "expected a non-empty string",
                    "use one request-facing model name",
                )

        raw_strategy = router.pop("strategy", None)
        if raw_strategy not in (None, ""):
            if isinstance(raw_strategy, str):
                inherited_strategy = raw_strategy.strip()
            else:
                context.add(
                    "invalid_strategy",
                    "global.router.strategy",
                    "expected a routing strategy string",
                    "use priority or confidence",
                )

        config_source = router.pop("config_source", None)
        if config_source not in (None, "", "file"):
            context.add(
                "unsupported_config_source",
                "global.router.config_source",
                f"source {config_source!r} cannot be represented by a standalone v0.4 manifest",
                "use vllm-sr serve --config for standalone YAML or the Management API for managed mode",
            )
        if "include_config_models_in_list" in router:
            router.pop("include_config_models_in_list", None)
            context.add(
                "removed_model_listing_policy",
                "global.router.include_config_models_in_list",
                "v0.4 discovery visibility is controlled by Entrypoints and access policy",
                "remove this field and grant the intended Entrypoints or Models explicitly",
            )
        if router.get("model_selection") not in (None, {}):
            router.pop("model_selection", None)
            context.add(
                "global_model_selection",
                "global.router.model_selection",
                "v0.3 global Model-selection policy cannot be copied into model-free Recipes",
                "choose an algorithm on each affected Recipe decision",
            )
        elif "model_selection" in router:
            router.pop("model_selection", None)

    integrations = global_config.get("integrations")
    if isinstance(integrations, dict):
        looper = integrations.get("looper")
        if isinstance(looper, dict):
            for family_name in ("remom", "fusion", "flow"):
                family = looper.get(family_name)
                if isinstance(family, dict) and "model_names" in family:
                    context.add(
                        "looper_model_binding",
                        f"global.integrations.looper.{family_name}.model_names",
                        "v0.4 keeps Model selection in Entrypoint assignments",
                        "remove the list and assign the required Models to the Recipe decision",
                    )

    return global_config, list(dict.fromkeys(auto_aliases)), inherited_strategy


def _translate_profile(
    *,
    name: str,
    description: Any,
    routing: dict[str, Any],
    path: str,
    default_model: str | None,
    inherited_strategy: str | None,
    reasoning_efforts: dict[str, set[str]],
    context: MigrationContext,
) -> _Profile:
    raw_decisions = as_list(routing.get("decisions"), f"{path}.decisions", context)
    if not raw_decisions and default_model:
        raw_decisions = [
            {
                "name": "Default",
                "description": "Route requests to the migrated default Model.",
                "priority": 0,
                "rules": {"operator": "AND", "conditions": []},
                "modelRefs": [{"model": default_model}],
            }
        ]
    elif not raw_decisions:
        context.add(
            "empty_recipe",
            f"{path}.decisions",
            f"Recipe {name!r} has no decision and no default Model",
            "add at least one decision with modelRefs before migrating",
        )

    decisions: list[dict[str, Any]] = []
    assignments: dict[str, dict[str, Any]] = {}
    for index, raw_decision in enumerate(raw_decisions):
        decision_path = f"{path}.decisions[{index}]"
        decision = as_mapping(raw_decision, decision_path, context)
        decision_name = _required_name(
            decision.get("name"), f"{decision_path}.name", context
        )
        if not decision_name:
            continue
        for model_path in _MODEL_SELECTION_PATHS:
            if _path_exists(decision, model_path):
                rendered = ".".join(item for item in model_path if item != "*")
                context.add(
                    "embedded_algorithm_model",
                    f"{decision_path}.{rendered}",
                    "this model selection cannot be represented inside a v0.4 Recipe",
                    "remove it and express the complete candidate set in the Entrypoint assignment",
                )

        refs = as_list(decision.get("modelRefs"), f"{decision_path}.modelRefs", context)
        if not refs and default_model:
            refs = [{"model": default_model}]
        assignment_models = _translate_model_refs(
            refs,
            decision_path,
            reasoning_efforts,
            context,
        )
        if not assignment_models:
            context.add(
                "empty_assignment",
                f"{decision_path}.modelRefs",
                f"decision {decision_name!r} has no candidate Model",
                "add modelRefs or configure providers.defaults.default_model",
            )
        assignments[decision_name] = {"models": assignment_models}
        decisions.append(
            {
                key: deepcopy(value)
                for key, value in decision.items()
                if key != "modelRefs"
            }
        )

    document: dict[str, Any] = {"decisions": decisions}
    for field_name in ("signals", "projections"):
        if routing.get(field_name) not in (None, {}):
            document[field_name] = deepcopy(routing[field_name])
    strategy = routing.get("strategy") or inherited_strategy
    if strategy:
        document["strategy"] = strategy
    recipe: dict[str, Any] = {"name": name, "document": document}
    if description not in (None, ""):
        recipe["description"] = description
    return _Profile(recipe=recipe, assignments=assignments)


def _translate_model_refs(
    refs: list[Any],
    decision_path: str,
    reasoning_efforts: dict[str, set[str]],
    context: MigrationContext,
) -> list[dict[str, Any]]:
    assignments: list[dict[str, Any]] = []
    for index, raw_ref in enumerate(refs):
        path = f"{decision_path}.modelRefs[{index}]"
        ref = as_mapping(raw_ref, path, context)
        reject_unknown_fields(ref, _MODEL_REF_FIELDS, path, context)
        model_name = _required_name(ref.get("model"), f"{path}.model", context)
        if not model_name:
            continue
        assignment: dict[str, Any] = {"model": model_name}
        weight = canonical_decimal(
            ref.get("weight", 1), f"{path}.weight", context, positive=True
        )
        if weight and weight != "1":
            assignment["weight"] = weight
        lora = _optional_trimmed(ref.get("lora_name"))
        if lora:
            assignment["lora"] = lora

        use_reasoning = ref.get("use_reasoning", False)
        effort = _optional_trimmed(ref.get("reasoning_effort"))
        reasoning_description = _optional_trimmed(ref.get("reasoning_description"))
        if use_reasoning is True:
            reasoning: dict[str, Any] = {"enabled": True}
            if effort:
                reasoning["effort"] = effort
                reasoning_efforts.setdefault(model_name, set()).add(effort)
            if reasoning_description:
                reasoning["description"] = reasoning_description
            assignment["reasoning"] = reasoning
        elif use_reasoning not in (False, None):
            context.add(
                "invalid_reasoning_flag",
                f"{path}.use_reasoning",
                "expected a boolean",
                "set use_reasoning to true or false",
            )
        elif effort or reasoning_description:
            context.add(
                "inconsistent_reasoning",
                path,
                "reasoning metadata is present while use_reasoning is false",
                "enable reasoning or remove reasoning_effort and reasoning_description",
            )
        assignments.append(assignment)
    return assignments


def _translate_entrypoints(
    source_entrypoints: list[Any],
    profiles: dict[str, _Profile],
    auto_aliases: list[str],
    context: MigrationContext,
) -> tuple[list[dict[str, Any]], set[str]]:
    entrypoints: list[dict[str, Any]] = []
    referenced_profiles: set[str] = set()
    default_entrypoint: dict[str, Any] | None = None

    for index, raw_entrypoint in enumerate(source_entrypoints):
        path = f"entrypoints[{index}]"
        entrypoint = as_mapping(raw_entrypoint, path, context)
        reject_unknown_fields(entrypoint, _ENTRYPOINT_FIELDS, path, context)
        recipe_name = _required_name(
            entrypoint.get("recipe"), f"{path}.recipe", context
        )
        raw_names = as_list(
            entrypoint.get("model_names"), f"{path}.model_names", context
        )
        names = []
        for name_index, value in enumerate(raw_names):
            name = _optional_trimmed(value)
            if name:
                names.append(name)
            else:
                context.add(
                    "invalid_entrypoint_name",
                    f"{path}.model_names[{name_index}]",
                    "expected a non-empty string",
                    "supply one request-facing name",
                )
        names = list(dict.fromkeys(names))
        if not names:
            context.add(
                "missing_entrypoint_name",
                f"{path}.model_names",
                "at least one request-facing name is required",
                "add one model name",
            )
            continue
        profile = profiles.get(recipe_name)
        if profile is None:
            context.add(
                "unknown_recipe",
                f"{path}.recipe",
                f"Entrypoint references unknown Recipe {recipe_name!r}",
                "reference default or one declared recipe",
            )
            continue
        translated: dict[str, Any] = {
            "name": names[0],
            "recipe": recipe_name,
            "assignments": deepcopy(profile.assignments),
        }
        if len(names) > 1:
            translated["aliases"] = names[1:]
        entrypoints.append(translated)
        referenced_profiles.add(recipe_name)
        if recipe_name == "default" and default_entrypoint is None:
            default_entrypoint = translated

    if "default" in profiles:
        if default_entrypoint is None:
            names = auto_aliases or ["vllm-sr/default"]
            default_entrypoint = {
                "name": names[0],
                "recipe": "default",
                "assignments": deepcopy(profiles["default"].assignments),
            }
            if len(names) > 1:
                default_entrypoint["aliases"] = names[1:]
            entrypoints.append(default_entrypoint)
            referenced_profiles.add("default")
        elif auto_aliases:
            current_names = {
                default_entrypoint["name"],
                *(default_entrypoint.get("aliases") or []),
            }
            merged_aliases = list(default_entrypoint.get("aliases") or [])
            merged_aliases.extend(
                alias for alias in auto_aliases if alias not in current_names
            )
            if merged_aliases:
                default_entrypoint["aliases"] = merged_aliases

    return entrypoints, referenced_profiles


def _path_exists(value: Any, path: tuple[str, ...]) -> bool:
    if not path:
        return True
    head, *tail = path
    if head == "*":
        return isinstance(value, list) and any(
            _path_exists(item, tuple(tail)) for item in value
        )
    return (
        isinstance(value, dict)
        and head in value
        and _path_exists(value[head], tuple(tail))
    )


def _required_name(value: Any, path: str, context: MigrationContext) -> str:
    name = _optional_trimmed(value)
    if not name:
        context.add(
            "missing_name",
            path,
            "a non-empty trimmed name is required",
            "supply a stable human-readable name",
        )
        return ""
    return name


def _optional_trimmed(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None
