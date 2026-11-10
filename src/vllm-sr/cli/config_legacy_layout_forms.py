"""The layouts before config v0.3, which `vllm-sr config migrate` rewrites
in one run, and the router settings entrypoints and in-process Loopers
replaced.

The lines mirror src/semantic-router/pkg/config/config_forms_legacy_layout.go.
"""

from __future__ import annotations

from typing import Any

MIGRATE = "run `vllm-sr config migrate`"
PREVIOUS_RELEASE_LAYOUT = "its place in the v0.5 layout"
PRE_V03_VERSIONS = ("v0.1", "v0.2")
CURRENT_VERSION = "v0.5"
PRE_V03_GUARD_PLUGINS = ("pii", "jailbreak")
# The algorithm of the decisions each Looper algorithm's model names ran.
LOOPER_DECISION_ALGORITHMS = {"remom": "remom", "fusion": "fusion", "flow": "workflows"}
PRE_V03_TOP_LEVEL_KEYS = (
    "signals",
    "decisions",
    "keyword_rules",
    "embedding_rules",
    "categories",
    "fact_check_rules",
    "user_feedback_rules",
    "reask_rules",
    "preference_rules",
    "language_rules",
    "context_rules",
    "structure_rules",
    "complexity_rules",
    "modality_rules",
    "role_bindings",
    "jailbreak",
    "hallucination",
    "pii",
    "kb",
    "conversation",
    "events",
    "metadata",
    "classifiers",
    "input_modality",
    "jailbreak_mapping_path",
    "mom_registry",
    "config_source",
    "model_config",
    "vllm_endpoints",
    "provider_profiles",
    "default_model",
    "reasoning_families",
    "default_reasoning_effort",
    "evaluation_catalog",
    "auto_model_names",
    "strategy",
    "auto_model_name",
    "list_backend_models",
    "include_config_models_in_list",
    "clear_route_cache",
    "model_selection",
    "streamed_body_mode",
    "max_streamed_body_bytes",
    "streamed_body_timeout_sec",
    "response_api",
    "router_replay",
    "api",
    "observability",
    "authz",
    "ratelimit",
    "semantic_cache",
    "response_cache",
    "memory",
    "vector_store",
    "tools",
    "looper",
    "system_models",
    "external_models",
    "embedding_models",
    "prompt_compression",
    "modality_detector",
    "bert_model",
    "prompt_guard",
    "feedback_detector",
    "classifier",
    "hallucination_mitigation",
)
PRE_V03_GLOBAL_KEYS = ("runtime", "models", "modules")
PRE_V03_PROVIDER_KEYS = (
    "model_targets",
    "backends",
    "auth_profiles",
    "default_model",
    "reasoning_families",
    "default_reasoning_effort",
)
PRE_V03_PROVIDER_DEFAULT_KEYS = (
    "default_model",
    "reasoning_families",
    "default_reasoning_effort",
)
PRE_V03_PROVIDER_MODEL_KEYS = (
    "access",
    "endpoints",
    "access_key",
    "quality_score",
    "reasoning_family",
    "evaluations",
)
PRE_V03_MODEL_CARD_KEYS = ("evaluations", "quality_score")


def removed_previous_release_forms(data: dict[str, Any]) -> list[str]:
    """The one-line load errors of the layouts before config v0.3; a schema
    version before v0.3 refuses the document by itself."""

    version = data.get("version")
    if version in PRE_V03_VERSIONS:
        return [
            f"version: invalid_value: config schema {version} is removed; "
            f"{MIGRATE} (version: {CURRENT_VERSION})"
        ]
    lines = [
        _previous(
            key,
            "removed_section",
            f"top-level {key} belongs to the layout before config v0.3",
        )
        for key in data
        if key in PRE_V03_TOP_LEVEL_KEYS
    ]
    global_section = _mapping(data.get("global"))
    lines += [
        _previous(
            f"global.{key}",
            "removed_field",
            f"global.{key} belongs to the global layout before config v0.3",
        )
        for key in PRE_V03_GLOBAL_KEYS
        if key in global_section
    ]
    providers = _mapping(data.get("providers"))
    paths = [f"providers.{key}" for key in PRE_V03_PROVIDER_KEYS if key in providers]
    defaults = _mapping(providers.get("defaults"))
    paths += [
        f"providers.defaults.{key}"
        for key in PRE_V03_PROVIDER_DEFAULT_KEYS
        if key in defaults
    ]
    for index, model in enumerate(_items(providers.get("models"))):
        path = f"providers.models[{_label(model, index)}]"
        paths += [
            f"{path}.{key}" for key in PRE_V03_PROVIDER_MODEL_KEYS if key in model
        ]
        for ref_index, ref in enumerate(_items(model.get("backend_refs"))):
            if "type" in ref:
                paths.append(f"{path}.backend_refs[{_label(ref, ref_index)}].type")
    routing = _mapping(data.get("routing"))
    if "models" in routing:
        paths.append("routing.models")
    for index, card in enumerate(_items(routing.get("modelCards"))):
        path = f"routing.modelCards[{_label(card, index)}]"
        paths += [f"{path}.{key}" for key in PRE_V03_MODEL_CARD_KEYS if key in card]
    lines += [
        _previous(
            path,
            "removed_field",
            f"{path} belongs to the provider catalog before config v0.3",
        )
        for path in paths
    ]
    for index, decision in enumerate(_items(routing.get("decisions"))):
        path = f"routing.decisions[{_label(decision, index)}]"
        for plugin_index, plugin in enumerate(_items(decision.get("plugins"))):
            kind = plugin.get("type")
            if kind in PRE_V03_GUARD_PLUGINS:
                lines.append(
                    _line(
                        f"{path}.plugins[{_label(plugin, plugin_index)}]",
                        "removed_plugin_type",
                        f"plugin {kind} was removed",
                        f"nothing; write a {kind} rule, and a decision that "
                        "refuses the requests it matches",
                    )
                )
    return lines


def removed_legacy_router_forms(data: dict[str, Any]) -> list[str]:
    """The one-line load errors of the router settings entrypoints and
    in-process Loopers replaced."""

    global_section = _mapping(data.get("global"))
    router = _mapping(global_section.get("router"))
    lines = [
        _line(
            f"global.router.{key}",
            "removed_field",
            "the auto model names are removed",
            "the model_names of the default recipe's entrypoint",
        )
        for key in ("auto_model_names", "auto_model_name")
        if key in router
    ]
    if "include_config_models_in_list" in router:
        lines.append(
            _line(
                "global.router.include_config_models_in_list",
                "removed_field",
                "include_config_models_in_list is removed",
                "global.router.list_backend_models",
            )
        )
    looper = _mapping(_mapping(global_section.get("integrations")).get("looper"))
    if "endpoint" in looper:
        lines.append(
            _line(
                "global.integrations.looper.endpoint",
                "removed_field",
                "the Looper endpoint is removed",
                "nothing; the Router makes a Looper's model calls in process, "
                "through each model's providers.models[].backend_refs",
            )
        )
    for algorithm, decisions in LOOPER_DECISION_ALGORITHMS.items():
        if "model_names" in _mapping(looper.get(algorithm)):
            lines.append(
                _line(
                    f"global.integrations.looper.{algorithm}.model_names",
                    "removed_field",
                    "the model names of a Looper algorithm are removed",
                    "an entrypoint whose model_names name a recipe with a decision "
                    f"that runs the {decisions} algorithm",
                )
            )
    return lines


def _previous(path: str, code: str, what: str) -> str:
    return _line(path, code, what, PREVIOUS_RELEASE_LAYOUT)


def _line(path: str, code: str, what: str, becomes: str) -> str:
    return f"{path}: {code}: {what}; {MIGRATE} ({becomes})"


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _items(value: Any) -> list[dict[str, Any]]:
    return (
        [item for item in value if isinstance(item, dict)]
        if isinstance(value, list)
        else []
    )


def _label(item: dict[str, Any], index: int) -> str:
    name = item.get("name")
    return name if isinstance(name, str) and name else str(index)
