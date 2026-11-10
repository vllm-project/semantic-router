"""Response cache, shadow and retention forms the Router no longer reads: the
response cache's plugin aliases and old store name, its flat threshold and
request-control fields (semantic.match.recall and request_controls), the
shadow_dispatch plugin (a shadow block in the decision's router_replay
plugin), and a decision's emits list (its retention attribute).

The Router refuses each one with a line that names `vllm-sr config migrate`,
which rewrites it; the CLI refuses them with the same lines.
"""

from __future__ import annotations

from typing import Any

from cli.config_recipe_scopes import recipe_scopes

MIGRATE = "run `vllm-sr config migrate`"

RESPONSE_CACHE = "response_cache"
RESPONSE_CACHE_ALIASES = ("semantic-cache", "semantic_cache", "response-cache")
FLAT_CACHE_FIELDS = ("similarity_threshold", "allow_request_controls", "control_header")


def removed_reuse_forms(data: dict[str, Any]) -> list[str]:
    """The one-line load errors of the removed cache, shadow and retention
    forms in data."""

    lines: list[str] = []
    global_section = data.get("global")
    stores = global_section.get("stores") if isinstance(global_section, dict) else None
    if isinstance(stores, dict) and "semantic_cache" in stores:
        lines.append(
            "global.stores.semantic_cache: removed_field: global.stores.semantic_cache "
            f"is removed; {MIGRATE} (global.stores.response_cache)"
        )
    for prefix, routing in recipe_scopes(data):
        decisions = routing.get("decisions")
        for index, decision in enumerate(
            decisions if isinstance(decisions, list) else []
        ):
            if not isinstance(decision, dict):
                continue
            name = decision.get("name")
            label = name if isinstance(name, str) and name else str(index)
            path = f"{prefix}.decisions[{label}]"
            plugins = decision.get("plugins")
            for position, plugin in enumerate(
                plugins if isinstance(plugins, list) else []
            ):
                if isinstance(plugin, dict):
                    lines.extend(
                        f"{path}.plugins[{position}]: {code}: {what}; {MIGRATE} ({becomes})"
                        for code, what, becomes in _plugin_forms(plugin)
                    )
            if "emits" in decision:
                lines.append(
                    f"{path}.emits: removed_field: emits is removed; {MIGRATE} "
                    "(the decision's retention attribute)"
                )
        if "data_policy" in routing:
            lines.append(
                f"{prefix}.data_policy: removed_field: a recipe's data_policy is "
                f"removed; {MIGRATE} (the recipe's router_replay plugin: enabled: "
                "false for replay: false, capture_personal_data: false for "
                "replay_personal_data: false)"
            )
    return lines


def _plugin_forms(plugin: dict[str, Any]) -> list[tuple[str, str, str]]:
    plugin_type = plugin.get("type")
    if plugin_type == "shadow_dispatch":
        return [
            (
                "removed_plugin_type",
                "plugin shadow_dispatch is removed",
                "a shadow block in the decision's router_replay plugin",
            )
        ]
    if plugin_type != RESPONSE_CACHE and plugin_type not in RESPONSE_CACHE_ALIASES:
        return []
    forms: list[tuple[str, str, str]] = []
    if plugin_type in RESPONSE_CACHE_ALIASES:
        forms.append(
            (
                "removed_plugin_type",
                f"plugin {plugin_type} is removed",
                f"type: {RESPONSE_CACHE}",
            )
        )
    configuration = plugin.get("configuration")
    if not isinstance(configuration, dict):
        return forms
    fields = [field for field in FLAT_CACHE_FIELDS if field in configuration]
    semantic = configuration.get("semantic")
    if isinstance(semantic, dict) and "similarity_threshold" in semantic:
        fields.append("semantic.similarity_threshold")
    if fields:
        what = (
            f"response_cache fields {', '.join(fields)} are removed"
            if len(fields) > 1
            else f"response_cache field {fields[0]} is removed"
        )
        forms.append(
            (
                "removed_field",
                what,
                "the similarity threshold is semantic.match.recall.min_similarity; "
                "allow_request_controls and control_header are "
                "request_controls.enabled and request_controls.header",
            )
        )
    return forms
