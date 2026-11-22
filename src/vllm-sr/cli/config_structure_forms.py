"""The v0.3 layout of a configuration and the store and service switches the
Router no longer reads: schema v0.3, the top-level routing block and its
model cards, each recipe's routing block, the response cache store's enabled
and similarity_threshold, the memory store's enabled, the replay service's
enablement and capture fields, and the model catalog's prompt compression,
now compress.signals of the context plugin in global.plugins.

The Router refuses each one with a line that names `vllm-sr config migrate`,
which rewrites it; the CLI refuses them with the same lines. A v0.3 document
is refused by its version alone.
"""

from __future__ import annotations

from typing import Any

from cli.config_recipe_scopes import recipe_scopes

MIGRATE = "run `vllm-sr config migrate`"
PREVIOUS_VERSION = "v0.3"
CURRENT_VERSION = "v0.5"

RESPONSE_CACHE_SWITCHES = ("enabled", "similarity_threshold")
ROUTER_REPLAY_PLUGIN_FIELDS = (
    "enabled",
    "capture_request_body",
    "capture_response_body",
    "capture_personal_data",
    "capture_tool_results",
    "max_records",
    "max_body_bytes",
    "max_tool_trace_bytes",
    "max_tool_trace_steps",
)


def removed_structure_forms(data: dict[str, Any]) -> list[str]:
    """The one-line load errors of the v0.3 layout and the removed switches
    in data."""

    if data.get("version") == PREVIOUS_VERSION:
        return [
            f"version: invalid_value: config schema {PREVIOUS_VERSION} is removed; "
            f"{MIGRATE} (version: {CURRENT_VERSION})"
        ]
    lines: list[str] = []
    routing = data.get("routing")
    if isinstance(routing, dict) and "modelCards" in routing:
        lines.append(
            "routing.modelCards: removed_section: routing.modelCards is removed; "
            f"{MIGRATE} (each card's fields on the providers.models[] entry of the same name)"
        )
    if "routing" in data:
        becomes = "recipes: [{name: default, ...}]"
        if isinstance(data.get("recipes"), list) and data["recipes"]:
            becomes += ", with an entrypoint marked default: true"
        lines.append(
            "routing: removed_section: the top-level routing block is removed; "
            f"{MIGRATE} ({becomes})"
        )
    for path, recipe in recipe_scopes(data):
        if "routing" in recipe:
            lines.append(
                f"{path}.routing: removed_field: recipes[].routing is removed; {MIGRATE} "
                "(its fields on the recipe: recipes[].signals, recipes[].decisions, ...)"
            )
    global_section = data.get("global")
    global_section = global_section if isinstance(global_section, dict) else {}
    stores = _mapping(global_section.get("stores"))
    cache = _mapping(stores.get("response_cache"))
    switches = [field for field in RESPONSE_CACHE_SWITCHES if field in cache]
    if switches:
        lines.append(
            "global.stores.response_cache: removed_field: "
            f"{_removed_fields('global.stores.response_cache', switches)}; {MIGRATE} "
            "(a response_cache plugin turns caching on: in global.plugins for every "
            "request, in a decision's plugins for its own)"
        )
    if "enabled" in _mapping(stores.get("memory")):
        lines.append(
            "global.stores.memory.enabled: removed_field: global.stores.memory.enabled "
            f"is removed; {MIGRATE} (a context plugin with a memory source in global.plugins)"
        )
    replay = _mapping(_mapping(global_section.get("services")).get("router_replay"))
    fields = [field for field in ROUTER_REPLAY_PLUGIN_FIELDS if field in replay]
    if fields:
        lines.append(
            "global.services.router_replay: removed_field: "
            f"{_removed_fields('global.services.router_replay', fields)}; {MIGRATE} "
            "(the configuration of a router_replay plugin in global.plugins)"
        )
    modules = _mapping(_mapping(global_section.get("model_catalog")).get("modules"))
    if "prompt_compression" in modules:
        path = "global.model_catalog.modules.prompt_compression"
        lines.append(
            f"{path}: removed_section: {path} is removed; {MIGRATE} "
            "(compress.signals of a context plugin in global.plugins)"
        )
    return lines


def _removed_fields(path: str, fields: list[str]) -> str:
    listed = fields[0]
    if len(fields) > 1:
        listed = ", ".join(fields[:-1]) + " and " + fields[-1]
    return f"{path}.{listed} {'is' if len(fields) == 1 else 'are'} removed"


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
