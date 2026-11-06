"""Tools forms the Router no longer reads: the tool_selection plugin and
global.integrations.tools (a select block in a decision's tools plugin), and
the tools plugin fields written at the top of its configuration (its policy
fields, which moved under policy, and those of the removed embedding
retrieval, which select replaces).

The Router refuses each one with a line that names `vllm-sr config migrate`,
which rewrites it; the CLI refuses them with the same lines.
"""

from __future__ import annotations

from typing import Any

MIGRATE = "run `vllm-sr config migrate`"

POLICY_FIELDS = (
    "mode",
    "allow_tools",
    "block_tools",
    "strip_tool_history",
    "trusted_facts",
)
RETRIEVAL_FIELDS = ("semantic_selection", "strategy", "dynamic_retrieval")


def removed_tools_forms(data: dict[str, Any]) -> list[str]:
    """The one-line load errors of the removed tools forms in data."""

    lines: list[str] = []
    global_section = data.get("global")
    integrations = (
        global_section.get("integrations") if isinstance(global_section, dict) else None
    )
    if isinstance(integrations, dict) and "tools" in integrations:
        lines.append(
            "global.integrations.tools: removed_section: global.integrations.tools "
            f"is removed; {MIGRATE} (tools.select.tools_db_path and match.judge.top_k "
            "on the decisions that retrieved tools from it)"
        )
    for prefix, routing in _routing_scopes(data):
        decisions = routing.get("decisions")
        for index, decision in enumerate(
            decisions if isinstance(decisions, list) else []
        ):
            if not isinstance(decision, dict):
                continue
            name = decision.get("name")
            label = name if isinstance(name, str) and name else str(index)
            plugins = decision.get("plugins")
            for position, plugin in enumerate(
                plugins if isinstance(plugins, list) else []
            ):
                if not isinstance(plugin, dict):
                    continue
                form = _plugin_form(plugin)
                if form is not None:
                    code, what, becomes = form
                    path = f"{prefix}.decisions[{label}].plugins[{position}]"
                    lines.append(f"{path}: {code}: {what}; {MIGRATE} ({becomes})")
    return lines


def _plugin_form(plugin: dict[str, Any]) -> tuple[str, str, str] | None:
    plugin_type = plugin.get("type")
    if plugin_type == "tool_selection":
        return (
            "removed_plugin_type",
            "plugin tool_selection is removed",
            "a select block in the decision's tools plugin",
        )
    configuration = plugin.get("configuration")
    if plugin_type != "tools" or not isinstance(configuration, dict):
        return None
    fields = [
        field for field in POLICY_FIELDS + RETRIEVAL_FIELDS if field in configuration
    ]
    if not fields:
        return None
    return (
        "removed_field",
        f"tools plugin fields {', '.join(fields)} are not read at the top of its "
        "configuration",
        "mode, allow_tools, block_tools, strip_tool_history and trusted_facts go "
        "under policy; select replaces semantic_selection, strategy and "
        "dynamic_retrieval",
    )


def _routing_scopes(data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    scopes: list[tuple[str, dict[str, Any]]] = []
    routing = data.get("routing")
    if isinstance(routing, dict):
        scopes.append(("routing", routing))
    recipes = data.get("recipes")
    for index, recipe in enumerate(recipes if isinstance(recipes, list) else []):
        if not isinstance(recipe, dict) or not isinstance(recipe.get("routing"), dict):
            continue
        name = recipe.get("name")
        label = name if isinstance(name, str) and name else str(index)
        scopes.append((f"recipes[{label}].routing", recipe["routing"]))
    return scopes
