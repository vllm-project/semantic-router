"""Plugin forms the Router no longer reads: the response_jailbreak and
hallucination plugins (a response_guard rule over the signals they checked),
masking's own threshold (the pii rule masking names), a system_prompt mode
other than insert or replace, and request_params' strip_unknown, which the
Router never read.

The Router refuses each one with a line that names `vllm-sr config migrate`,
which rewrites it; the CLI refuses them with the same lines.
"""

from __future__ import annotations

from typing import Any

from cli.config_recipe_scopes import recipe_scopes

MIGRATE = "run `vllm-sr config migrate`"

REMOVED_PLUGIN_TYPES = {
    "response_jailbreak": "a response_guard rule over a jailbreak signal with stage: response",
    "hallucination": "a response_guard rule over the fact_check and hallucination signals",
}
STRIP_UNKNOWN = (
    "removed_field",
    "request_params strip_unknown is removed: the Router never read it",
    "nothing; the Router refuses a request field its wire format does not declare",
)


def removed_plugin_forms(data: dict[str, Any]) -> list[str]:
    """The one-line load errors of the removed plugin forms in data."""

    lines: list[str] = []
    for prefix, routing in recipe_scopes(data):
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
        lines += _strip_unknown_lines(f"{prefix}.plugins", routing.get("plugins"))
    global_section = data.get("global")
    if isinstance(global_section, dict):
        lines += _strip_unknown_lines("global.plugins", global_section.get("plugins"))
    return lines


def _strip_unknown_lines(path: str, plugins: Any) -> list[str]:
    """The lines of a recipe's plugins or global.plugins, where strip_unknown
    is the only one of these forms the Router looks for."""

    code, what, becomes = STRIP_UNKNOWN
    return [
        f"{path}[{position}]: {code}: {what}; {MIGRATE} ({becomes})"
        for position, plugin in enumerate(plugins if isinstance(plugins, list) else [])
        if _sets_strip_unknown(plugin)
    ]


def _sets_strip_unknown(plugin: Any) -> bool:
    if not isinstance(plugin, dict):
        return False
    configuration = plugin.get("configuration")
    return (
        plugin.get("type") == "request_params"
        and isinstance(configuration, dict)
        and "strip_unknown" in configuration
    )


def _plugin_form(plugin: dict[str, Any]) -> tuple[str, str, str] | None:
    plugin_type = plugin.get("type")
    configuration = plugin.get("configuration")
    if not isinstance(configuration, dict):
        configuration = {}
    if plugin_type in REMOVED_PLUGIN_TYPES:
        return (
            "removed_plugin_type",
            f"plugin {plugin_type} was removed",
            REMOVED_PLUGIN_TYPES[plugin_type],
        )
    if plugin_type == "masking":
        if "threshold" in configuration:
            return (
                "removed_field",
                "masking.threshold was removed",
                "masking.signal naming a pii rule, whose predicate applies",
            )
        if not configuration.get("signal") and not configuration.get("rules"):
            return (
                "removed_field",
                "masking without a signal was removed",
                "masking.signal naming the pii rule whose detections it masks",
            )
    if plugin_type == "system_prompt":
        mode = configuration.get("mode")
        if isinstance(mode, str) and mode not in ("", "insert", "replace"):
            return (
                "invalid_value",
                f'system_prompt mode "{mode}" is neither insert nor replace',
                "mode: replace, which now drops every system and developer instruction the caller sent, not only the first system message",
            )
    if _sets_strip_unknown(plugin):
        return STRIP_UNKNOWN
    return None
