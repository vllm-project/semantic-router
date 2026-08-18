"""Validate command implementation."""

import sys

from cli.config_contract import iter_routing_profiles
from cli.parser import ConfigParseError, parse_user_config
from cli.terminal import echo, error, fields, heading, success
from cli.validator import print_validation_errors, validate_user_config

_SIGNAL_SUMMARY_FIELDS = (
    ("Keyword signals", "keywords"),
    ("Embedding signals", "embeddings"),
    ("Domains", "domains"),
    ("Fact check signals", "fact_check"),
    ("User feedback signals", "user_feedbacks"),
    ("Reask signals", "reasks"),
    ("Preference signals", "preferences"),
    ("Language signals", "language"),
    ("Context signals", "context"),
    ("Structure signals", "structure"),
    ("Complexity signals", "complexity"),
    ("Modality signals", "modality"),
    ("Authz signals", "role_bindings"),
    ("Jailbreak signals", "jailbreak"),
    ("PII signals", "pii"),
    ("Knowledge-base signals", "kb"),
    ("Conversation signals", "conversation"),
    ("Event signals", "events"),
    ("Metadata signals", "metadata"),
    ("Classifier signals", "classifiers"),
)


def _count_items(value) -> int:
    return len(value or [])


def _signal_summary_lines(signals) -> list[str]:
    """Return non-empty signal summary lines for the canonical v0.3 surface."""
    if not signals:
        return []

    lines = []
    for label, field_name in _SIGNAL_SUMMARY_FIELDS:
        count = _count_items(getattr(signals, field_name, None))
        if count > 0:
            lines.append(f"  {label}: {count}")
    return lines


def _projection_summary_lines(projections) -> list[str]:
    if not projections:
        return []

    lines = []
    for label, field_name in (
        ("Projection partitions", "partitions"),
        ("Projection scores", "scores"),
        ("Projection mappings", "mappings"),
    ):
        count = _count_items(getattr(projections, field_name, None))
        if count > 0:
            lines.append(f"  {label}: {count}")
    return lines


def _aggregate_signal_summary_lines(routing_profiles) -> list[str]:
    lines = []
    profiles = list(routing_profiles)
    for label, field_name in _SIGNAL_SUMMARY_FIELDS:
        count = sum(
            _count_items(getattr(profile.signals, field_name, None))
            for _, profile in profiles
        )
        if count > 0:
            lines.append(f"  {label}: {count}")
    return lines


def _aggregate_projection_summary_lines(routing_profiles) -> list[str]:
    lines = []
    profiles = list(routing_profiles)
    for label, field_name in (
        ("Projection partitions", "partitions"),
        ("Projection scores", "scores"),
        ("Projection mappings", "mappings"),
    ):
        count = sum(
            _count_items(getattr(profile.projections, field_name, None))
            for _, profile in profiles
        )
        if count > 0:
            lines.append(f"  {label}: {count}")
    return lines


def _plugin_summary_lines(decisions) -> list[str]:
    plugin_types: dict[str, int] = {}
    decisions_with_plugins = 0
    for decision in decisions:
        if not decision.plugins:
            continue
        decisions_with_plugins += 1
        for plugin in decision.plugins:
            plugin_type = (
                plugin.type.value if hasattr(plugin.type, "value") else str(plugin.type)
            )
            plugin_types[plugin_type] = plugin_types.get(plugin_type, 0) + 1

    total_plugins = sum(plugin_types.values())
    if total_plugins == 0:
        return []
    type_counts = ", ".join(
        f"{plugin_type}: {count}" for plugin_type, count in sorted(plugin_types.items())
    )
    return [
        f"  Plugins: {total_plugins} total ({decisions_with_plugins} decisions)",
        f"    Types: {type_counts}",
    ]


def validate_command(config_path: str):
    """
    Validate user configuration.

    Args:
        config_path: Path to user config.yaml
    """
    # Parse config
    try:
        user_config = parse_user_config(config_path, log_summary=False)
    except ConfigParseError as e:
        error(f"Configuration parsing failed: {e}")
        sys.exit(1)

    # Validate config
    errors = validate_user_config(user_config, log_summary=False)

    if errors:
        print_validation_errors(errors)
        sys.exit(1)

    success("Configuration is valid")
    heading("Configuration summary")
    fields(
        (
            ("Path", config_path),
            ("Version", user_config.version),
            ("Listeners", len(user_config.listeners)),
        )
    )

    routing_profiles = list(iter_routing_profiles(user_config))
    signal_lines = _aggregate_signal_summary_lines(routing_profiles)
    if signal_lines:
        for line in signal_lines:
            echo(line)
    else:
        echo(
            "  Signals: None (catch-all routing is supported; domain categories will auto-generate when needed)"
        )

    for line in _aggregate_projection_summary_lines(routing_profiles):
        echo(line)

    default_decisions = len(user_config.decisions)
    recipe_decisions = sum(
        len(profile.decisions)
        for name, profile in routing_profiles
        if name != "default"
    )
    all_decisions = [
        decision for _, profile in routing_profiles for decision in profile.decisions
    ]
    echo(f"  Entrypoints: {len(user_config.entrypoints)}")
    echo(f"  Recipes: {len(user_config.recipes)}")
    echo(
        f"  Decisions: {len(all_decisions)} total "
        f"({default_decisions} default, {recipe_decisions} recipe-owned)"
    )

    for line in _plugin_summary_lines(all_decisions):
        echo(line)

    echo(f"  Models: {len(user_config.providers.models)}")
    echo(f"  Default model: {user_config.providers.default_model}")
