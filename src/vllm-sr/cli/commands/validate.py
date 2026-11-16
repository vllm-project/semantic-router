"""Validate command implementation."""

import sys
from collections.abc import Callable

from cli.catalog_provider_projection import provider_projection_errors
from cli.config_contract import (
    PROJECTION_FAMILY_SPECS,
    SIGNAL_FAMILY_SPECS,
    iter_routing_profiles,
)
from cli.models import UserConfig
from cli.parser import ConfigParseError, parse_user_config
from cli.router_validation import (
    ROUTER_BINARY_ENV,
    RouterValidationUnavailableError,
    RouterVerdict,
    with_moving_tag_hint,
)
from cli.terminal import echo, error, fields, heading, success
from cli.validation_error import ValidationError
from cli.validator import (
    collect_validation_warnings,
    print_validation_errors,
    print_validation_warnings,
    validate_user_config,
)


def _count_items(value) -> int:
    return len(value or [])


def _signal_summary_lines(signals) -> list[str]:
    """Return non-empty signal summary lines for the canonical v0.5 surface."""
    if not signals:
        return []

    lines = []
    for spec in SIGNAL_FAMILY_SPECS:
        count = _count_items(getattr(signals, spec.signal_attr, None))
        if count > 0:
            lines.append(f"  {spec.display_name}: {count}")
    return lines


def _projection_summary_lines(projections) -> list[str]:
    if not projections:
        return []

    lines = []
    for spec in PROJECTION_FAMILY_SPECS:
        count = _count_items(getattr(projections, spec.projection_attr, None))
        if count > 0:
            lines.append(f"  Projection {spec.display_name.lower()}: {count}")
    return lines


def _aggregate_signal_summary_lines(routing_profiles) -> list[str]:
    lines = []
    profiles = list(routing_profiles)
    for spec in SIGNAL_FAMILY_SPECS:
        count = sum(
            _count_items(getattr(profile.signals, spec.signal_attr, None))
            for _, profile in profiles
        )
        if count > 0:
            lines.append(f"  {spec.display_name}: {count}")
    return lines


def _aggregate_projection_summary_lines(routing_profiles) -> list[str]:
    lines = []
    profiles = list(routing_profiles)
    for spec in PROJECTION_FAMILY_SPECS:
        count = sum(
            _count_items(getattr(profile.projections, spec.projection_attr, None))
            for _, profile in profiles
        )
        if count > 0:
            lines.append(f"  Projection {spec.display_name.lower()}: {count}")
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


def collect_config_errors(user_config: UserConfig) -> list[ValidationError]:
    """Return the semantic and provider projection errors for a parsed config."""

    errors = validate_user_config(user_config, log_summary=False)
    if not errors:
        errors.extend(provider_projection_errors(user_config))
    return errors


def ask_router(
    router_verdict: Callable[[], RouterVerdict] | None,
) -> tuple[RouterVerdict | None, str]:
    """The Router's verdict, or why it could not give one."""

    if router_verdict is None:
        return None, ""
    try:
        return router_verdict(), ""
    except RouterValidationUnavailableError as unavailable:
        return None, str(unavailable)
    except SystemExit:
        return None, "the container runtime is not reachable"


def router_refusal(
    verdict: RouterVerdict | None,
    name_router: str = "`--image` or `VLLM_SR_IMAGE`",
    *,
    pulled: bool = False,
) -> list[ValidationError]:
    """The Router's refusal; *name_router* says how this command names another
    Router, and *pulled* that serve fetched the image in this run."""
    if verdict is None or verdict.valid:
        return []
    message = with_moving_tag_hint(
        f"{verdict.source} refuses it: {verdict.error}",
        verdict,
        name_router,
        pulled=pulled,
    )
    return [ValidationError(message)]


def router_warnings(verdict: RouterVerdict | None) -> list[ValidationError]:
    if verdict is None:
        return []
    return [
        ValidationError(item.message, field=item.field or None)
        for item in verdict.warnings
    ]


def router_check_line(verdict: RouterVerdict | None, unavailable: str, ask: str) -> str:
    """Who checked the file: the Router as well, or only the CLI's structure.

    ask says how to get the Router's verdict when no Router could run.
    """

    if verdict is not None:
        built = f" (built {verdict.built})" if verdict.pull and verdict.built else ""
        line = f"  Checked by {verdict.source}{built} as well as the CLI."
        if verdict.running_router:
            line += (
                "\n  This stack's running Router is a different image: "
                f"{verdict.running_router}."
            )
        return line
    if not unavailable:
        return (
            "  Only the CLI's structural checks ran (--offline); the Router "
            "did not check the model deployments, bindings or its other "
            "load-time rules."
        )
    return (
        f"  Only the CLI's structural checks ran: {unavailable}. The Router "
        "checks the model deployments, bindings and its other load-time "
        f"rules; {ask}."
    )


def validate_command(
    config_path: str,
    *,
    router_verdict: Callable[[], RouterVerdict] | None = None,
):
    """
    Validate user configuration.

    Args:
        config_path: Path to user config.yaml
        router_verdict: Runs the Router's own validation of the file, whatever
            the CLI's own checks find. None runs only the CLI's checks.
    """
    # Parse config
    try:
        user_config = parse_user_config(config_path, log_summary=False)
    except ConfigParseError as e:
        error(f"Configuration parsing failed: {e}")
        sys.exit(1)

    errors = collect_config_errors(user_config)
    verdict, unavailable = ask_router(router_verdict)
    errors.extend(router_refusal(verdict))
    if errors:
        print_validation_errors(errors)
        print_validation_warnings(router_warnings(verdict))
        sys.exit(1)

    success("Configuration is valid")
    echo(
        router_check_line(
            verdict,
            unavailable,
            f"pass --router-binary (or set {ROUTER_BINARY_ENV}), --image or "
            "--endpoint to ask it",
        )
    )
    print_validation_warnings(
        router_warnings(verdict) + collect_validation_warnings(user_config)
    )
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
        echo("  Signals: None (catch-all routing is supported)")

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
