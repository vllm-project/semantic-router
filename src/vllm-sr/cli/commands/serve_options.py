"""The option groups of `vllm-sr serve`.

`serve` runs in one of three modes: the Router on the docker target, the
Router on the kubernetes target, or engine mode (`vllm-sr serve MODEL`, the
model runtime in a container). Every option belongs to one group, and a group
names the modes it applies to. `--help` prints the groups, and an option set
in a mode its group does not apply to is an error that names where it applies.
"""

from __future__ import annotations

from dataclasses import dataclass

import click
from click.core import ParameterSource

from cli.deployment_backend import TARGET_DOCKER, TARGET_KUBERNETES
from cli.gateway_mode import runs_envoy

# Router mode's modes are its targets.
MODE_DOCKER = TARGET_DOCKER
MODE_KUBERNETES = TARGET_KUBERNETES
MODE_ENGINE = "engine"


@dataclass(frozen=True)
class OptionGroup:
    title: str
    modes: frozenset[str]
    # Completes "<option> applies to ..." when the option is set elsewhere.
    applies_to: str
    options: tuple[str, ...]


SERVE_OPTION_GROUPS = (
    OptionGroup(
        "Common options (docker, kubernetes and engine mode)",
        frozenset({MODE_DOCKER, MODE_KUBERNETES, MODE_ENGINE}),
        "every mode",
        ("platform", "image", "log_level"),
    ),
    OptionGroup(
        "Router options (docker and kubernetes targets)",
        frozenset({MODE_DOCKER, MODE_KUBERNETES}),
        "router mode; engine mode serves only models",
        (
            "config",
            "target",
            "gateway",
            "minimal",
            "readonly",
            "algorithm",
            "decision_model",
        ),
    ),
    OptionGroup(
        "Container options (docker target and engine mode)",
        frozenset({MODE_DOCKER, MODE_ENGINE}),
        "the docker target and engine mode",
        ("image_pull_policy", "container_runtime", "runtime"),
    ),
    OptionGroup(
        "Docker target",
        frozenset({MODE_DOCKER}),
        "the docker target",
        (
            "router_image",
            "envoy_image",
            "dashboard_image",
            "startup_timeout",
            "replace_active_config",
            "recipe_env_names",
        ),
    ),
    OptionGroup(
        "Kubernetes target",
        frozenset({MODE_KUBERNETES}),
        "the kubernetes target (--target kubernetes)",
        ("namespace", "context", "profile", "chart_dir"),
    ),
    OptionGroup(
        "Engine mode (vllm-sr serve MODEL)",
        frozenset({MODE_ENGINE}),
        "engine mode; pass a MODEL or --models",
        ("models_file", "revision", "device", "host", "port", "runtime_profile"),
    ),
)

OPTION_GROUP = {name: group for group in SERVE_OPTION_GROUPS for name in group.options}

# A clearer message than the group's for an option a user may carry over
# from another mode.
_MOVED_OPTIONS = {
    (MODE_ENGINE, "profile"): (
        "--profile is the kubernetes deployment profile; engine mode takes "
        "--runtime-profile"
    ),
    (MODE_ENGINE, "decision_model"): (
        "--decision-model applies to the Router stack: it chooses the model that "
        "answers the Router's questions. Engine mode (vllm-sr serve MODEL) serves "
        "the models you name; drop MODEL to serve the Router"
    ),
}


class GroupedServeCommand(click.Command):
    """A command whose --help lists its options by group."""

    def format_options(
        self, ctx: click.Context, formatter: click.HelpFormatter
    ) -> None:
        sections: dict[str, list[tuple[str, str]]] = {
            group.title: [] for group in SERVE_OPTION_GROUPS
        }
        other: list[tuple[str, str]] = []
        for param in self.get_params(ctx):
            record = param.get_help_record(ctx)
            if record is None:
                continue
            group = OPTION_GROUP.get(param.name or "")
            (sections[group.title] if group else other).append(record)
        for title, records in sections.items():
            if records:
                with formatter.section(title):
                    formatter.write_dl(records)
        if other:
            with formatter.section("Other options"):
                formatter.write_dl(other)


def explicit(ctx: click.Context, name: str) -> bool:
    """Whether the command line or the environment set the option."""

    return ctx.get_parameter_source(name) not in (None, ParameterSource.DEFAULT)


def reject_misplaced_options(ctx: click.Context, mode: str) -> None:
    """Fail on the first option whose group does not apply to *mode*."""

    for group in SERVE_OPTION_GROUPS:
        if mode in group.modes:
            continue
        for name in group.options:
            if not explicit(ctx, name):
                continue
            message = _MOVED_OPTIONS.get((mode, name))
            raise click.UsageError(
                message or f"{_flag(ctx, name)} applies to {group.applies_to}",
                ctx=ctx,
            )


def reject_envoy_options(ctx: click.Context, gateway: str) -> None:
    """A standalone stack runs no Envoy container to take an Envoy image."""

    if not runs_envoy(gateway) and explicit(ctx, "envoy_image"):
        raise click.UsageError(
            "--envoy-image applies to --gateway extproc; the standalone Router "
            "runs no Envoy container",
            ctx=ctx,
        )


def resolve_container_runtime(ctx: click.Context, container_runtime, runtime):
    """--container-runtime, or the --runtime spelling it replaced, with a warning."""

    if runtime is None:
        return container_runtime
    if container_runtime is not None and container_runtime != runtime:
        raise click.UsageError(
            "--runtime is the old name of --container-runtime; pass only one", ctx=ctx
        )
    click.echo(
        "Warning: --runtime is renamed to --container-runtime; --runtime keeps "
        "working for this release only",
        err=True,
    )
    return runtime


def _flag(ctx: click.Context, name: str) -> str:
    for param in ctx.command.params:
        if param.name == name and param.opts:
            return max(param.opts, key=len)
    return "--" + name.replace("_", "-")
