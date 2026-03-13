"""Runtime-oriented Click command entrypoints."""

from __future__ import annotations

import webbrowser
from pathlib import Path

import click

from cli.bootstrap import (
    ensure_bootstrap_workspace,
)
from cli.commands.common import exit_with_logged_error
from cli.commands.runtime_support import (
    ALGORITHM_TYPES,
    append_passthrough_env_vars,
    apply_runtime_mode_env_vars,
    log_bootstrap_result,
    resolve_effective_config_path,
    validate_setup_mode_flags,
)
from cli.commands.runtime_support import (
    inject_algorithm_into_config as _inject_algorithm_into_config,
)
from cli.consts import (
    DEFAULT_IMAGE_PULL_POLICY,
    IMAGE_PULL_POLICY_ALWAYS,
    IMAGE_PULL_POLICY_IF_NOT_PRESENT,
    IMAGE_PULL_POLICY_NEVER,
    VLLM_SR_DOCKER_IMAGE_DEFAULT,
)
from cli.core import show_logs, show_status, start_vllm_sr, stop_vllm_sr
from cli.docker_cli import docker_container_status
from cli.runtime_stack import resolve_runtime_stack
from cli.utils import getLogger

log = getLogger(__name__)
inject_algorithm_into_config = _inject_algorithm_into_config


@click.command()
@click.option(
    "--config",
    default="config.yaml",
    help="Path to config file (default: config.yaml)",
)
@click.option(
    "--image",
    default=None,
    help=f"Docker image to use (default: {VLLM_SR_DOCKER_IMAGE_DEFAULT})",
)
@click.option(
    "--image-pull-policy",
    type=click.Choice(
        [
            IMAGE_PULL_POLICY_ALWAYS,
            IMAGE_PULL_POLICY_IF_NOT_PRESENT,
            IMAGE_PULL_POLICY_NEVER,
        ],
        case_sensitive=False,
    ),
    default=DEFAULT_IMAGE_PULL_POLICY,
    help=f"Image pull policy: always, ifnotpresent, never (default: {DEFAULT_IMAGE_PULL_POLICY})",
)
@click.option(
    "--readonly",
    is_flag=True,
    default=False,
    help="Run dashboard in read-only mode (disable config editing, allow playground only)",
)
@click.option(
    "--minimal",
    is_flag=True,
    default=False,
    help="Start in minimal mode: only router + envoy, no dashboard or observability (Jaeger, Prometheus, Grafana)",
)
@click.option(
    "--platform",
    default=None,
    help="Platform branding (e.g., 'amd' for AMD GPU deployments). "
    "When set to amd, serve defaults to the ROCm image unless --image or VLLM_SR_IMAGE is provided.",
)
@click.option(
    "--algorithm",
    type=click.Choice(ALGORITHM_TYPES, case_sensitive=False),
    default=None,
    help="Model selection algorithm: static (default), elo (rating-based), "
    "router_dc (embedding similarity), automix (cost-quality optimization), "
    "hybrid (combined methods). Overrides config file setting.",
)
@exit_with_logged_error(log, interrupt_message="\nInterrupted by user")
def serve(
    config: str,
    image: str | None,
    image_pull_policy: str,
    readonly: bool,
    minimal: bool,
    platform: str | None,
    algorithm: str | None,
) -> None:
    """
    Start vLLM Semantic Router.

    Ports are configured in config.yaml under 'listeners' section.

    MODEL SELECTION ALGORITHMS:

    \b
    static     - Use first configured model (default, no learning)
    elo        - Rating-based selection using user feedback
    router_dc  - Query-model matching via embedding similarity
    automix    - Cost-quality optimization using POMDP
    hybrid     - Combine multiple methods with configurable weights
    thompson   - Thompson Sampling with exploration/exploitation (RL-driven)
    gmtrouter  - Graph neural network for personalized routing (RL-driven)
    router_r1  - LLM-as-router with think/route actions (RL-driven)

    Examples:
        # Basic usage (uses config.yaml)
        vllm-sr serve

        # Custom config file
        vllm-sr serve --config my-config.yaml

        # Use Elo rating selection (learns from feedback)
        vllm-sr serve --algorithm elo

        # Use cost-optimized selection
        vllm-sr serve --algorithm automix

        # Custom image
        vllm-sr serve --image ghcr.io/vllm-project/semantic-router/vllm-sr:latest

        # Pull policy
        vllm-sr serve --image-pull-policy always

        # Read-only dashboard (for public beta)
        vllm-sr serve --readonly

        # Minimal mode (no dashboard, no observability)
        vllm-sr serve --minimal

        # Platform branding (for AMD deployments)
        vllm-sr serve --platform amd
    """
    requested_config = config
    bootstrap = ensure_bootstrap_workspace(Path(config))
    config_path = bootstrap.config_path
    setup_mode = bootstrap.setup_mode

    log_bootstrap_result(requested_config, bootstrap)
    log.info(f"Using config file: {config_path}")

    validate_setup_mode_flags(setup_mode, minimal, readonly)

    env_vars: dict[str, str] = {}
    append_passthrough_env_vars(env_vars)
    apply_runtime_mode_env_vars(env_vars, minimal, readonly, setup_mode, platform)

    effective_config_path = resolve_effective_config_path(
        config_path, algorithm, setup_mode
    )
    start_vllm_sr(
        config_file=str(effective_config_path.absolute()),
        env_vars=env_vars,
        image=image,
        pull_policy=image_pull_policy,
        enable_observability=not minimal and not setup_mode,
    )


@click.command()
@click.argument(
    "service", type=click.Choice(["envoy", "router", "dashboard", "all"]), default="all"
)
@exit_with_logged_error(log)
def status(service: str) -> None:
    """
    Show status of vLLM Semantic Router services.

    Examples:
        vllm-sr status              # Show all services
        vllm-sr status all          # Show all services
        vllm-sr status envoy        # Show envoy status
        vllm-sr status router       # Show router status
        vllm-sr status dashboard    # Show dashboard status
    """
    show_status(service)


@click.command()
@click.argument("service", type=click.Choice(["envoy", "router", "dashboard"]))
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
@exit_with_logged_error(log, interrupt_message="\nLog streaming stopped")
def logs(service: str, follow: bool) -> None:
    """
    Show logs from vLLM Semantic Router service.

    Examples:
        vllm-sr logs envoy
        vllm-sr logs router
        vllm-sr logs dashboard
        vllm-sr logs envoy --follow
        vllm-sr logs router -f
    """
    show_logs(service, follow=follow)


@click.command()
@exit_with_logged_error(log)
def stop() -> None:
    """
    Stop vLLM Semantic Router.

    Examples:
        vllm-sr stop
    """
    stop_vllm_sr()


@click.command()
@click.option("--no-open", is_flag=True, help="Don't open browser, just show URL")
@exit_with_logged_error(log)
def dashboard(no_open: bool) -> None:
    """
    Open the dashboard in your default web browser.

    Examples:
        vllm-sr dashboard
        vllm-sr dashboard --no-open
    """
    stack_layout = resolve_runtime_stack()
    status = docker_container_status(stack_layout.container_name)
    if status != "running":
        raise ValueError("vLLM Semantic Router is not running")

    dashboard_url = stack_layout.dashboard_url
    if no_open:
        log.info(f"Dashboard URL: {dashboard_url}")
        return

    log.info(f"Opening dashboard: {dashboard_url}")
    webbrowser.open(dashboard_url)
    log.info("Dashboard opened in browser")
