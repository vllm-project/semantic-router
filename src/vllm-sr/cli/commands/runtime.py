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
    configure_runtime_override_env_vars,
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
from cli.deployment_backend import DEFAULT_TARGET, VALID_TARGETS, resolve_target
from cli.utils import get_logger

log = get_logger(__name__)
inject_algorithm_into_config = _inject_algorithm_into_config

TARGET_HELP = (
    f"Deployment target: {', '.join(VALID_TARGETS)} (default: {DEFAULT_TARGET})"
)


def _build_backend(target: str | None, **k8s_kwargs):
    """Instantiate the right DeploymentBackend for *target*."""
    resolved = resolve_target(target)
    if resolved == "k8s":
        from cli.k8s_backend import K8sBackend  # noqa: PLC0415

        return K8sBackend(**{k: v for k, v in k8s_kwargs.items() if v is not None})

    from cli.docker_backend import DockerBackend  # noqa: PLC0415

    return DockerBackend()


def _execute_serve(
    config: str,
    image: str | None,
    router_image: str | None,
    envoy_image: str | None,
    dashboard_image: str | None,
    image_pull_policy: str,
    readonly: bool,
    minimal: bool,
    log_level: str | None,
    platform: str | None,
    algorithm: str | None,
    target: str | None,
    namespace: str | None,
    context: str | None,
    profile: str | None,
    chart_dir: str | None,
) -> None:
    """Bootstrap workspace, resolve config, and delegate to the deployment backend."""
    requested_config = config
    bootstrap = ensure_bootstrap_workspace(Path(config))
    config_path = bootstrap.config_path
    setup_mode = bootstrap.setup_mode

    log_bootstrap_result(requested_config, bootstrap)
    log.info(f"Using config file: {config_path}")

    validate_setup_mode_flags(setup_mode, minimal, readonly)

    env_vars: dict[str, str] = {}
    append_passthrough_env_vars(env_vars)
    apply_runtime_mode_env_vars(
        env_vars,
        minimal,
        readonly,
        setup_mode,
        platform,
        algorithm,
        log_level=log_level,
    )

    effective_config_path = resolve_effective_config_path(
        config_path, algorithm, setup_mode, platform
    )
    configure_runtime_override_env_vars(env_vars, config_path, effective_config_path)

    backend = _build_backend(
        target,
        namespace=namespace,
        context=context,
        profile=profile,
        chart_dir=chart_dir,
    )
    backend.deploy(
        config_file=str(effective_config_path.absolute()),
        source_config_file=str(config_path.absolute()),
        runtime_config_file=str(effective_config_path.absolute()),
        env_vars=env_vars,
        image=image,
        router_image=router_image,
        envoy_image=envoy_image,
        dashboard_image=dashboard_image,
        pull_policy=image_pull_policy,
        enable_observability=not minimal and not setup_mode,
    )


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
    "--router-image",
    default=None,
    help="Docker image for the router container (Docker target only; defaults to --image or VLLM_SR_IMAGE)",
)
@click.option(
    "--envoy-image",
    default=None,
    help="Docker image for the Envoy container (Docker target only; defaults to --image or VLLM_SR_IMAGE)",
)
@click.option(
    "--dashboard-image",
    default=None,
    help="Docker image for the dashboard container (Docker target only; defaults to --image or VLLM_SR_IMAGE)",
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
    "--log-level",
    type=click.Choice(
        ["debug", "info", "warn", "warning", "error", "dpanic", "panic", "fatal"],
        case_sensitive=False,
    ),
    default=None,
    help="Router log level override (debug, info, warn, error, dpanic, panic, fatal)",
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
@click.option("--target", default=None, help=TARGET_HELP)
@click.option(
    "--namespace", default=None, help="Kubernetes namespace (k8s target only)"
)
@click.option(
    "--context", default=None, help="kubectl / Helm context (k8s target only)"
)
@click.option(
    "--profile",
    default=None,
    help="Deployment profile: dev, prod (k8s target only). Selects values-<profile>.yaml defaults.",
)
@click.option(
    "--chart-dir", default=None, help="Path to Helm chart directory (k8s target only)"
)
@exit_with_logged_error(log, interrupt_message="\nInterrupted by user")
def serve(
    config: str,
    image: str | None,
    router_image: str | None,
    envoy_image: str | None,
    dashboard_image: str | None,
    image_pull_policy: str,
    readonly: bool,
    minimal: bool,
    log_level: str | None,
    platform: str | None,
    algorithm: str | None,
    target: str | None,
    namespace: str | None,
    context: str | None,
    profile: str | None,
    chart_dir: str | None,
) -> None:
    """
    Start vLLM Semantic Router.

    Ports are configured in config.yaml under 'listeners' section.

    DEPLOYMENT TARGETS:

    \b
    docker  - Local Docker deployment (default)
    k8s     - Kubernetes deployment via Helm

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
        # Basic usage (uses config.yaml, Docker target)
        vllm-sr serve

        # Deploy to Kubernetes
        vllm-sr serve --target k8s --namespace my-ns --profile dev

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

        # Start router with debug logs
        vllm-sr serve --log-level debug

        # Platform branding (for AMD deployments)
        vllm-sr serve --platform amd

    """
    _execute_serve(
        config,
        image,
        router_image,
        envoy_image,
        dashboard_image,
        image_pull_policy,
        readonly,
        minimal,
        log_level,
        platform,
        algorithm,
        target,
        namespace,
        context,
        profile,
        chart_dir,
    )


@click.command()
@click.argument(
    "service",
    type=click.Choice(["envoy", "router", "dashboard", "simulator", "all"]),
    default="all",
)
@click.option("--target", default=None, help=TARGET_HELP)
@click.option(
    "--namespace", default=None, help="Kubernetes namespace (k8s target only)"
)
@click.option(
    "--context", default=None, help="kubectl / Helm context (k8s target only)"
)
@exit_with_logged_error(log)
def status(
    service: str,
    target: str | None,
    namespace: str | None,
    context: str | None,
) -> None:
    """
    Show status of vLLM Semantic Router services.

    Examples:
        vllm-sr status              # Show all services (Docker)
        vllm-sr status all          # Show all services
        vllm-sr status router       # Show router status
        vllm-sr status dashboard    # Show dashboard status
        vllm-sr status simulator    # Show simulator status
        vllm-sr status --target k8s # Show Kubernetes status
    """
    backend = _build_backend(target, namespace=namespace, context=context)
    backend.status(service)


@click.command()
@click.argument(
    "service", type=click.Choice(["envoy", "router", "dashboard", "simulator"])
)
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
@click.option("--target", default=None, help=TARGET_HELP)
@click.option(
    "--namespace", default=None, help="Kubernetes namespace (k8s target only)"
)
@click.option(
    "--context", default=None, help="kubectl / Helm context (k8s target only)"
)
@exit_with_logged_error(log, interrupt_message="\nLog streaming stopped")
def logs(
    service: str,
    follow: bool,
    target: str | None,
    namespace: str | None,
    context: str | None,
) -> None:
    """
    Show logs from vLLM Semantic Router service.

    Examples:
        vllm-sr logs envoy
        vllm-sr logs router
        vllm-sr logs dashboard
        vllm-sr logs simulator
        vllm-sr logs envoy --follow
        vllm-sr logs router -f
        vllm-sr logs router --target k8s        # Kubernetes logs
        vllm-sr logs router --target k8s -f     # Follow K8s logs
    """
    backend = _build_backend(target, namespace=namespace, context=context)
    backend.logs(service, follow=follow)


@click.command()
@click.option("--target", default=None, help=TARGET_HELP)
@click.option(
    "--namespace", default=None, help="Kubernetes namespace (k8s target only)"
)
@click.option(
    "--context", default=None, help="kubectl / Helm context (k8s target only)"
)
@exit_with_logged_error(log)
def stop(
    target: str | None,
    namespace: str | None,
    context: str | None,
) -> None:
    """
    Stop vLLM Semantic Router.

    Examples:
        vllm-sr stop                # Stop Docker stack
        vllm-sr stop --target k8s   # Uninstall Helm release
    """
    backend = _build_backend(target, namespace=namespace, context=context)
    backend.teardown()


@click.command()
@click.option("--no-open", is_flag=True, help="Don't open browser, just show URL")
@click.option("--target", default=None, help=TARGET_HELP)
@click.option(
    "--namespace", default=None, help="Kubernetes namespace (k8s target only)"
)
@click.option(
    "--context", default=None, help="kubectl / Helm context (k8s target only)"
)
@exit_with_logged_error(log)
def dashboard(
    no_open: bool,
    target: str | None,
    namespace: str | None,
    context: str | None,
) -> None:
    """
    Open the dashboard in your default web browser.

    Examples:
        vllm-sr dashboard                   # Docker dashboard
        vllm-sr dashboard --target k8s      # Show K8s dashboard URL
        vllm-sr dashboard --no-open
    """
    backend = _build_backend(target, namespace=namespace, context=context)
    if not backend.is_running():
        raise ValueError("vLLM Semantic Router is not running")

    dashboard_url = backend.get_dashboard_url()
    if dashboard_url is None:
        log.info("Dashboard URL could not be determined")
        return

    if no_open:
        log.info(f"Dashboard URL: {dashboard_url}")
        return

    log.info(f"Opening dashboard: {dashboard_url}")
    webbrowser.open(dashboard_url)
    log.info("Dashboard opened in browser")
