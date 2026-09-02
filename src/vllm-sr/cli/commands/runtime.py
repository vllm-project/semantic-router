"""Runtime-oriented Click command entrypoints."""

from __future__ import annotations

import os
import webbrowser
from pathlib import Path

import click

from cli.bootstrap import (
    ensure_bootstrap_workspace,
    is_setup_mode_config,
)
from cli.commands.common import exit_with_logged_error
from cli.commands.runtime_config_mutation import (
    ALGORITHM_TYPES,
)
from cli.commands.runtime_config_mutation import (
    inject_algorithm_into_config as _inject_algorithm_into_config,
)
from cli.commands.runtime_help import SERVE_HELP
from cli.commands.runtime_serve_config import _prepare_effective_serve_config
from cli.commands.runtime_support import (
    append_passthrough_env_vars,
    apply_container_runtime_override,
    apply_runtime_mode_env_vars,
    configure_recipe_env_bindings,
    configure_runtime_override_env_vars,
    log_bootstrap_result,
    validate_setup_mode_flags,
)
from cli.consts import (
    DEFAULT_IMAGE_PULL_POLICY,
    IMAGE_PULL_POLICY_ALWAYS,
    IMAGE_PULL_POLICY_IF_NOT_PRESENT,
    IMAGE_PULL_POLICY_NEVER,
    PLATFORM_AMD,
    PLATFORM_NVIDIA,
    SUPPORTED_CONTAINER_RUNTIMES,
    VLLM_SR_CONTAINER_IMAGE_DEFAULT,
)
from cli.deployment_backend import DEFAULT_TARGET, VALID_TARGETS, resolve_target
from cli.terminal import fields, heading, success
from cli.utils import get_logger

log = get_logger(__name__)


def inject_algorithm_into_config(config_path: Path, algorithm: str) -> Path:
    return _inject_algorithm_into_config(config_path, algorithm)


TARGET_HELP = (
    f"Deployment target: {', '.join(VALID_TARGETS)} (default: {DEFAULT_TARGET})"
)

RUNTIME_HELP = (
    "Container runtime for the local Docker target: "
    f"{', '.join(SUPPORTED_CONTAINER_RUNTIMES)}. "
    "Equivalent to setting CONTAINER_RUNTIME=<runtime>. Has no effect on the k8s target."
)


def _build_backend(target: str | None, **k8s_kwargs):
    """Instantiate the right DeploymentBackend for *target*."""
    resolved = resolve_target(target)
    if resolved == "k8s":
        from cli.k8s_backend import K8sBackend  # noqa: PLC0415

        return K8sBackend(**{k: v for k, v in k8s_kwargs.items() if v is not None})

    from cli.container_backend import ContainerBackend  # noqa: PLC0415

    return ContainerBackend()


def _resolve_serve_config(
    config: str,
    resolved_target: str,
) -> tuple[Path, bool]:
    """Resolve one user-owned config or bootstrap the local Dashboard workspace."""

    if resolved_target != "docker":
        config_path = Path(config).expanduser()
        if not config_path.is_file():
            raise ValueError(
                "Kubernetes deployment requires an existing complete --config; "
                "empty-directory Dashboard setup is supported only by local Docker"
            )
        if is_setup_mode_config(config_path):
            raise ValueError(
                "Kubernetes deployment does not support Dashboard setup-mode "
                "configs; complete the config locally or provide a canonical config"
            )
        return config_path, False
    bootstrap = ensure_bootstrap_workspace(Path(config))
    log_bootstrap_result(config, bootstrap)
    return bootstrap.config_path, bootstrap.setup_mode


def _validate_target_platform(resolved_target: str, platform: str | None) -> None:
    """Reject local-only GPU shorthand before workspace or backend mutation."""

    platform_hint = (
        (platform or "").strip()
        or os.getenv("VLLM_SR_PLATFORM", "").strip()
        or os.getenv("DASHBOARD_PLATFORM", "").strip()
    ).lower()
    if resolved_target != "docker" and platform_hint in {
        PLATFORM_AMD,
        PLATFORM_NVIDIA,
    }:
        raise ValueError(
            "--platform amd/nvidia is supported only for local Docker deployments. "
            "For Kubernetes, configure the GPU image, resources, and device plugin "
            "through a Helm profile or the operator."
        )


def _deploy_serve_backend(
    *,
    resolved_target: str,
    config_path: Path,
    effective_config_path: Path,
    effective_config_document: dict[str, object] | None,
    runtime_lock,
    env_vars: dict[str, str],
    namespace: str | None,
    context: str | None,
    profile: str | None,
    chart_dir: str | None,
    image: str | None,
    router_image: str | None,
    envoy_image: str | None,
    dashboard_image: str | None,
    image_pull_policy: str,
    minimal: bool,
    readonly: bool,
) -> None:
    """Deploy one prepared runtime."""

    backend = _build_backend(
        resolved_target,
        namespace=namespace,
        context=context,
        profile=profile,
        chart_dir=chart_dir,
    )
    backend.deploy(
        config_file=str(effective_config_path.absolute()),
        source_config_file=str(config_path.absolute()),
        runtime_config_file=str(effective_config_path.absolute()),
        runtime_config_lock=runtime_lock,
        config_document=effective_config_document,
        env_vars=env_vars,
        image=image,
        router_image=router_image,
        envoy_image=envoy_image,
        dashboard_image=dashboard_image,
        pull_policy=image_pull_policy,
        enable_observability=not minimal,
        minimal=minimal,
        readonly=readonly,
    )


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
    runtime: str | None,
    recipe_env_names: tuple[str, ...] = (),
) -> None:
    """Bootstrap workspace, resolve config, and delegate to the deployment backend."""
    resolved_target = resolve_target(target)
    _validate_target_platform(resolved_target, platform)
    apply_container_runtime_override(runtime)
    config_path, source_setup_mode = _resolve_serve_config(config, resolved_target)
    log.info(f"Using config file: {config_path}")

    env_vars: dict[str, str] = {}
    append_passthrough_env_vars(env_vars, config_path)
    recipe_env_bindings = configure_recipe_env_bindings(env_vars, recipe_env_names)
    runtime_lock = None
    try:
        effective_config_path, setup_mode, runtime_lock, effective_config_document = (
            _prepare_effective_serve_config(
                config_path,
                resolved_target=resolved_target,
                algorithm=algorithm,
                source_setup_mode=source_setup_mode,
                platform=platform,
                recipe_env_bindings=recipe_env_bindings,
            )
        )
        validate_setup_mode_flags(setup_mode, minimal, readonly)
        apply_runtime_mode_env_vars(
            env_vars,
            minimal,
            readonly,
            setup_mode,
            platform,
            algorithm,
            log_level=log_level,
        )
        if resolved_target == "docker":
            configure_runtime_override_env_vars(
                env_vars,
                config_path,
                effective_config_path,
            )
        _deploy_serve_backend(
            resolved_target=resolved_target,
            config_path=config_path,
            effective_config_path=effective_config_path,
            effective_config_document=effective_config_document,
            runtime_lock=runtime_lock,
            env_vars=env_vars,
            namespace=namespace,
            context=context,
            profile=profile,
            chart_dir=chart_dir,
            image=image,
            router_image=router_image,
            envoy_image=envoy_image,
            dashboard_image=dashboard_image,
            image_pull_policy=image_pull_policy,
            minimal=minimal,
            readonly=readonly,
        )
    finally:
        if runtime_lock is not None:
            runtime_lock.close()


@click.command(help=SERVE_HELP)
@click.option(
    "--config",
    default="config.yaml",
    show_default=True,
    help="Path to the Router configuration.",
)
@click.option(
    "--image",
    default=None,
    help=f"Docker image to use (default: {VLLM_SR_CONTAINER_IMAGE_DEFAULT})",
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
    help="Platform for local Docker GPU deployments: 'amd' enables ROCm passthrough, "
    "'nvidia' enables NVIDIA GPU passthrough (--gpus all). "
    "When set to amd or nvidia, serve defaults to the matching GPU image "
    "(ROCm / CUDA) and flips use_cpu to false for router internal models under "
    "global.model_catalog, unless --image or VLLM_SR_IMAGE is provided. "
    "Set VLLM_SR_<PLATFORM>_PRESERVE_CPU=1 to keep CPU settings. "
    "For Kubernetes, configure GPU images and resources through a Helm profile "
    "or the operator.",
)
@click.option(
    "--algorithm",
    type=click.Choice(ALGORITHM_TYPES, case_sensitive=False),
    default=None,
    help="Request-time base algorithm override: static, router_dc, automix, hybrid, "
    "workflows, latency_aware, knn, kmeans, svm, mlp, or multi_factor. "
    "Cross-request learning uses global.router.learning.adaptation/protection.",
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
@click.option(
    "--runtime",
    type=click.Choice(SUPPORTED_CONTAINER_RUNTIMES, case_sensitive=False),
    default=None,
    help=RUNTIME_HELP,
)
@click.option(
    "--recipe-env",
    "recipe_env_names",
    multiple=True,
    metavar="NAME",
    help=(
        "Explicitly bind one host environment variable for the active Recipe. "
        "Repeat for multiple names; NAME=value is rejected."
    ),
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
    runtime: str | None,
    recipe_env_names: tuple[str, ...],
) -> None:
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
        runtime,
        recipe_env_names,
    )


@click.command()
@click.argument(
    "service",
    type=click.Choice(["envoy", "router", "dashboard", "all"]),
    default="all",
)
@click.option("--target", default=None, help=TARGET_HELP)
@click.option(
    "--namespace", default=None, help="Kubernetes namespace (k8s target only)"
)
@click.option(
    "--context", default=None, help="kubectl / Helm context (k8s target only)"
)
@click.option(
    "--runtime",
    type=click.Choice(SUPPORTED_CONTAINER_RUNTIMES, case_sensitive=False),
    default=None,
    help=RUNTIME_HELP,
)
@exit_with_logged_error(log)
def status(
    service: str,
    target: str | None,
    namespace: str | None,
    context: str | None,
    runtime: str | None,
) -> None:
    """
    Show status of vLLM Semantic Router services.

    Examples:
        vllm-sr status              # Show all services (Docker)
        vllm-sr status all          # Show all services
        vllm-sr status router       # Show router status
        vllm-sr status dashboard    # Show dashboard status
        vllm-sr status --target k8s # Show Kubernetes status
    """
    apply_container_runtime_override(runtime)
    backend = _build_backend(target, namespace=namespace, context=context)
    backend.status(service)


@click.command()
@click.argument("service", type=click.Choice(["envoy", "router", "dashboard"]))
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
@click.option("--target", default=None, help=TARGET_HELP)
@click.option(
    "--namespace", default=None, help="Kubernetes namespace (k8s target only)"
)
@click.option(
    "--context", default=None, help="kubectl / Helm context (k8s target only)"
)
@click.option(
    "--runtime",
    type=click.Choice(SUPPORTED_CONTAINER_RUNTIMES, case_sensitive=False),
    default=None,
    help=RUNTIME_HELP,
)
@exit_with_logged_error(log, interrupt_message="\nLog streaming stopped")
def logs(
    service: str,
    follow: bool,
    target: str | None,
    namespace: str | None,
    context: str | None,
    runtime: str | None,
) -> None:
    """
    Show logs from vLLM Semantic Router service.

    Examples:
        vllm-sr logs envoy
        vllm-sr logs router
        vllm-sr logs dashboard
        vllm-sr logs envoy --follow
        vllm-sr logs router -f
        vllm-sr logs router --target k8s        # Kubernetes logs
        vllm-sr logs router --target k8s -f     # Follow K8s logs
    """
    apply_container_runtime_override(runtime)
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
@click.option(
    "--runtime",
    type=click.Choice(SUPPORTED_CONTAINER_RUNTIMES, case_sensitive=False),
    default=None,
    help=RUNTIME_HELP,
)
@exit_with_logged_error(log)
def stop(
    target: str | None,
    namespace: str | None,
    context: str | None,
    runtime: str | None,
) -> None:
    """
    Stop vLLM Semantic Router.

    Examples:
        vllm-sr stop                # Stop Docker stack
        vllm-sr stop --target k8s   # Uninstall Helm release
    """
    apply_container_runtime_override(runtime)
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
@click.option(
    "--runtime",
    type=click.Choice(SUPPORTED_CONTAINER_RUNTIMES, case_sensitive=False),
    default=None,
    help=RUNTIME_HELP,
)
@exit_with_logged_error(log)
def dashboard(
    no_open: bool,
    target: str | None,
    namespace: str | None,
    context: str | None,
    runtime: str | None,
) -> None:
    """
    Open the dashboard in your default web browser.

    Examples:
        vllm-sr dashboard                   # Docker dashboard
        vllm-sr dashboard --target k8s      # Show K8s dashboard URL
        vllm-sr dashboard --no-open
    """
    apply_container_runtime_override(runtime)
    backend = _build_backend(target, namespace=namespace, context=context)
    if not backend.is_running():
        raise ValueError("vLLM Semantic Router is not running")

    dashboard_url = backend.get_dashboard_url()
    if dashboard_url is None:
        raise ValueError("Dashboard URL could not be determined")

    if no_open:
        heading("Dashboard")
        fields((("URL", dashboard_url),))
        return

    if not webbrowser.open(dashboard_url):
        raise ValueError(
            f"Could not open a browser. Open {dashboard_url} manually or use --no-open."
        )
    success("Dashboard opened in your browser")
    fields((("URL", dashboard_url),))
