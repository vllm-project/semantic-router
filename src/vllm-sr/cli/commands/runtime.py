"""Runtime-oriented Click command entrypoints."""

from __future__ import annotations

import os
import webbrowser
from pathlib import Path

import click
import yaml

from cli.bootstrap import ensure_bootstrap_workspace
from cli.commands.common import exit_with_logged_error
from cli.commands.runtime_help import SERVE_HELP
from cli.commands.runtime_paths import (
    _compiled_bootstrap_output_path,
    assert_user_bootstrap_source,
    materialize_compiled_bootstrap,
)
from cli.commands.runtime_support import (
    append_passthrough_env_vars,
    apply_container_runtime_override,
    apply_runtime_mode_env_vars,
    build_effective_config_bytes,
    build_effective_config_document,
    log_bootstrap_result,
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
from cli.runtime_config_lock import acquire_compiled_bootstrap_lock
from cli.runtime_stack import resolve_runtime_stack
from cli.terminal import fields, heading, success
from cli.utils import get_logger

log = get_logger(__name__)


TARGET_HELP = (
    f"Deployment target: {', '.join(VALID_TARGETS)} (default: {DEFAULT_TARGET})"
)
DEFAULT_SERVE_CONFIG = "config.yaml"

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


def _prepare_docker_compiled_bootstrap(
    config_path: Path,
    platform: str | None,
    *,
    state_root_dir: Path | None = None,
):
    stack_layout = resolve_runtime_stack()
    state_root_dir = state_root_dir or (
        Path(os.environ["VLLM_SR_STATE_ROOT_DIR"]).expanduser().absolute()
        if os.getenv("VLLM_SR_STATE_ROOT_DIR", "").strip()
        else config_path.expanduser().absolute().parent
    )
    effective_config_path = _compiled_bootstrap_output_path(
        config_path,
        state_root_dir=state_root_dir,
        stack_name=stack_layout.stack_name,
    )
    bootstrap_lock = acquire_compiled_bootstrap_lock(
        compiled_bootstrap_path=effective_config_path,
        state_root_dir=state_root_dir,
        stack_name=stack_layout.stack_name,
        timeout_seconds=0,
    )
    try:
        effective_config_bytes = build_effective_config_bytes(config_path, platform)
        effective_config_path = materialize_compiled_bootstrap(
            config_path,
            effective_config_bytes,
            state_root_dir=state_root_dir,
            stack_name=stack_layout.stack_name,
        )
        return effective_config_path, bootstrap_lock
    except Exception:
        bootstrap_lock.close()
        raise


def _resolve_serve_config(
    config: str,
    resolved_target: str,
) -> Path:
    """Resolve the explicit bootstrap config for one deployment target."""

    requested_config = assert_user_bootstrap_source(Path(config))
    if resolved_target != "docker":
        config_path = requested_config
        if not config_path.is_file():
            raise ValueError(
                "Kubernetes deployment requires a complete ./config.yaml; "
                "automatic first-run workspace creation is available only for Docker"
            )
        try:
            document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(
                "Kubernetes deployment requires a valid YAML config"
            ) from exc
        if isinstance(document, dict) and document.get("setup") is not None:
            raise ValueError(
                "Kubernetes deployment does not support Dashboard setup-mode configs; "
                "provide a complete Router config and explicit Kubernetes Secrets"
            )
        return config_path
    state_root_dir = (
        Path(os.environ["VLLM_SR_STATE_ROOT_DIR"]).expanduser().absolute()
        if os.getenv("VLLM_SR_STATE_ROOT_DIR", "").strip()
        else requested_config.parent
    )
    bootstrap = ensure_bootstrap_workspace(
        requested_config,
        state_root_dir=state_root_dir,
        stack_layout=resolve_runtime_stack(),
    )
    log_bootstrap_result(config, bootstrap)
    return assert_user_bootstrap_source(bootstrap.config_path)


def _prepare_effective_serve_config(
    config_path: Path,
    *,
    resolved_target: str,
    platform: str | None,
):
    """Prepare the target-specific config and its optional runtime lock."""

    if resolved_target == "docker":
        effective_path, bootstrap_lock = _prepare_docker_compiled_bootstrap(
            config_path,
            platform,
        )
        return effective_path, bootstrap_lock, None

    # Kubernetes is an in-memory translation flow and never publishes local
    # compiled-bootstrap staging.
    effective_config_document = build_effective_config_document(
        config_path,
        platform,
        materialize_local_runtime=False,
    )
    return config_path, None, effective_config_document


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
    compiled_bootstrap_lock,
    env_vars: dict[str, str],
    namespace: str | None,
    context: str | None,
    profile: str | None,
    chart_dir: str | None,
    image: str | None,
    router_image: str | None,
    envoy_image: str | None,
    dashboard_image: str | None,
    sim_image: str | None,
    image_pull_policy: str,
    minimal: bool,
    with_observability: bool,
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
        compiled_bootstrap_file=str(effective_config_path.absolute()),
        compiled_bootstrap_lock=compiled_bootstrap_lock,
        config_document=effective_config_document,
        env_vars=env_vars,
        image=image,
        router_image=router_image,
        envoy_image=envoy_image,
        dashboard_image=dashboard_image,
        sim_image=sim_image,
        pull_policy=image_pull_policy,
        enable_observability=with_observability and not minimal,
        minimal=minimal,
        readonly=readonly,
    )


def _execute_serve(
    image: str | None,
    router_image: str | None,
    envoy_image: str | None,
    dashboard_image: str | None,
    sim_image: str | None,
    image_pull_policy: str,
    readonly: bool,
    minimal: bool,
    with_observability: bool,
    log_level: str | None,
    platform: str | None,
    target: str | None,
    namespace: str | None,
    context: str | None,
    profile: str | None,
    chart_dir: str | None,
    runtime: str | None,
    config: str | None,
) -> None:
    """Bootstrap workspace, resolve config, and delegate to the deployment backend."""
    resolved_target = resolve_target(target)
    _validate_target_platform(resolved_target, platform)
    apply_container_runtime_override(runtime)
    selected_config = config or DEFAULT_SERVE_CONFIG
    if config is not None and not Path(selected_config).expanduser().is_file():
        raise ValueError(f"bootstrap config file does not exist: {selected_config}")
    config_path = _resolve_serve_config(selected_config, resolved_target)
    log.info(f"Using config file: {config_path}")

    env_vars: dict[str, str] = {}
    append_passthrough_env_vars(env_vars, config_path)
    compiled_bootstrap_lock = None
    try:
        effective_config_path, compiled_bootstrap_lock, effective_config_document = (
            _prepare_effective_serve_config(
                config_path,
                resolved_target=resolved_target,
                platform=platform,
            )
        )
        apply_runtime_mode_env_vars(
            env_vars,
            minimal,
            readonly,
            platform,
            log_level=log_level,
        )
        _deploy_serve_backend(
            resolved_target=resolved_target,
            config_path=config_path,
            effective_config_path=effective_config_path,
            effective_config_document=effective_config_document,
            compiled_bootstrap_lock=compiled_bootstrap_lock,
            env_vars=env_vars,
            namespace=namespace,
            context=context,
            profile=profile,
            chart_dir=chart_dir,
            image=image,
            router_image=router_image,
            envoy_image=envoy_image,
            dashboard_image=dashboard_image,
            sim_image=sim_image,
            image_pull_policy=image_pull_policy,
            minimal=minimal,
            with_observability=with_observability,
            readonly=readonly,
        )
    finally:
        if compiled_bootstrap_lock is not None:
            compiled_bootstrap_lock.close()


@click.command(help=SERVE_HELP)
@click.option(
    "--config",
    default=None,
    metavar="PATH",
    help=(
        "Canonical v0.3 config (default: ./config.yaml; a missing default creates "
        "a secure local Management workspace for Docker)."
    ),
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
    "--sim-image",
    default=None,
    help="Docker image for the simulator sidecar (Docker target only; defaults to VLLM_SR_SIM_IMAGE)",
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
    help=(
        "Omit Dashboard and optional observability. Configured Management and "
        "runtime stores still start when external endpoints are not provided."
    ),
)
@click.option(
    "--with-observability",
    is_flag=True,
    default=False,
    help="Also start Prometheus, Grafana, and Jaeger.",
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
@exit_with_logged_error(log, interrupt_message="\nInterrupted by user")
def serve(
    image: str | None,
    router_image: str | None,
    envoy_image: str | None,
    dashboard_image: str | None,
    sim_image: str | None,
    image_pull_policy: str,
    readonly: bool,
    minimal: bool,
    with_observability: bool,
    log_level: str | None,
    platform: str | None,
    target: str | None,
    namespace: str | None,
    context: str | None,
    profile: str | None,
    chart_dir: str | None,
    runtime: str | None,
    config: str | None,
) -> None:
    _execute_serve(
        image,
        router_image,
        envoy_image,
        dashboard_image,
        sim_image,
        image_pull_policy,
        readonly,
        minimal,
        with_observability,
        log_level,
        platform,
        target,
        namespace,
        context,
        profile,
        chart_dir,
        runtime,
        config,
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
        vllm-sr status simulator    # Show simulator status
        vllm-sr status --target k8s # Show Kubernetes status
    """
    apply_container_runtime_override(runtime)
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
        vllm-sr logs simulator
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
