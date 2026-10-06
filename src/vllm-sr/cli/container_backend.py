"""Docker deployment backend — wraps the existing container-based workflow."""

from __future__ import annotations

from typing import Any

import os

from cli import apple_runtime

from cli.consts import HEALTH_CHECK_TIMEOUT
from cli.container_cli import container_status
from cli.container_runtime import get_container_runtime
from cli.core import show_logs, show_status, start_vllm_sr, stop_vllm_sr
from cli.gateway_mode import GATEWAY_EXTPROC, runs_envoy
from cli.instance_setup import (
    attach_controller,
    stop_managed_controller,
)
from cli.runtime_lifecycle import validate_startup_timeout
from cli.runtime_lifecycle_lock import acquire_runtime_lifecycle_lock
from cli.runtime_stack import resolve_runtime_stack
from cli.utils import get_logger

log = get_logger(__name__)


class ContainerBackend:
    """Local Docker deployment backend."""

    def deploy(
        self,
        config_file: str,
        env_vars: dict[str, str] | None = None,
        *,
        source_config_file: str | None = None,
        runtime_config_file: str | None = None,
        image: str | None = None,
        router_image: str | None = None,
        envoy_image: str | None = None,
        dashboard_image: str | None = None,
        topology: str | None = None,
        pull_policy: str | None = None,
        enable_observability: bool = True,
        runtime_config_lock: Any = None,
        startup_timeout: int = HEALTH_CHECK_TIMEOUT,
        gateway: str = GATEWAY_EXTPROC,
        **kwargs: Any,
    ) -> None:
        validate_startup_timeout(startup_timeout)
        if source_config_file is None:
            source_config_file = kwargs.get("source_config_file")
        if runtime_config_file is None:
            runtime_config_file = kwargs.get("runtime_config_file")
        with self._lifecycle_lock():
            apple = (
                (env_vars or {}).get("VLLM_SR_PLATFORM")
                or os.getenv("VLLM_SR_PLATFORM")
                or os.getenv("DASHBOARD_PLATFORM")
                or ""
            ).strip().lower() == "apple"
            if apple:
                from cli.container_images import get_runtime_images

                images = get_runtime_images(
                    image=image,
                    router_image=router_image,
                    envoy_image=envoy_image,
                    dashboard_image=dashboard_image,
                    pull_policy=pull_policy,
                    platform="apple",
                    include_envoy=runs_envoy(gateway),
                    include_dashboard=(env_vars or {}).get("DISABLE_DASHBOARD")
                    != "true",
                )
                state = apple_runtime.start_bridge(images["router"])
                router_image = state["image"]
                envoy_image = images.get("envoy")
                dashboard_image = images.get("dashboard")
                pull_policy = "never"
                os.environ[apple_runtime.ENDPOINT_ENV] = (
                    f"http://host.docker.internal:{state['port']}"
                )
                os.environ[apple_runtime.TOKEN_ENV] = state["token"]
            else:
                apple_runtime.stop_bridge()
            try:
                start_vllm_sr(
                    config_file,
                    env_vars=env_vars,
                    source_config_file=source_config_file,
                    runtime_config_file=runtime_config_file,
                    image=image,
                    router_image=router_image,
                    envoy_image=envoy_image,
                    dashboard_image=dashboard_image,
                    topology=topology,
                    pull_policy=pull_policy,
                    enable_observability=enable_observability,
                    runtime_config_lock=runtime_config_lock,
                    startup_timeout=startup_timeout,
                    gateway=gateway,
                )
            except BaseException:
                if apple:
                    apple_runtime.stop_bridge()
                raise
            finally:
                if apple:
                    os.environ.pop(apple_runtime.ENDPOINT_ENV, None)
                    os.environ.pop(apple_runtime.TOKEN_ENV, None)

        attach_controller(
            source_config_file or config_file,
            runtime_config_file or config_file,
            env_vars,
            gateway,
            startup_timeout,
        )

    def teardown(self) -> None:
        with self._lifecycle_lock():
            try:
                stop_managed_controller()
                stop_vllm_sr()
            finally:
                apple_runtime.stop_bridge()

    @staticmethod
    def _lifecycle_lock():
        stack_layout = resolve_runtime_stack()
        return acquire_runtime_lifecycle_lock(
            runtime=get_container_runtime(),
            stack_name=stack_layout.stack_name,
        )

    def logs(self, service: str, follow: bool = False) -> None:
        if service == "model-runtime":
            apple_runtime.bridge_logs(follow)
            return
        show_logs(service, follow=follow)

    def status(self, service: str = "all") -> None:
        if service == "model-runtime":
            apple_runtime.bridge_status()
            return
        if service == "all" and apple_runtime.read_state() is not None:
            apple_runtime.bridge_status()
        show_status(service)

    def get_dashboard_url(self) -> str | None:
        stack_layout = resolve_runtime_stack()
        if container_status(stack_layout.dashboard_container_name) == "running":
            return stack_layout.dashboard_url
        return None

    def is_running(self) -> bool:
        stack_layout = resolve_runtime_stack()
        return any(
            container_status(container_name) == "running"
            for container_name in stack_layout.runtime_container_names
        )
