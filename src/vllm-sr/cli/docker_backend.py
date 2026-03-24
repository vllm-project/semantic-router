"""Docker deployment backend — wraps the existing container-based workflow."""

from __future__ import annotations

from typing import Any

from cli.core import show_logs, show_status, start_vllm_sr, stop_vllm_sr
from cli.docker_cli import docker_container_status
from cli.runtime_stack import resolve_runtime_stack
from cli.utils import get_logger

log = get_logger(__name__)


class DockerBackend:
    """DeploymentBackend implementation for local Docker / Podman."""

    def deploy(
        self,
        config_file: str,
        env_vars: dict[str, str] | None = None,
        *,
        image: str | None = None,
        pull_policy: str | None = None,
        enable_observability: bool = True,
        **kwargs: Any,
    ) -> None:
        start_vllm_sr(
            config_file,
            env_vars=env_vars,
            image=image,
            pull_policy=pull_policy,
            enable_observability=enable_observability,
        )

    def teardown(self) -> None:
        stop_vllm_sr()

    def logs(self, service: str, follow: bool = False) -> None:
        show_logs(service, follow=follow)

    def status(self, service: str = "all") -> None:
        show_status(service)

    def get_dashboard_url(self) -> str | None:
        stack_layout = resolve_runtime_stack()
        if docker_container_status(stack_layout.container_name) != "running":
            return None
        return stack_layout.dashboard_url

    def is_running(self) -> bool:
        stack_layout = resolve_runtime_stack()
        return docker_container_status(stack_layout.container_name) == "running"
