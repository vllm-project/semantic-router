"""Environment assembly and process isolation for container startup."""

import os
from collections.abc import Mapping

from cli.runtime_stack import PORT_OFFSET_ENV, RuntimeStackLayout

DASHBOARD_SECRET_ENV_NAMES = frozenset(
    {"DASHBOARD_ADMIN_PASSWORD", "DASHBOARD_JWT_SECRET"}
)


def _build_common_runtime_env(
    env_vars: dict[str, str],
    stack_layout: RuntimeStackLayout,
) -> dict[str, str]:
    common_env = dict(env_vars or {})
    common_env["VLLM_SR_STATE_ROOT_DIR"] = "/app"
    common_env["VLLM_SR_CONFIG_BASE_DIR"] = "/app"
    common_env[PORT_OFFSET_ENV] = str(stack_layout.port_offset)
    stack_name_value = os.getenv("VLLM_SR_STACK_NAME", "").strip()
    if stack_name_value:
        common_env.setdefault("VLLM_SR_STACK_NAME", stack_name_value)
    common_env.setdefault(
        "VLLM_SR_ROUTER_CONTAINER_NAME", stack_layout.router_container_name
    )
    common_env.setdefault(
        "VLLM_SR_ENVOY_CONTAINER_NAME", stack_layout.envoy_container_name
    )
    common_env.setdefault(
        "VLLM_SR_DASHBOARD_CONTAINER_NAME", stack_layout.dashboard_container_name
    )
    return common_env


def _dashboard_secret_environment(
    common_env: Mapping[str, str], dashboard_runtime_env: Mapping[str, str]
) -> dict[str, str]:
    return {
        name: value
        for name in DASHBOARD_SECRET_ENV_NAMES
        if (
            value := (
                common_env.get(name, "").strip()
                or os.environ.get(name, "").strip()
                or dashboard_runtime_env.get(name, "").strip()
            )
        )
    }


def service_child_environment(
    service_name: str,
    router_secret_values: Mapping[str, str],
    dashboard_secret_values: Mapping[str, str],
) -> dict[str, str] | None:
    """Build one child environment without leaking secrets across services."""

    blocked = set(router_secret_values) | set(dashboard_secret_values)
    if not blocked:
        return None
    environment = {
        name: value for name, value in os.environ.items() if name not in blocked
    }
    if service_name == "router":
        environment.update(router_secret_values)
    elif service_name == "dashboard":
        environment.update(dashboard_secret_values)
    return environment
