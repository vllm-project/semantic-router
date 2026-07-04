"""Docker CLI operations for vLLM Semantic Router."""

from cli.container_images import get_container_image, get_fleet_sim_container_image
from cli.container_runtime import (
    container_image_exists,
    container_pull_image,
    get_container_runtime,
)
from cli.container_services import (
    container_create_network,
    container_exec,
    container_logs,
    container_logs_since,
    container_network_connect,
    container_network_disconnect,
    container_remove_container,
    container_remove_network,
    container_start_container,
    container_start_fleet_sim,
    container_start_grafana,
    container_start_jaeger,
    container_start_postgres,
    container_start_prometheus,
    container_start_redis,
    container_status,
    container_stop_container,
    load_openclaw_registry,
)
from cli.container_start import container_start_vllm_sr

__all__ = [
    "container_create_network",
    "container_exec",
    "container_image_exists",
    "container_logs",
    "container_logs_since",
    "container_network_connect",
    "container_network_disconnect",
    "container_pull_image",
    "container_remove_container",
    "container_remove_network",
    "container_start_container",
    "container_start_fleet_sim",
    "container_start_grafana",
    "container_start_jaeger",
    "container_start_postgres",
    "container_start_prometheus",
    "container_start_redis",
    "container_start_vllm_sr",
    "container_status",
    "container_stop_container",
    "get_container_image",
    "get_container_runtime",
    "get_fleet_sim_container_image",
    "load_openclaw_registry",
]
