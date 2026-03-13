"""Derived names and host ports for the local vLLM-SR runtime stack."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from cli.consts import (
    DEFAULT_API_PORT,
    DEFAULT_DASHBOARD_PORT,
    DEFAULT_METRICS_PORT,
    DEFAULT_ROUTER_PORT,
    DEFAULT_STACK_NAME,
    VLLM_SR_DOCKER_NAME,
)

STACK_NAME_ENV = "VLLM_SR_STACK_NAME"
PORT_OFFSET_ENV = "VLLM_SR_PORT_OFFSET"

DEFAULT_JAEGER_OTLP_PORT = 4318
DEFAULT_JAEGER_UI_PORT = 16686
DEFAULT_PROMETHEUS_PORT = 9090
DEFAULT_GRAFANA_PORT = 3000

STACK_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class RuntimeStackLayout:
    stack_name: str
    port_offset: int
    container_name: str
    network_name: str
    jaeger_container_name: str
    prometheus_container_name: str
    grafana_container_name: str
    router_port: int
    metrics_port: int
    dashboard_port: int
    api_port: int
    jaeger_otlp_port: int
    jaeger_ui_port: int
    prometheus_port: int
    grafana_port: int

    @property
    def dashboard_url(self) -> str:
        return f"http://localhost:{self.dashboard_port}"

    @property
    def metrics_url(self) -> str:
        return f"http://localhost:{self.metrics_port}/metrics"

    @property
    def jaeger_ui_url(self) -> str:
        return f"http://localhost:{self.jaeger_ui_port}"

    @property
    def grafana_url(self) -> str:
        return f"http://localhost:{self.grafana_port}"

    @property
    def prometheus_url(self) -> str:
        return f"http://localhost:{self.prometheus_port}"

    @property
    def jaeger_service_url(self) -> str:
        return f"http://{self.jaeger_container_name}:16686"

    @property
    def prometheus_service_url(self) -> str:
        return f"http://{self.prometheus_container_name}:9090"

    @property
    def grafana_service_url(self) -> str:
        return f"http://{self.grafana_container_name}:3000"

    @property
    def otlp_service_endpoint(self) -> str:
        return f"http://{self.jaeger_container_name}:4317"


def resolve_runtime_stack(
    *, stack_name: str | None = None, port_offset: int | None = None
) -> RuntimeStackLayout:
    resolved_stack_name = normalize_stack_name(
        stack_name if stack_name is not None else os.getenv(STACK_NAME_ENV)
    )
    resolved_port_offset = normalize_port_offset(
        port_offset if port_offset is not None else os.getenv(PORT_OFFSET_ENV)
    )

    if resolved_stack_name == DEFAULT_STACK_NAME:
        container_name = VLLM_SR_DOCKER_NAME
        network_name = f"{DEFAULT_STACK_NAME}-network"
        jaeger_container_name = f"{DEFAULT_STACK_NAME}-jaeger"
        prometheus_container_name = f"{DEFAULT_STACK_NAME}-prometheus"
        grafana_container_name = f"{DEFAULT_STACK_NAME}-grafana"
    else:
        container_name = f"{resolved_stack_name}-vllm-sr-container"
        network_name = f"{resolved_stack_name}-vllm-sr-network"
        jaeger_container_name = f"{resolved_stack_name}-vllm-sr-jaeger"
        prometheus_container_name = f"{resolved_stack_name}-vllm-sr-prometheus"
        grafana_container_name = f"{resolved_stack_name}-vllm-sr-grafana"

    return RuntimeStackLayout(
        stack_name=resolved_stack_name,
        port_offset=resolved_port_offset,
        container_name=container_name,
        network_name=network_name,
        jaeger_container_name=jaeger_container_name,
        prometheus_container_name=prometheus_container_name,
        grafana_container_name=grafana_container_name,
        router_port=DEFAULT_ROUTER_PORT + resolved_port_offset,
        metrics_port=DEFAULT_METRICS_PORT + resolved_port_offset,
        dashboard_port=DEFAULT_DASHBOARD_PORT + resolved_port_offset,
        api_port=DEFAULT_API_PORT + resolved_port_offset,
        jaeger_otlp_port=DEFAULT_JAEGER_OTLP_PORT + resolved_port_offset,
        jaeger_ui_port=DEFAULT_JAEGER_UI_PORT + resolved_port_offset,
        prometheus_port=DEFAULT_PROMETHEUS_PORT + resolved_port_offset,
        grafana_port=DEFAULT_GRAFANA_PORT + resolved_port_offset,
    )


def normalize_stack_name(raw_value: str | None) -> str:
    if raw_value is None:
        return DEFAULT_STACK_NAME
    cleaned = STACK_NAME_PATTERN.sub("-", raw_value.strip()).strip("-")
    if not cleaned:
        return DEFAULT_STACK_NAME
    return cleaned


def normalize_port_offset(raw_value: str | int | None) -> int:
    if raw_value in (None, ""):
        return 0
    offset = int(raw_value)
    if offset < 0:
        raise ValueError(f"{PORT_OFFSET_ENV} must be >= 0, got {offset}")
    return offset
