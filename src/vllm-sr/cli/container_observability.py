"""Observability template and service-start helpers for local containers."""

import os
import subprocess
from pathlib import Path

from cli.runtime_stack import RuntimeStackLayout
from cli.utils import get_logger

log = get_logger(__name__)


def _ensure_hidden_config_dir(config_dir):
    if config_dir is None:
        config_dir = os.path.join(os.getcwd(), ".vllm-sr")
    else:
        config_dir = os.path.join(config_dir, ".vllm-sr")
    os.makedirs(config_dir, exist_ok=True)
    return config_dir


def _render_template_copy(
    source_path: str, destination_path: str, stack_layout: RuntimeStackLayout
) -> None:
    source = Path(source_path)
    destination = Path(destination_path)
    rendered = render_observability_template(source.read_text(), stack_layout)
    destination.write_text(rendered, encoding="utf-8")


def render_observability_template(
    template_text: str, stack_layout: RuntimeStackLayout
) -> str:
    replacements = (
        ("vllm-sr-container:9190", f"{stack_layout.router_container_name}:9190"),
        ("vllm-sr-container", stack_layout.router_container_name),
        ("vllm-sr-prometheus", stack_layout.prometheus_container_name),
        ("vllm-sr-jaeger", stack_layout.jaeger_container_name),
    )
    rendered = template_text
    for source, target in replacements:
        rendered = rendered.replace(source, target)
    return rendered


def _run_service_start(cmd, service_name):
    log.info(f"Starting {service_name} container...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return (0, result.stdout, result.stderr)
    except subprocess.CalledProcessError as exc:
        return (exc.returncode, exc.stdout, exc.stderr)
