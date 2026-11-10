"""Every image the CLI pulls by default names its registry.

Podman without unqualified-search registries, as Debian and Ubuntu package it,
cannot pull a short name such as `envoyproxy/envoy:v1.35.3`, while Docker
resolves `docker.io/envoyproxy/envoy:v1.35.3` to the same image. Default images
are module-level constants named `*_IMAGE`, `*_IMAGE_DEFAULT`, `*_IMAGE_ROCM` or
`*_IMAGE_CUDA` anywhere in the CLI package, so the scan finds a new one without
a list to update.
"""

import ast
import importlib
import re
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "cli"
IMAGE_CONSTANT = re.compile(r"[A-Z0-9_]+_IMAGE(?:_DEFAULT|_ROCM|_CUDA)?")


def _image_constants() -> list[tuple[str, str]]:
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            found += [
                (module, target.id)
                for target in targets
                if isinstance(target, ast.Name) and IMAGE_CONSTANT.fullmatch(target.id)
            ]
    return found


def _names_a_registry(image: str) -> bool:
    first, slash, _rest = image.partition("/")
    return bool(slash) and ("." in first or ":" in first or first == "localhost")


@pytest.mark.parametrize("module, name", _image_constants())
def test_every_default_image_names_its_registry(module: str, name: str):
    image = getattr(importlib.import_module(module), name)

    assert isinstance(image, str), f"{module}.{name} is not an image reference"
    assert _names_a_registry(image), f"{module}.{name} = {image!r} is a short name"


def test_the_scan_sees_every_image_a_default_stack_pulls():
    names = {name for _module, name in _image_constants()}

    assert {
        "VLLM_SR_CONTAINER_IMAGE_DEFAULT",
        "VLLM_SR_CONTAINER_IMAGE_ROCM",
        "VLLM_SR_CONTAINER_IMAGE_CUDA",
        "VLLM_SR_ENVOY_CONTAINER_IMAGE_DEFAULT",
        "VLLM_SR_DASHBOARD_CONTAINER_IMAGE_DEFAULT",
        "REDIS_IMAGE",
        "POSTGRES_IMAGE",
        "MILVUS_IMAGE",
        "JAEGER_IMAGE",
        "PROMETHEUS_IMAGE",
        "GRAFANA_IMAGE",
        "GRADER_BASE_IMAGE",
    } <= names


@pytest.mark.parametrize(
    "image, qualified",
    [
        ("docker.io/envoyproxy/envoy:v1.35.3", True),
        ("ghcr.io/vllm-project/semantic-router/vllm-sr:latest", True),
        ("localhost/vllm-sr:dev", True),
        ("localhost:5000/vllm-sr:dev", True),
        ("envoyproxy/envoy:v1.35.3", False),
        ("python:3.12-slim-bookworm", False),
    ],
)
def test_a_registry_is_told_apart_from_a_short_name(image: str, qualified: bool):
    assert _names_a_registry(image) is qualified
