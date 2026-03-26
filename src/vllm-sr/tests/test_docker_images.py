import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli import docker_images  # noqa: E402
from cli.consts import (  # noqa: E402
    VLLM_SR_DASHBOARD_DOCKER_IMAGE_DEFAULT,
    VLLM_SR_DOCKER_IMAGE_DEFAULT,
    VLLM_SR_DOCKER_IMAGE_ROCM,
    VLLM_SR_ENVOY_DOCKER_IMAGE_DEFAULT,
)


@pytest.fixture(autouse=True)
def clear_runtime_image_env(monkeypatch):
    for env_name in (
        "VLLM_SR_IMAGE",
        "VLLM_SR_ROUTER_IMAGE",
        "VLLM_SR_ENVOY_IMAGE",
        "VLLM_SR_DASHBOARD_IMAGE",
        "VLLM_SR_PLATFORM",
    ):
        monkeypatch.delenv(env_name, raising=False)


def test_get_runtime_images_falls_back_to_base_image(monkeypatch):
    ensured = []
    monkeypatch.setattr(
        docker_images,
        "_resolve_selected_image",
        lambda image, normalized_platform: "base:latest",
    )
    monkeypatch.setattr(
        docker_images,
        "_ensure_image_available",
        lambda image, pull_policy: ensured.append((image, pull_policy)),
    )

    images = docker_images.get_runtime_images(pull_policy="never")

    assert images == {
        "router": "base:latest",
        "envoy": "base:latest",
        "dashboard": "base:latest",
    }
    assert ensured == [("base:latest", "never")]


def test_get_runtime_images_prefers_service_specific_env_overrides(monkeypatch):
    ensured = []
    monkeypatch.setattr(
        docker_images,
        "_resolve_selected_image",
        lambda image, normalized_platform: "base:latest",
    )
    monkeypatch.setattr(
        docker_images,
        "_ensure_image_available",
        lambda image, pull_policy: ensured.append((image, pull_policy)),
    )
    monkeypatch.setenv("VLLM_SR_ENVOY_IMAGE", "envoy:separate")

    images = docker_images.get_runtime_images(pull_policy="ifnotpresent")

    assert images == {
        "router": "base:latest",
        "envoy": "envoy:separate",
        "dashboard": "base:latest",
    }
    assert ensured == [
        ("base:latest", "ifnotpresent"),
        ("envoy:separate", "ifnotpresent"),
    ]


def test_get_runtime_images_derives_official_envoy_image_from_official_base(
    monkeypatch,
):
    ensured = []
    monkeypatch.setattr(
        docker_images,
        "_resolve_selected_image",
        lambda image, normalized_platform: VLLM_SR_DOCKER_IMAGE_DEFAULT,
    )
    monkeypatch.setattr(
        docker_images,
        "_ensure_image_available",
        lambda image, pull_policy: ensured.append((image, pull_policy)),
    )

    images = docker_images.get_runtime_images(pull_policy="never")

    assert images == {
        "router": VLLM_SR_DOCKER_IMAGE_DEFAULT,
        "envoy": VLLM_SR_ENVOY_DOCKER_IMAGE_DEFAULT,
        "dashboard": VLLM_SR_DASHBOARD_DOCKER_IMAGE_DEFAULT,
    }
    assert ensured == [
        (VLLM_SR_DOCKER_IMAGE_DEFAULT, "never"),
        (VLLM_SR_ENVOY_DOCKER_IMAGE_DEFAULT, "never"),
        (VLLM_SR_DASHBOARD_DOCKER_IMAGE_DEFAULT, "never"),
    ]


def test_get_runtime_images_derives_official_envoy_image_from_rocm_base(monkeypatch):
    ensured = []
    monkeypatch.setattr(
        docker_images,
        "_resolve_selected_image",
        lambda image, normalized_platform: VLLM_SR_DOCKER_IMAGE_ROCM,
    )
    monkeypatch.setattr(
        docker_images,
        "_ensure_image_available",
        lambda image, pull_policy: ensured.append((image, pull_policy)),
    )

    images = docker_images.get_runtime_images(pull_policy="never", platform="amd")

    assert images == {
        "router": VLLM_SR_DOCKER_IMAGE_ROCM,
        "envoy": VLLM_SR_ENVOY_DOCKER_IMAGE_DEFAULT,
        "dashboard": VLLM_SR_DASHBOARD_DOCKER_IMAGE_DEFAULT,
    }
    assert ensured == [
        (VLLM_SR_DOCKER_IMAGE_ROCM, "never"),
        (VLLM_SR_ENVOY_DOCKER_IMAGE_DEFAULT, "never"),
        (VLLM_SR_DASHBOARD_DOCKER_IMAGE_DEFAULT, "never"),
    ]


def test_get_runtime_images_reuses_official_router_image_from_official_tag(
    monkeypatch,
):
    ensured = []
    monkeypatch.setattr(
        docker_images,
        "_resolve_selected_image",
        lambda image, normalized_platform: "ghcr.io/vllm-project/semantic-router/vllm-sr:v1.2.3",
    )
    monkeypatch.setattr(
        docker_images,
        "_ensure_image_available",
        lambda image, pull_policy: ensured.append((image, pull_policy)),
    )

    images = docker_images.get_runtime_images(pull_policy="never")

    assert images["router"] == "ghcr.io/vllm-project/semantic-router/vllm-sr:v1.2.3"
    assert ensured == [
        ("ghcr.io/vllm-project/semantic-router/vllm-sr:v1.2.3", "never"),
        (VLLM_SR_ENVOY_DOCKER_IMAGE_DEFAULT, "never"),
        ("ghcr.io/vllm-project/semantic-router/dashboard:v1.2.3", "never"),
    ]


def test_get_runtime_images_derives_official_dashboard_image_from_official_tag(
    monkeypatch,
):
    ensured = []
    monkeypatch.setattr(
        docker_images,
        "_resolve_selected_image",
        lambda image, normalized_platform: "ghcr.io/vllm-project/semantic-router/vllm-sr:v1.2.3",
    )
    monkeypatch.setattr(
        docker_images,
        "_ensure_image_available",
        lambda image, pull_policy: ensured.append((image, pull_policy)),
    )

    images = docker_images.get_runtime_images(pull_policy="never")

    assert (
        images["dashboard"] == "ghcr.io/vllm-project/semantic-router/dashboard:v1.2.3"
    )
    assert ensured == [
        ("ghcr.io/vllm-project/semantic-router/vllm-sr:v1.2.3", "never"),
        (VLLM_SR_ENVOY_DOCKER_IMAGE_DEFAULT, "never"),
        ("ghcr.io/vllm-project/semantic-router/dashboard:v1.2.3", "never"),
    ]


def test_get_runtime_images_prefers_explicit_base_image_over_service_env(monkeypatch):
    ensured = []
    monkeypatch.setattr(
        docker_images,
        "_resolve_selected_image",
        lambda image, normalized_platform: image,
    )
    monkeypatch.setattr(
        docker_images,
        "_ensure_image_available",
        lambda image, pull_policy: ensured.append((image, pull_policy)),
    )
    monkeypatch.setenv("VLLM_SR_ROUTER_IMAGE", "router:from-env")

    images = docker_images.get_runtime_images(
        image="base:explicit",
        pull_policy="never",
    )

    assert images == {
        "router": "base:explicit",
        "envoy": "base:explicit",
        "dashboard": "base:explicit",
    }
    assert ensured == [("base:explicit", "never")]


def test_get_runtime_images_prefers_explicit_service_image_over_explicit_base_image(
    monkeypatch,
):
    ensured = []
    monkeypatch.setattr(
        docker_images,
        "_resolve_selected_image",
        lambda image, normalized_platform: image,
    )
    monkeypatch.setattr(
        docker_images,
        "_ensure_image_available",
        lambda image, pull_policy: ensured.append((image, pull_policy)),
    )

    images = docker_images.get_runtime_images(
        image="base:explicit",
        envoy_image="envoy:explicit",
        pull_policy="never",
    )

    assert images == {
        "router": "base:explicit",
        "envoy": "envoy:explicit",
        "dashboard": "base:explicit",
    }
    assert ensured == [
        ("base:explicit", "never"),
        ("envoy:explicit", "never"),
    ]


def test_get_runtime_images_upgrades_router_override_to_rocm_on_amd(monkeypatch):
    ensured = []
    monkeypatch.setattr(
        docker_images,
        "_resolve_selected_image",
        lambda image, normalized_platform: "base:latest",
    )
    monkeypatch.setattr(
        docker_images,
        "_ensure_image_available",
        lambda image, pull_policy: ensured.append((image, pull_policy)),
    )

    images = docker_images.get_runtime_images(
        router_image="ghcr.io/vllm-project/semantic-router/vllm-sr:custom",
        pull_policy="never",
        platform="amd",
    )

    assert (
        images["router"] == "ghcr.io/vllm-project/semantic-router/vllm-sr-rocm:custom"
    )
    assert images["envoy"] == "base:latest"
    assert images["dashboard"] == "base:latest"
    assert ensured == [
        ("ghcr.io/vllm-project/semantic-router/vllm-sr-rocm:custom", "never"),
        ("base:latest", "never"),
    ]


def test_get_runtime_images_leaves_nonofficial_router_override_unchanged_on_amd(
    monkeypatch,
):
    ensured = []
    monkeypatch.setattr(
        docker_images,
        "_resolve_selected_image",
        lambda image, normalized_platform: "base:latest",
    )
    monkeypatch.setattr(
        docker_images,
        "_ensure_image_available",
        lambda image, pull_policy: ensured.append((image, pull_policy)),
    )

    images = docker_images.get_runtime_images(
        router_image="router:custom",
        pull_policy="never",
        platform="amd",
    )

    assert images["router"] == "router:custom"
    assert images["envoy"] == "base:latest"
    assert images["dashboard"] == "base:latest"
    assert ensured == [
        ("router:custom", "never"),
        ("base:latest", "never"),
    ]
