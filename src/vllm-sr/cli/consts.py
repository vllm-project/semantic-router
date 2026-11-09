"""Constants for vLLM Semantic Router CLI."""

import re

from cli import __version__
from cli.model_catalog import available_catalog_versions
from cli.source_revision import SOURCE_REVISION


def _release_catalog_installed(release: str) -> bool:
    return release in available_catalog_versions()


def image_tag_for_cli_version(
    cli_version: str, released=_release_catalog_installed
) -> str:
    """Keep released CLI installations on their matching release images.

    A release build ships its vMAJOR.MINOR catalog snapshot, which the release
    contract requires before the tag. `main` already carries the next version
    without that snapshot, so a source install of it runs the main-channel
    images.
    """
    match = re.fullmatch(r"(\d+)\.(\d+)\.\d+", cli_version)
    if match and released(f"v{match.group(1)}.{match.group(2)}"):
        return f"v{cli_version}"
    return "latest"


def image_tag_for_cli_build(
    cli_version: str, source_revision: str, released=_release_catalog_installed
) -> str:
    """Run the images published from this build's commit.

    A development package records its main commit, whose images main publishes
    before the package. The moving `latest` images can already reject the
    config this CLI writes, so only a source checkout, which records no commit,
    follows them.
    """
    tag = image_tag_for_cli_version(cli_version, released)
    if tag == "latest" and source_revision:
        return source_revision
    return tag


_DEFAULT_IMAGE_TAG = image_tag_for_cli_build(__version__, SOURCE_REVISION)

# Docker image configuration. Every image the CLI pulls by default names its
# registry: Podman without unqualified-search registries, as Debian and Ubuntu
# package it, cannot pull a short name, and Docker resolves both to one image.
VLLM_SR_CONTAINER_IMAGE_DEFAULT = (
    f"ghcr.io/vllm-project/semantic-router/vllm-sr:{_DEFAULT_IMAGE_TAG}"
)
VLLM_SR_CONTAINER_IMAGE_ROCM = (
    f"ghcr.io/vllm-project/semantic-router/vllm-sr-rocm:{_DEFAULT_IMAGE_TAG}"
)
VLLM_SR_CONTAINER_IMAGE_CUDA = (
    f"ghcr.io/vllm-project/semantic-router/vllm-sr-cuda:{_DEFAULT_IMAGE_TAG}"
)
VLLM_SR_ENVOY_CONTAINER_IMAGE_DEFAULT = "docker.io/envoyproxy/envoy:v1.35.3"
VLLM_SR_DASHBOARD_CONTAINER_IMAGE_DEFAULT = (
    f"ghcr.io/vllm-project/semantic-router/dashboard:{_DEFAULT_IMAGE_TAG}"
)
DEFAULT_STACK_NAME = "vllm-sr"
PLATFORM_ROCM = "rocm"
PLATFORM_CUDA = "cuda"
RUNTIME_TOPOLOGY_ENV = "VLLM_SR_TOPOLOGY"
RUNTIME_TOPOLOGY_SPLIT = "split"
DEFAULT_RUNTIME_TOPOLOGY = RUNTIME_TOPOLOGY_SPLIT

# Image pull policies
IMAGE_PULL_POLICY_ALWAYS = "always"
IMAGE_PULL_POLICY_IF_NOT_PRESENT = "ifnotpresent"
IMAGE_PULL_POLICY_NEVER = "never"
DEFAULT_IMAGE_PULL_POLICY = IMAGE_PULL_POLICY_ALWAYS

# Default ports
DEFAULT_ENVOY_PORT = 9901
DEFAULT_ROUTER_PORT = 50051
DEFAULT_API_PORT = 8080
DEFAULT_LISTENER_PORT = 8899
DEFAULT_DASHBOARD_PORT = 8700
DEFAULT_METRICS_PORT = 9190
DEFAULT_MILVUS_PORT = 19530

# Health check
HEALTH_CHECK_TIMEOUT = 1800  # Default local startup readiness budget: 30 minutes.
HEALTH_CHECK_INTERVAL = 2

# File descriptor limits
DEFAULT_NOFILE_LIMIT = 65536
MIN_NOFILE_LIMIT = 8192

# Container runtime selection
CONTAINER_RUNTIME_DOCKER = "docker"
CONTAINER_RUNTIME_PODMAN = "podman"
SUPPORTED_CONTAINER_RUNTIMES = (
    CONTAINER_RUNTIME_DOCKER,
    CONTAINER_RUNTIME_PODMAN,
)
CONTAINER_RUNTIME_ENV = "CONTAINER_RUNTIME"
