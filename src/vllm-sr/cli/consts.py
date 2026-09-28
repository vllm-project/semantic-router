"""Constants for vLLM Semantic Router CLI."""

import re

from cli import __version__


def image_tag_for_cli_version(cli_version: str) -> str:
    """Keep stable CLI installations on their matching release images."""
    if re.fullmatch(r"\d+\.\d+\.\d+", cli_version):
        return f"v{cli_version}"
    return "latest"


_DEFAULT_IMAGE_TAG = image_tag_for_cli_version(__version__)

# Docker image configuration
VLLM_SR_CONTAINER_IMAGE_DEFAULT = (
    f"ghcr.io/vllm-project/semantic-router/vllm-sr:{_DEFAULT_IMAGE_TAG}"
)
VLLM_SR_CONTAINER_IMAGE_ROCM = (
    f"ghcr.io/vllm-project/semantic-router/vllm-sr-rocm:{_DEFAULT_IMAGE_TAG}"
)
VLLM_SR_CONTAINER_IMAGE_CUDA = (
    f"ghcr.io/vllm-project/semantic-router/vllm-sr-cuda:{_DEFAULT_IMAGE_TAG}"
)
VLLM_SR_ENVOY_CONTAINER_IMAGE_DEFAULT = "envoyproxy/envoy:v1.35.3"
VLLM_SR_DASHBOARD_CONTAINER_IMAGE_DEFAULT = (
    f"ghcr.io/vllm-project/semantic-router/dashboard:{_DEFAULT_IMAGE_TAG}"
)
DEFAULT_STACK_NAME = "vllm-sr"
PLATFORM_AMD = "amd"
PLATFORM_NVIDIA = "nvidia"
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
