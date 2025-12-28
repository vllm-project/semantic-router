"""Constants for vLLM Semantic Router CLI."""

# Docker image configuration
VLLM_SR_DOCKER_IMAGE_DEFAULT = "ghcr.io/vllm-project/semantic-router/vllm-sr:latest"
VLLM_SR_DOCKER_IMAGE_DEV = "vllm-sr:dev"
VLLM_SR_DOCKER_IMAGE_RELEASE = "vllm-sr:0.1.0"
VLLM_SR_DOCKER_NAME = "vllm-sr-container"

# Image pull policies
IMAGE_PULL_POLICY_ALWAYS = "always"
IMAGE_PULL_POLICY_IF_NOT_PRESENT = "ifnotpresent"
IMAGE_PULL_POLICY_NEVER = "never"
DEFAULT_IMAGE_PULL_POLICY = IMAGE_PULL_POLICY_ALWAYS

# Service names
SERVICE_NAME_ROUTER = "router"
SERVICE_NAME_ENVOY = "envoy"

# Default ports
DEFAULT_ENVOY_PORT = 8801
DEFAULT_ROUTER_PORT = 50051
DEFAULT_API_PORT = 8080

# Health check
HEALTH_CHECK_TIMEOUT = 1800  # 5 minutes (increased for model loading)
HEALTH_CHECK_INTERVAL = 2

# Log prefixes
LOG_PREFIX_ROUTER = "[router]"
LOG_PREFIX_ENVOY = "[envoy]"
LOG_PREFIX_ACCESS = "[access_logs]"
