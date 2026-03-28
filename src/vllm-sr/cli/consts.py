"""Constants for vLLM Semantic Router CLI."""

from cli import __version__

# Docker image configuration
VLLM_SR_DOCKER_IMAGE_DEFAULT = "ghcr.io/vllm-project/semantic-router/vllm-sr:latest"
VLLM_SR_DOCKER_IMAGE_ROCM = "ghcr.io/vllm-project/semantic-router/vllm-sr-rocm:latest"
VLLM_SR_ROUTER_DOCKER_IMAGE_DEFAULT = VLLM_SR_DOCKER_IMAGE_DEFAULT
VLLM_SR_ROUTER_DOCKER_IMAGE_ROCM = VLLM_SR_DOCKER_IMAGE_ROCM
VLLM_SR_ENVOY_DOCKER_IMAGE_DEFAULT = "envoyproxy/envoy:v1.34-latest"
VLLM_SR_DASHBOARD_DOCKER_IMAGE_DEFAULT = (
    "ghcr.io/vllm-project/semantic-router/dashboard:latest"
)
VLLM_SR_DOCKER_IMAGE_DEV = "vllm-sr:dev"
VLLM_SR_DOCKER_IMAGE_RELEASE = f"vllm-sr:{__version__}"
VLLM_SR_SIM_DOCKER_IMAGE_DEFAULT = (
    "ghcr.io/vllm-project/semantic-router/vllm-sr-sim:latest"
)
VLLM_SR_DOCKER_NAME = "vllm-sr-container"
VLLM_SR_SIM_DOCKER_NAME = "vllm-sr-sim-container"
DEFAULT_STACK_NAME = "vllm-sr"
PLATFORM_AMD = "amd"
RUNTIME_TOPOLOGY_ENV = "VLLM_SR_TOPOLOGY"
RUNTIME_TOPOLOGY_LEGACY = "legacy"
RUNTIME_TOPOLOGY_SPLIT = "split"
DEFAULT_RUNTIME_TOPOLOGY = RUNTIME_TOPOLOGY_SPLIT

# Image pull policies
IMAGE_PULL_POLICY_ALWAYS = "always"
IMAGE_PULL_POLICY_IF_NOT_PRESENT = "ifnotpresent"
IMAGE_PULL_POLICY_NEVER = "never"
DEFAULT_IMAGE_PULL_POLICY = IMAGE_PULL_POLICY_ALWAYS

# Service names
SERVICE_NAME_ROUTER = "router"
SERVICE_NAME_ENVOY = "envoy"

# Default ports
DEFAULT_ENVOY_PORT = 9901
DEFAULT_ROUTER_PORT = 50051
DEFAULT_API_PORT = 8080
DEFAULT_LISTENER_PORT = 8899
DEFAULT_DASHBOARD_PORT = 8700
DEFAULT_METRICS_PORT = 9190
DEFAULT_FLEET_SIM_PORT = 8810

# Health check
HEALTH_CHECK_TIMEOUT = 1800  # 5 minutes (increased for model loading)
HEALTH_CHECK_INTERVAL = 2

# Log prefixes
LOG_PREFIX_ROUTER = "[router]"
LOG_PREFIX_ENVOY = "[envoy]"
LOG_PREFIX_ACCESS = "[access_logs]"

# File descriptor limits
DEFAULT_NOFILE_LIMIT = 65536
MIN_NOFILE_LIMIT = 8192

# External API model formats (routed through Envoy to external API endpoints)
# These models don't require vLLM endpoints - they use external APIs like Anthropic
EXTERNAL_API_MODEL_FORMATS = ["anthropic"]
