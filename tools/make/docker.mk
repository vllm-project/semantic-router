# ======== docker.mk ============
# = Docker build and management =
# ======== docker.mk ============

##@ Docker

# Docker image tags
DOCKER_REGISTRY ?= ghcr.io/vllm-project/semantic-router
DOCKER_TAG ?= latest

# Build all Docker images
# Note: extproc-rocm is excluded because it requires x86_64 + ROCm hardware.
# Build it explicitly with: make docker-build-extproc-rocm
docker-build-all: ## Build all Docker images
docker-build-all: docker-build-extproc docker-build-llm-katan docker-build-dashboard docker-build-precommit

# Build extproc Docker image
docker-build-extproc: ## Build extproc Docker image
docker-build-extproc:
	@$(LOG_TARGET)
	@echo "Building extproc Docker image..."
	@$(CONTAINER_RUNTIME) build -f tools/docker/Dockerfile.extproc -t $(DOCKER_REGISTRY)/extproc:$(DOCKER_TAG) .

# Build extproc-rocm Docker image (AMD GPU / ROCm, x86_64 only)
docker-build-extproc-rocm: ## Build extproc-rocm Docker image (AMD GPU)
docker-build-extproc-rocm:
	@$(LOG_TARGET)
	@echo "Building extproc-rocm Docker image (x86_64 only, ROCm 7.0)..."
	@$(CONTAINER_RUNTIME) build -f tools/docker/Dockerfile.extproc-rocm -t $(DOCKER_REGISTRY)/extproc-rocm:$(DOCKER_TAG) .

# Build llm-katan Docker image
docker-build-llm-katan: ## Build llm-katan Docker image
docker-build-llm-katan:
	@$(LOG_TARGET)
	@echo "Building llm-katan Docker image..."
	@$(CONTAINER_RUNTIME) build -f e2e/testing/llm-katan/Dockerfile -t $(DOCKER_REGISTRY)/llm-katan:$(DOCKER_TAG) e2e/testing/llm-katan/

# Build dashboard Docker image
docker-build-dashboard: ## Build dashboard Docker image
docker-build-dashboard:
	@$(LOG_TARGET)
	@echo "Building dashboard Docker image..."
	@$(CONTAINER_RUNTIME) build -f dashboard/backend/Dockerfile -t $(DOCKER_REGISTRY)/dashboard:$(DOCKER_TAG) .

# Build precommit Docker image
docker-build-precommit: ## Build precommit Docker image
docker-build-precommit:
	@$(LOG_TARGET)
	@echo "Building precommit Docker image..."
	@$(CONTAINER_RUNTIME) build -f tools/docker/Dockerfile.precommit -t $(DOCKER_REGISTRY)/precommit:$(DOCKER_TAG) .

# Test llm-katan Docker image locally
docker-test-llm-katan: ## Test llm-katan Docker image locally
docker-test-llm-katan:
	@$(LOG_TARGET)
	@echo "Testing llm-katan Docker image..."
	@curl -f http://localhost:8000/v1/models || (echo "Models endpoint failed" && exit 1)
	@echo "\nllm-katan Docker image test passed"

# Run llm-katan Docker image locally
docker-run-llm-katan: ## Run llm-katan Docker image locally
docker-run-llm-katan: docker-build-llm-katan
	@$(LOG_TARGET)
	@echo "Running llm-katan Docker image on port 8000..."
	@echo "Access the server at: http://localhost:8000"
	@echo "Press Ctrl+C to stop"
	@$(CONTAINER_RUNTIME) run --rm -p 8000:8000 $(DOCKER_REGISTRY)/llm-katan:$(DOCKER_TAG)

# Run llm-katan with custom served model name
docker-run-llm-katan-custom: ## Run with custom served model name, by append SERVED_NAME=name
docker-run-llm-katan-custom:
	@$(LOG_TARGET)
	@echo "Running llm-katan with custom served model name..."
	@echo "Usage: make docker-run-llm-katan-custom SERVED_NAME=your-served-model-name"
	@if [ -z "$(SERVED_NAME)" ]; then \
		echo "Error: SERVED_NAME variable is required"; \
		echo "Example: make docker-run-llm-katan-custom SERVED_NAME=claude-3-haiku"; \
		exit 1; \
	fi
	@$(CONTAINER_RUNTIME) run --rm -p 8000:8000 $(DOCKER_REGISTRY)/llm-katan:$(DOCKER_TAG) \
		llm-katan --model "Qwen/Qwen3-0.6B" --served-model-name "$(SERVED_NAME)" --host 0.0.0.0 --port 8000

# Clean up Docker images
docker-clean: ## Clean up Docker images
docker-clean:
	@$(LOG_TARGET)
	@echo "Cleaning up Docker images..."
	@$(CONTAINER_RUNTIME) image prune -f
	@echo "Docker cleanup completed"

# Push Docker images (for CI/CD)
# Note: extproc-rocm is excluded; push it explicitly with: make docker-push-extproc-rocm
docker-push-all: ## Push all Docker images
docker-push-all: docker-push-extproc docker-push-llm-katan
	@$(LOG_TARGET)
	@echo "All Docker images pushed successfully"

docker-push-extproc: ## Push extproc Docker image
docker-push-extproc:
	@$(LOG_TARGET)
	@echo "Pushing extproc Docker image..."
	@$(CONTAINER_RUNTIME) push $(DOCKER_REGISTRY)/extproc:$(DOCKER_TAG)

docker-push-extproc-rocm: ## Push extproc-rocm Docker image
docker-push-extproc-rocm:
	@$(LOG_TARGET)
	@echo "Pushing extproc-rocm Docker image..."
	@$(CONTAINER_RUNTIME) push $(DOCKER_REGISTRY)/extproc-rocm:$(DOCKER_TAG)

docker-push-llm-katan: ## Push llm-katan Docker image
docker-push-llm-katan:
	@$(LOG_TARGET)
	@echo "Pushing llm-katan Docker image..."
	@$(CONTAINER_RUNTIME) push $(DOCKER_REGISTRY)/llm-katan:$(DOCKER_TAG)

# Help target for Docker commands
docker-help:
docker-help: ## Show help for Docker-related make targets and environment variables
	@echo "Environment Variables:"
	@echo "  CONTAINER_RUNTIME - Container runtime (default: docker, can be set to podman)"
	@echo "  DOCKER_REGISTRY   - Docker registry (default: ghcr.io/vllm-project/semantic-router)"
	@echo "  DOCKER_TAG        - Docker tag (default: latest)"
	@echo "  SERVED_NAME       - Served model name for custom runs"
	@echo "  VLLM_SR_PLATFORM  - vllm-sr platform hint (set to amd to use ROCm defaults)"
	@echo "  VLLM_SR_TARGETARCH - target image architecture (default: host-native, amd64 for ROCm)"
	@echo "  VLLM_SR_BUILDPLATFORM - Docker build platform (default: host-native, linux/amd64 for ROCm)"
	@echo "  VLLM_SR_DOCKERFILE_AMD - Dockerfile used when VLLM_SR_PLATFORM=amd"

##@ vLLM-SR (Semantic Router CLI)

# vLLM-SR specific variables
VLLM_SR_IMAGE ?= ghcr.io/vllm-project/semantic-router/vllm-sr:latest
VLLM_SR_IMAGE_ROCM ?= ghcr.io/vllm-project/semantic-router/vllm-sr-rocm:latest
VLLM_SR_CONTAINER ?= vllm-sr-container
VLLM_SR_PLATFORM ?=
VLLM_SR_PLATFORM_NORMALIZED := $(shell echo "$(VLLM_SR_PLATFORM)" | tr '[:upper:]' '[:lower:]')
VLLM_SR_DOCKERFILE ?= src/vllm-sr/Dockerfile
VLLM_SR_DOCKERFILE_AMD ?= src/vllm-sr/Dockerfile.rocm
VLLM_SR_HOST_ARCH_RAW := $(shell uname -m)
ifeq ($(VLLM_SR_HOST_ARCH_RAW),arm64)
VLLM_SR_TARGETARCH ?= arm64
VLLM_SR_BUILDPLATFORM ?= linux/arm64
else ifeq ($(VLLM_SR_HOST_ARCH_RAW),aarch64)
VLLM_SR_TARGETARCH ?= arm64
VLLM_SR_BUILDPLATFORM ?= linux/arm64
else
VLLM_SR_TARGETARCH ?= amd64
VLLM_SR_BUILDPLATFORM ?= linux/amd64
endif

# AMD platform defaults (can still be overridden via env/CLI variables)
ifeq ($(VLLM_SR_PLATFORM_NORMALIZED),amd)
ifeq ($(origin VLLM_SR_IMAGE),file)
VLLM_SR_IMAGE := $(VLLM_SR_IMAGE_ROCM)
endif
ifeq ($(origin VLLM_SR_DOCKERFILE),file)
VLLM_SR_DOCKERFILE := $(VLLM_SR_DOCKERFILE_AMD)
endif
ifeq ($(origin VLLM_SR_TARGETARCH),file)
VLLM_SR_TARGETARCH := amd64
endif
ifeq ($(origin VLLM_SR_BUILDPLATFORM),file)
VLLM_SR_BUILDPLATFORM := linux/amd64
endif
endif

# Default 1 so vllm-sr build works behind corporate proxies; set GIT_SSL_NO_VERIFY=0 for strict SSL verification.
GIT_SSL_NO_VERIFY ?= 1
VLLM_SR_BUILD_ARGS := --network=host --build-arg TARGETARCH=$(VLLM_SR_TARGETARCH) --build-arg BUILDPLATFORM=$(VLLM_SR_BUILDPLATFORM)
ifeq ($(GIT_SSL_NO_VERIFY),1)
VLLM_SR_BUILD_ARGS += --build-arg GIT_SSL_NO_VERIFY=1
endif

vllm-sr-dev: ## Rebuild vLLM Semantic Router image and install CLI
vllm-sr-dev:
	@$(LOG_TARGET)
	@echo "=========================================="
	@echo "vLLM Semantic Router Development Setup"
	@echo "=========================================="
	@echo ""
	@echo "This will:"
	@echo "  1. Clean up old containers"
	@echo "  2. Rebuild Docker image with all dependencies"
	@echo "  3. Install vLLM-SR CLI in development mode"
	@echo ""
	@echo "1. Cleaning up old containers..."
	@$(CONTAINER_RUNTIME) rm -f $(VLLM_SR_CONTAINER) 2>/dev/null || echo "  No container to remove"
	@echo ""
	@echo "2. Rebuilding Docker image..."
	@echo "  Building from: $(PWD)"
	@echo "  Platform: $(if $(VLLM_SR_PLATFORM_NORMALIZED),$(VLLM_SR_PLATFORM_NORMALIZED),default)"
	@echo "  Target arch: $(VLLM_SR_TARGETARCH)"
	@echo "  Build platform: $(VLLM_SR_BUILDPLATFORM)"
	@echo "  Dockerfile: $(VLLM_SR_DOCKERFILE)"
	@echo "  Image: $(VLLM_SR_IMAGE)"
	@echo ""
	@$(CONTAINER_RUNTIME) build $(VLLM_SR_BUILD_ARGS) -t $(VLLM_SR_IMAGE) -f $(VLLM_SR_DOCKERFILE) .
	@echo ""
	@echo "Image built: $(VLLM_SR_IMAGE)"
	@echo ""
	@echo "3. Installing vLLM-SR CLI in development mode..."
	@pip install -e src/vllm-sr
	@echo "vLLM-SR CLI installed"
	@echo ""
	@echo "=========================================="
	@echo "Development Setup Complete"
	@echo "=========================================="
	@echo ""
	@echo "Next steps:"
	@echo "  Start service: cd src/vllm-sr && vllm-sr serve config.yaml"
	@echo "  Or use:        make vllm-sr-start"
	@echo ""

vllm-sr-build: ## Build vLLM Semantic Router Docker image
vllm-sr-build:
	@$(LOG_TARGET)
	@echo "Building vLLM Semantic Router Docker image..."
	@echo "  Platform: $(if $(VLLM_SR_PLATFORM_NORMALIZED),$(VLLM_SR_PLATFORM_NORMALIZED),default)"
	@echo "  Target arch: $(VLLM_SR_TARGETARCH)"
	@echo "  Build platform: $(VLLM_SR_BUILDPLATFORM)"
	@echo "  Dockerfile: $(VLLM_SR_DOCKERFILE)"
	@$(CONTAINER_RUNTIME) build $(VLLM_SR_BUILD_ARGS) -t $(VLLM_SR_IMAGE) -f $(VLLM_SR_DOCKERFILE) .
	@echo "Image built: $(VLLM_SR_IMAGE)"

vllm-sr-start: ## Start vLLM Semantic Router service
vllm-sr-start: vllm-sr-dev
	@$(LOG_TARGET)
	@echo "Starting vLLM Semantic Router service..."
	@vllm-sr serve --image-pull-policy=ifnotpresent --image $(VLLM_SR_IMAGE)
	@vllm-sr dashboard

##@ vLLM-SR Tests (e2e tests for vllm-sr CLI)
# Tests are located in e2e/testing/vllm-sr-cli/

vllm-sr-test: ## Run CLI unit tests (fast, no Docker image required)
vllm-sr-test:
	@$(LOG_TARGET)
	@cd e2e/testing/vllm-sr-cli && python run_cli_tests.py --verbose

vllm-sr-test-integration: ## Run CLI unit + integration tests (requires Docker image)
vllm-sr-test-integration: vllm-sr-build
	@$(LOG_TARGET)
	@cd e2e/testing/vllm-sr-cli && RUN_INTEGRATION_TESTS=true python run_cli_tests.py --verbose --integration
