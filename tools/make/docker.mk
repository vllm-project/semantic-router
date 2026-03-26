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
docker-build-all: docker-build-extproc docker-build-llm-katan docker-build-dashboard docker-build-precommit docker-build-vllm-sr-sim

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

# Ensure the official Envoy image is available locally
docker-build-vllm-sr-envoy: ## Build vllm-sr-envoy Docker image
docker-build-vllm-sr-envoy:
	@$(LOG_TARGET)
	@echo "Ensuring official Envoy image is available..."
	@$(CONTAINER_RUNTIME) image inspect $(VLLM_SR_ENVOY_IMAGE) >/dev/null 2>&1 || $(CONTAINER_RUNTIME) pull $(VLLM_SR_ENVOY_IMAGE)

# Build router runtime image using the existing vllm-sr Dockerfile
docker-build-vllm-sr-router: ## Build vllm-sr-router Docker image
docker-build-vllm-sr-router:
	@$(LOG_TARGET)
	@echo "Building vllm-sr-router Docker image..."
	@$(CONTAINER_RUNTIME) build $(VLLM_SR_BUILD_ARGS) -f $(VLLM_SR_DOCKERFILE) -t $(DOCKER_REGISTRY)/vllm-sr-router:$(DOCKER_TAG) .

# Build vllm-sr-sim Docker image
docker-build-vllm-sr-sim: ## Build vllm-sr-sim Docker image
docker-build-vllm-sr-sim:
	@$(LOG_TARGET)
	@echo "Building vllm-sr-sim Docker image..."
	@$(CONTAINER_RUNTIME) build -f src/fleet-sim/Dockerfile -t $(DOCKER_REGISTRY)/vllm-sr-sim:$(DOCKER_TAG) .

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
docker-push-all: docker-push-extproc docker-push-llm-katan docker-push-dashboard docker-push-vllm-sr-sim
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

docker-push-dashboard: ## Push dashboard Docker image
docker-push-dashboard:
	@$(LOG_TARGET)
	@echo "Pushing dashboard Docker image..."
	@$(CONTAINER_RUNTIME) push $(DOCKER_REGISTRY)/dashboard:$(DOCKER_TAG)

docker-push-vllm-sr-router: ## Push vllm-sr-router Docker image
docker-push-vllm-sr-router:
	@$(LOG_TARGET)
	@echo "Pushing vllm-sr-router Docker image..."
	@$(CONTAINER_RUNTIME) push $(DOCKER_REGISTRY)/vllm-sr-router:$(DOCKER_TAG)

docker-push-vllm-sr-envoy: ## Push vllm-sr-envoy Docker image
docker-push-vllm-sr-envoy:
	@$(LOG_TARGET)
	@echo "Skipping push for upstream-managed Envoy image: $(VLLM_SR_ENVOY_IMAGE)"

docker-push-vllm-sr-sim: ## Push vllm-sr-sim Docker image
docker-push-vllm-sr-sim:
	@$(LOG_TARGET)
	@echo "Pushing vllm-sr-sim Docker image..."
	@$(CONTAINER_RUNTIME) push $(DOCKER_REGISTRY)/vllm-sr-sim:$(DOCKER_TAG)

# Help target for Docker commands
docker-help:
docker-help: ## Show help for Docker-related make targets and environment variables
	@echo "Environment Variables:"
	@echo "  CONTAINER_RUNTIME - Container runtime (default: docker; Podman is not supported)"
	@echo "  DOCKER_REGISTRY   - Docker registry (default: ghcr.io/vllm-project/semantic-router)"
	@echo "  DOCKER_TAG        - Docker tag (default: latest)"
	@echo "  SKIP_COMPAT_IMAGE - set to 1 only when the local router compatibility image is already up to date"
	@echo "  SERVED_NAME       - Served model name for custom runs"
	@echo "  VLLM_SR_PLATFORM  - vllm-sr platform hint (set to amd to use ROCm defaults)"
	@echo "  VLLM_SR_TOPOLOGY  - local runtime topology (legacy default, split for router/envoy/dashboard containers)"
	@echo "  VLLM_SR_TARGETARCH - target image architecture (default: host-native, amd64 for ROCm)"
	@echo "  VLLM_SR_BUILDPLATFORM - Docker build platform (default: host-native, linux/amd64 for ROCm)"
	@echo "  VLLM_SR_DOCKERFILE_AMD - Dockerfile used when VLLM_SR_PLATFORM=amd"
	@echo "  VLLM_SR_ROUTER_IMAGE - router runtime image override (defaults to VLLM_SR_ROUTER_IMAGE_DEFAULT)"
	@echo "  VLLM_SR_ENVOY_IMAGE - envoy runtime image override (defaults to VLLM_SR_ENVOY_IMAGE_DEFAULT)"
	@echo "  VLLM_SR_DASHBOARD_IMAGE - dashboard runtime image override (defaults to VLLM_SR_DASHBOARD_IMAGE_DEFAULT)"
	@echo "  VLLM_SR_SIM_PORT  - host port for the vllm-sr-sim service container"

##@ vLLM-SR (Semantic Router CLI)

# vLLM-SR specific variables
VLLM_SR_IMAGE ?= ghcr.io/vllm-project/semantic-router/vllm-sr:latest
VLLM_SR_IMAGE_ROCM ?= ghcr.io/vllm-project/semantic-router/vllm-sr-rocm:latest
VLLM_SR_ROUTER_IMAGE_DEFAULT ?= $(VLLM_SR_IMAGE)
VLLM_SR_ROUTER_IMAGE_ROCM ?= $(VLLM_SR_IMAGE_ROCM)
VLLM_SR_ENVOY_IMAGE_DEFAULT ?= envoyproxy/envoy:v1.34-latest
VLLM_SR_DASHBOARD_IMAGE_DEFAULT ?= ghcr.io/vllm-project/semantic-router/dashboard:latest
VLLM_SR_ROUTER_IMAGE ?= $(VLLM_SR_ROUTER_IMAGE_DEFAULT)
VLLM_SR_ENVOY_IMAGE ?= $(VLLM_SR_ENVOY_IMAGE_DEFAULT)
VLLM_SR_DASHBOARD_IMAGE ?= $(VLLM_SR_DASHBOARD_IMAGE_DEFAULT)
VLLM_SR_CONTAINER ?= vllm-sr-container
VLLM_SR_ROUTER_CONTAINER ?= vllm-sr-router-container
VLLM_SR_ENVOY_CONTAINER ?= vllm-sr-envoy-container
VLLM_SR_DASHBOARD_CONTAINER ?= vllm-sr-dashboard-container
VLLM_SR_RUNTIME_CONTAINERS ?= $(VLLM_SR_CONTAINER) $(VLLM_SR_ROUTER_CONTAINER) $(VLLM_SR_ENVOY_CONTAINER) $(VLLM_SR_DASHBOARD_CONTAINER)
VLLM_SR_PLATFORM ?=
VLLM_SR_PLATFORM_NORMALIZED := $(shell echo "$(VLLM_SR_PLATFORM)" | tr '[:upper:]' '[:lower:]')
VLLM_SR_TOPOLOGY ?= legacy
VLLM_SR_TOPOLOGY_NORMALIZED := $(shell echo "$(VLLM_SR_TOPOLOGY)" | tr '[:upper:]' '[:lower:]')
VLLM_SR_DOCKERFILE ?= src/vllm-sr/Dockerfile
VLLM_SR_DOCKERFILE_AMD ?= src/vllm-sr/Dockerfile.rocm
VLLM_SR_ROUTER_DOCKERFILE ?= src/vllm-sr/Dockerfile.router
VLLM_SR_ROUTER_DOCKERFILE_AMD ?= src/vllm-sr/Dockerfile.router.rocm
VLLM_SR_ENVOY_DOCKERFILE ?= src/vllm-sr/Dockerfile.envoy
VLLM_SR_DASHBOARD_DOCKERFILE ?= dashboard/backend/Dockerfile
VLLM_SR_SIM_IMAGE ?= ghcr.io/vllm-project/semantic-router/vllm-sr-sim:latest
VLLM_SR_SIM_CONTAINER ?= vllm-sr-sim-container
VLLM_SR_SIM_DOCKERFILE ?= src/fleet-sim/Dockerfile
VLLM_SR_SIM_DIR ?= src/fleet-sim
VLLM_SR_SIM_PORT ?= 8810
SKIP_COMPAT_IMAGE_SOURCE := $(origin SKIP_COMPAT_IMAGE)
SKIP_COMPAT_IMAGE_DEFAULT := 0
ifeq ($(SKIP_COMPAT_IMAGE_SOURCE),undefined)
SKIP_COMPAT_IMAGE_EFFECTIVE := $(SKIP_COMPAT_IMAGE_DEFAULT)
else
SKIP_COMPAT_IMAGE_EFFECTIVE := $(SKIP_COMPAT_IMAGE)
endif
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
ifeq ($(origin VLLM_SR_ROUTER_IMAGE),file)
VLLM_SR_ROUTER_IMAGE := $(VLLM_SR_ROUTER_IMAGE_ROCM)
endif
ifeq ($(origin VLLM_SR_DOCKERFILE),file)
VLLM_SR_DOCKERFILE := $(VLLM_SR_DOCKERFILE_AMD)
endif
ifeq ($(origin VLLM_SR_ROUTER_DOCKERFILE),file)
VLLM_SR_ROUTER_DOCKERFILE := $(VLLM_SR_ROUTER_DOCKERFILE_AMD)
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
	@echo "Topology: $(VLLM_SR_TOPOLOGY_NORMALIZED)"
	@echo "This will:"
	@echo "  1. Clean up old containers"
	@if [ "$(SKIP_COMPAT_IMAGE_EFFECTIVE)" = "1" ]; then \
		echo "  2. Reuse the existing vLLM-SR router compatibility Docker image"; \
	else \
		echo "  2. Rebuild the vLLM-SR router compatibility Docker image"; \
	fi
	@if [ "$(VLLM_SR_TOPOLOGY_NORMALIZED)" = "split" ]; then \
		echo "  3. Ensure the official Envoy image is available"; \
		echo "  4. Rebuild the dashboard Docker image"; \
		echo "  5. Build the vLLM-SR-Sim service image"; \
		echo "  6. Install vLLM-SR and vLLM-SR-Sim CLIs in development mode"; \
	else \
		echo "  3. Build the vLLM-SR-Sim service image"; \
		echo "  4. Install vLLM-SR and vLLM-SR-Sim CLIs in development mode"; \
	fi
	@echo ""
	@echo "1. Cleaning up old containers..."
	@$(CONTAINER_RUNTIME) rm -f $(VLLM_SR_RUNTIME_CONTAINERS) 2>/dev/null || echo "  No runtime containers to remove"
	@$(CONTAINER_RUNTIME) rm -f $(VLLM_SR_SIM_CONTAINER) 2>/dev/null || echo "  No simulator container to remove"
	@echo ""
	@if [ "$(SKIP_COMPAT_IMAGE_EFFECTIVE)" = "1" ]; then \
		echo "2. Reusing existing vLLM-SR router compatibility Docker image (SKIP_COMPAT_IMAGE=1)"; \
		echo "   Only use this when the local router compatibility image already includes your latest code changes."; \
		echo ""; \
	else \
		echo "2. Rebuilding vLLM-SR router compatibility Docker image..."; \
		echo "  Building from: $(PWD)"; \
		echo "  Platform: $(if $(VLLM_SR_PLATFORM_NORMALIZED),$(VLLM_SR_PLATFORM_NORMALIZED),default)"; \
		echo "  Target arch: $(VLLM_SR_TARGETARCH)"; \
		echo "  Build platform: $(VLLM_SR_BUILDPLATFORM)"; \
		echo "  Dockerfile: $(VLLM_SR_DOCKERFILE)"; \
		echo "  Image: $(VLLM_SR_IMAGE)"; \
		echo ""; \
		$(CONTAINER_RUNTIME) build $(VLLM_SR_BUILD_ARGS) -t $(VLLM_SR_IMAGE) -f $(VLLM_SR_DOCKERFILE) .; \
		echo ""; \
		echo "Router compatibility image built: $(VLLM_SR_IMAGE)"; \
		echo ""; \
	fi
	@if [ "$(VLLM_SR_TOPOLOGY_NORMALIZED)" = "split" ]; then \
		echo "3. Ensuring official Envoy image is available..."; \
		echo "  Image: $(VLLM_SR_ENVOY_IMAGE)"; \
		echo ""; \
		$(CONTAINER_RUNTIME) image inspect $(VLLM_SR_ENVOY_IMAGE) >/dev/null 2>&1 || $(CONTAINER_RUNTIME) pull $(VLLM_SR_ENVOY_IMAGE); \
		echo ""; \
		echo "Envoy image available: $(VLLM_SR_ENVOY_IMAGE)"; \
		echo ""; \
		echo "4. Rebuilding dashboard Docker image..."; \
		echo "  Dockerfile: $(VLLM_SR_DASHBOARD_DOCKERFILE)"; \
		echo "  Image: $(VLLM_SR_DASHBOARD_IMAGE)"; \
		echo ""; \
		$(CONTAINER_RUNTIME) build $(VLLM_SR_BUILD_ARGS) -t $(VLLM_SR_DASHBOARD_IMAGE) -f $(VLLM_SR_DASHBOARD_DOCKERFILE) .; \
		echo ""; \
		echo "Dashboard image built: $(VLLM_SR_DASHBOARD_IMAGE)"; \
		echo ""; \
		echo "5. Building vLLM-SR-Sim service image..."; \
	else \
		echo "3. Building vLLM-SR-Sim service image..."; \
	fi
	@$(CONTAINER_RUNTIME) build -t $(VLLM_SR_SIM_IMAGE) -f $(VLLM_SR_SIM_DOCKERFILE) .
	@echo ""
	@echo "Simulator image built: $(VLLM_SR_SIM_IMAGE)"
	@echo ""
	@if [ "$(VLLM_SR_TOPOLOGY_NORMALIZED)" = "split" ]; then \
		echo "6. Installing vLLM-SR and vLLM-SR-Sim CLIs in development mode..."; \
	else \
		echo "4. Installing vLLM-SR and vLLM-SR-Sim CLIs in development mode..."; \
	fi
	@python3 -m pip install -e src/vllm-sr -e "$(VLLM_SR_SIM_DIR)[dev]"
	@echo "vLLM-SR CLI and vLLM-SR-Sim installed"
	@echo ""
	@echo "=========================================="
	@echo "Development Setup Complete"
	@echo "=========================================="
	@echo ""
	@echo "Next steps:"
	@if [ "$(VLLM_SR_TOPOLOGY_NORMALIZED)" = "split" ]; then \
		echo "  Start service: cd src/vllm-sr && vllm-sr serve --topology split --config config.yaml"; \
	else \
		echo "  Start service: cd src/vllm-sr && vllm-sr serve --config config.yaml"; \
	fi
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

vllm-sr-router-build: ## Build vLLM Semantic Router router Docker image
vllm-sr-router-build:
	@$(LOG_TARGET)
	@echo "Building vLLM Semantic Router router Docker image..."
	@echo "  Target arch: $(VLLM_SR_TARGETARCH)"
	@echo "  Build platform: $(VLLM_SR_BUILDPLATFORM)"
	@echo "  Dockerfile: $(VLLM_SR_DOCKERFILE)"
	@$(CONTAINER_RUNTIME) build $(VLLM_SR_BUILD_ARGS) -t $(VLLM_SR_ROUTER_IMAGE) -f $(VLLM_SR_DOCKERFILE) .
	@echo "Image built: $(VLLM_SR_ROUTER_IMAGE)"

vllm-sr-envoy-build: ## Build vLLM Semantic Router Envoy Docker image
vllm-sr-envoy-build:
	@$(LOG_TARGET)
	@echo "Ensuring official Envoy image is available..."
	@$(CONTAINER_RUNTIME) image inspect $(VLLM_SR_ENVOY_IMAGE) >/dev/null 2>&1 || $(CONTAINER_RUNTIME) pull $(VLLM_SR_ENVOY_IMAGE)
	@echo "Image available: $(VLLM_SR_ENVOY_IMAGE)"

vllm-sr-dashboard-build: ## Build vLLM Semantic Router dashboard Docker image
vllm-sr-dashboard-build:
	@$(LOG_TARGET)
	@echo "Building vLLM Semantic Router dashboard Docker image..."
	@echo "  Target arch: $(VLLM_SR_TARGETARCH)"
	@echo "  Build platform: $(VLLM_SR_BUILDPLATFORM)"
	@echo "  Dockerfile: $(VLLM_SR_DASHBOARD_DOCKERFILE)"
	@$(CONTAINER_RUNTIME) build $(VLLM_SR_BUILD_ARGS) -t $(VLLM_SR_DASHBOARD_IMAGE) -f $(VLLM_SR_DASHBOARD_DOCKERFILE) .
	@echo "Image built: $(VLLM_SR_DASHBOARD_IMAGE)"

vllm-sr-start: ## Start vLLM Semantic Router service
vllm-sr-start: vllm-sr-dev
	@$(LOG_TARGET)
	@echo "Starting vLLM Semantic Router service..."
	@VLLM_SR_IMAGE=$(VLLM_SR_IMAGE) \
	VLLM_SR_ROUTER_IMAGE=$(VLLM_SR_ROUTER_IMAGE) \
	VLLM_SR_ENVOY_IMAGE=$(VLLM_SR_ENVOY_IMAGE) \
	VLLM_SR_DASHBOARD_IMAGE=$(VLLM_SR_DASHBOARD_IMAGE) \
	VLLM_SR_TOPOLOGY=$(VLLM_SR_TOPOLOGY_NORMALIZED) \
	vllm-sr serve --image-pull-policy=ifnotpresent
	@vllm-sr dashboard

##@ vLLM-SR Tests (e2e tests for vllm-sr CLI)
# Tests are located in e2e/testing/vllm-sr-cli/

vllm-sr-install-cli: ## Install vLLM-SR CLI in editable mode for local test execution
vllm-sr-install-cli:
	@python3 -m pip install -e src/vllm-sr

vllm-sr-sim-install-cli: ## Install vLLM-SR-Sim with dev extras for local execution
vllm-sr-sim-install-cli:
	@python3 -m pip install -e "$(VLLM_SR_SIM_DIR)[dev]"

vllm-sr-sim-test: ## Run vLLM-SR-Sim tests
vllm-sr-sim-test: vllm-sr-sim-install-cli
	@$(LOG_TARGET)
	@cd $(VLLM_SR_SIM_DIR) && python3 -m pytest tests -v

vllm-sr-sim-build: ## Build the vLLM-SR-Sim service image
vllm-sr-sim-build:
	@$(LOG_TARGET)
	@$(CONTAINER_RUNTIME) build -t $(VLLM_SR_SIM_IMAGE) -f $(VLLM_SR_SIM_DOCKERFILE) .

vllm-sr-sim-start: ## Start the vLLM-SR-Sim service container
vllm-sr-sim-start: vllm-sr-sim-build
	@$(LOG_TARGET)
	@echo "Starting vLLM-SR-Sim service on http://localhost:$(VLLM_SR_SIM_PORT)"
	@$(CONTAINER_RUNTIME) rm -f $(VLLM_SR_SIM_CONTAINER) 2>/dev/null || true
	@$(CONTAINER_RUNTIME) run -d --name $(VLLM_SR_SIM_CONTAINER) -p $(VLLM_SR_SIM_PORT):8000 $(VLLM_SR_SIM_IMAGE)

vllm-sr-test: ## Run CLI unit tests (fast, no Docker image required)
vllm-sr-test: vllm-sr-install-cli
	@$(LOG_TARGET)
	@cd e2e/testing/vllm-sr-cli && python run_cli_tests.py --verbose

vllm-sr-test-integration: ## Run CLI unit + integration tests (requires local router + simulator images)
vllm-sr-test-integration: vllm-sr-build vllm-sr-envoy-build vllm-sr-dashboard-build vllm-sr-sim-build vllm-sr-install-cli
	@$(LOG_TARGET)
	@cd e2e/testing/vllm-sr-cli && CONTAINER_RUNTIME=$(CONTAINER_RUNTIME) VLLM_SR_IMAGE=$(VLLM_SR_IMAGE) VLLM_SR_ROUTER_IMAGE=$(VLLM_SR_ROUTER_IMAGE) VLLM_SR_ENVOY_IMAGE=$(VLLM_SR_ENVOY_IMAGE) VLLM_SR_DASHBOARD_IMAGE=$(VLLM_SR_DASHBOARD_IMAGE) VLLM_SR_SIM_IMAGE=$(VLLM_SR_SIM_IMAGE) RUN_INTEGRATION_TESTS=true python run_cli_tests.py --verbose --integration

memory-test-integration: ## Run memory integration tests with local Milvus, llm-katan, and vllm-sr serve
memory-test-integration: vllm-sr-build vllm-sr-envoy-build vllm-sr-dashboard-build vllm-sr-sim-build vllm-sr-install-cli docker-build-llm-katan
	@$(LOG_TARGET)
	@CONTAINER_RUNTIME=$(CONTAINER_RUNTIME) \
	DOCKER_REGISTRY=$(DOCKER_REGISTRY) \
	DOCKER_TAG=$(DOCKER_TAG) \
	VLLM_SR_IMAGE=$(VLLM_SR_IMAGE) \
	VLLM_SR_ROUTER_IMAGE=$(VLLM_SR_ROUTER_IMAGE) \
	VLLM_SR_ENVOY_IMAGE=$(VLLM_SR_ENVOY_IMAGE) \
	VLLM_SR_DASHBOARD_IMAGE=$(VLLM_SR_DASHBOARD_IMAGE) \
	VLLM_SR_SIM_IMAGE=$(VLLM_SR_SIM_IMAGE) \
	bash e2e/testing/run_memory_integration.sh
