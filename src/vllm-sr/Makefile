.PHONY: help build install test clean docker-build docker-build-router docker-build-envoy docker-build-dashboard docker-run docker-stop dev-setup dev-serve dev-test rebuild-and-test config-router config-envoy

# Configuration
IMAGE_NAME ?= ghcr.io/vllm-project/semantic-router/vllm-sr
ROUTER_IMAGE_NAME ?= $(IMAGE_NAME)
ENVOY_IMAGE ?= envoyproxy/envoy:v1.34-latest
DASHBOARD_IMAGE_NAME ?= ghcr.io/vllm-project/semantic-router/dashboard
IMAGE_TAG ?= latest
FULL_IMAGE = $(IMAGE_NAME):$(IMAGE_TAG)
FULL_ROUTER_IMAGE = $(ROUTER_IMAGE_NAME):$(IMAGE_TAG)
FULL_ENVOY_IMAGE = $(ENVOY_IMAGE)
FULL_DASHBOARD_IMAGE = $(DASHBOARD_IMAGE_NAME):$(IMAGE_TAG)
CONFIG_FILE ?= config.yaml
OUTPUT_DIR ?= .vllm-sr
RUNTIME_CONTAINER ?= vllm-sr-router-container
PLATFORMS ?= linux/amd64,linux/arm64
BUILDX_BUILDER ?= vllm-sr-builder

help:
	@echo "vLLM Semantic Router CLI - Development Commands"
	@echo ""
	@echo "Quick Start:"
	@echo "  make dev-setup       - Complete setup (build + install)"
	@echo "  make dev-serve       - Start service with config"
	@echo "  make dev-test        - Test configuration generation"
	@echo "  make rebuild-and-test - Full rebuild and test workflow"
	@echo ""
	@echo "Build & Install:"
	@echo "  make build        - Build Docker images ($(FULL_IMAGE), $(FULL_ROUTER_IMAGE), $(FULL_ENVOY_IMAGE), $(FULL_DASHBOARD_IMAGE))"
	@echo "  make docker-build-router - Build router runtime image ($(FULL_ROUTER_IMAGE))"
	@echo "  make docker-build-envoy - Build Envoy runtime image ($(FULL_ENVOY_IMAGE))"
	@echo "  make docker-build-dashboard - Build dashboard runtime image ($(FULL_DASHBOARD_IMAGE))"
	@echo "  make install      - Install CLI in development mode"
	@echo "  make dev-setup    - Complete setup (build + install)"
	@echo ""
	@echo "Configuration:"
	@echo "  make validate      - Validate $(CONFIG_FILE)"
	@echo "  make config-router - Print canonical router config for $(CONFIG_FILE)"
	@echo "  make config-envoy  - Print generated Envoy config for $(CONFIG_FILE)"
	@echo ""
	@echo "Docker Operations:"
	@echo "  make docker-build       - Build Docker image (local)"
	@echo "  make docker-build-router - Build router runtime image (local)"
	@echo "  make docker-build-envoy - Build Envoy runtime image (local)"
	@echo "  make docker-build-dashboard - Build dashboard runtime image (local)"
	@echo "  make docker-buildx      - Build multi-platform image with buildx"
	@echo "  make docker-buildx-push - Build and push multi-platform image"
	@echo "  make docker-run         - Run container with $(CONFIG_FILE)"
	@echo "  make docker-stop        - Stop and remove container"
	@echo "  make docker-logs        - Show container logs"
	@echo "  make docker-shell       - Open shell in container"
	@echo ""
	@echo "Development:"
	@echo "  make dev-serve       - Start service (builds image if needed)"
	@echo "  make dev-test        - Validate config and render router/envoy output"
	@echo "  make rebuild-and-test - Full rebuild and test workflow"
	@echo "  make test            - Run tests"
	@echo "  make clean           - Clean up build artifacts"
	@echo ""
	@echo "Variables:"
	@echo "  IMAGE_NAME=$(IMAGE_NAME)"
	@echo "  ROUTER_IMAGE_NAME=$(ROUTER_IMAGE_NAME)"
	@echo "  ENVOY_IMAGE=$(ENVOY_IMAGE)"
	@echo "  DASHBOARD_IMAGE_NAME=$(DASHBOARD_IMAGE_NAME)"
	@echo "  RUNTIME_TOPOLOGY=split"
	@echo "  IMAGE_TAG=$(IMAGE_TAG)"
	@echo "  CONFIG_FILE=$(CONFIG_FILE)"
	@echo "  OUTPUT_DIR=$(OUTPUT_DIR)"
	@echo "  RUNTIME_CONTAINER=$(RUNTIME_CONTAINER)"
	@echo "  PLATFORMS=$(PLATFORMS)"
	@echo ""

DEV_SETUP_DEPS := docker-build docker-build-envoy docker-build-dashboard
BUILD_DEPS := docker-build docker-build-envoy docker-build-dashboard
DOCKER_RUN_DEPS := docker-build docker-build-envoy docker-build-dashboard

dev-setup: $(DEV_SETUP_DEPS) install
	@echo ""
	@echo "Development setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Edit $(CONFIG_FILE) with your configuration"
	@echo "  2. Run: make dev-serve"
	@echo ""

# Development serve - builds image if needed, then starts service
dev-serve:
	@echo "Checking Docker runtime images..."
	@if ! docker images -q $(FULL_IMAGE) 2>/dev/null | grep -q . || ! docker images -q $(FULL_ENVOY_IMAGE) 2>/dev/null | grep -q . || ! docker images -q $(FULL_DASHBOARD_IMAGE) 2>/dev/null | grep -q .; then \
		echo "One or more split runtime images are missing. Preparing..."; \
		$(MAKE) docker-build docker-build-envoy docker-build-dashboard; \
	else \
		echo "Docker images exist: $(FULL_IMAGE), $(FULL_ENVOY_IMAGE), $(FULL_DASHBOARD_IMAGE)"; \
	fi
	@echo ""
	@echo "Starting vLLM Semantic Router service..."
	@echo "Using config: $(CONFIG_FILE)"
	@echo "Topology: split"
	@echo "Router image: $(FULL_ROUTER_IMAGE)"
	@echo "Envoy image: $(FULL_ENVOY_IMAGE)"
	@echo "Dashboard image: $(FULL_DASHBOARD_IMAGE)"
	@echo ""
	VLLM_SR_IMAGE=$(FULL_IMAGE) VLLM_SR_ROUTER_IMAGE=$(FULL_ROUTER_IMAGE) VLLM_SR_ENVOY_IMAGE=$(FULL_ENVOY_IMAGE) VLLM_SR_DASHBOARD_IMAGE=$(FULL_DASHBOARD_IMAGE) python -m cli.main serve --config $(CONFIG_FILE)

# Development test - validate config and render router/envoy outputs
dev-test: install
	@echo "Testing canonical configuration rendering..."
	python -m cli.main validate --config $(CONFIG_FILE)
	python -m cli.main config router --config $(CONFIG_FILE) >/dev/null
	python -m cli.main config envoy --config $(CONFIG_FILE) >/dev/null
	@echo ""
	@echo "Configuration validated and rendered successfully!"
	@echo "  Router and Envoy config render path succeeded for: $(CONFIG_FILE)"
	@echo ""

build: $(BUILD_DEPS)

docker-build:
	@echo "Building vLLM Semantic Router Docker image (local platform)..."
	@echo "Image: $(FULL_IMAGE)"
	@echo "This may take a few minutes..."
	@echo ""
	cd ../.. && docker build -t $(FULL_IMAGE) -f src/vllm-sr/Dockerfile .
	@echo ""
	@echo "Docker image built: $(FULL_IMAGE)"
	@echo ""

docker-build-router:
	@echo "Building vLLM Semantic Router router Docker image (local platform)..."
	@echo "Image: $(FULL_ROUTER_IMAGE)"
	@echo "This may take a few minutes..."
	@echo ""
	cd ../.. && docker build -t $(FULL_ROUTER_IMAGE) -f src/vllm-sr/Dockerfile .
	@echo ""
	@echo "Docker image built: $(FULL_ROUTER_IMAGE)"
	@echo ""

docker-build-envoy:
	@echo "Ensuring official Envoy image is available locally..."
	@echo "Image: $(FULL_ENVOY_IMAGE)"
	@echo ""
	@docker image inspect $(FULL_ENVOY_IMAGE) >/dev/null 2>&1 || docker pull $(FULL_ENVOY_IMAGE)
	@echo ""
	@echo "Docker image available: $(FULL_ENVOY_IMAGE)"
	@echo ""

docker-build-dashboard:
	@echo "Building vLLM Semantic Router dashboard Docker image (local platform)..."
	@echo "Image: $(FULL_DASHBOARD_IMAGE)"
	@echo "This may take a few minutes..."
	@echo ""
	cd ../.. && docker build -t $(FULL_DASHBOARD_IMAGE) -f dashboard/backend/Dockerfile .
	@echo ""
	@echo "Docker image built: $(FULL_DASHBOARD_IMAGE)"
	@echo ""

# Setup buildx builder
docker-buildx-setup:
	@echo "Setting up Docker buildx builder..."
	@if ! docker buildx ls | grep -q $(BUILDX_BUILDER); then \
		echo "Creating buildx builder: $(BUILDX_BUILDER)"; \
		docker buildx create --name $(BUILDX_BUILDER) --use; \
		docker buildx inspect --bootstrap; \
	else \
		echo "Buildx builder exists: $(BUILDX_BUILDER)"; \
		docker buildx use $(BUILDX_BUILDER); \
	fi
	@echo ""

# Build multi-platform image (load to local)
docker-buildx: docker-buildx-setup
	@echo "Building multi-platform Docker image..."
	@echo "Image: $(FULL_IMAGE)"
	@echo "Platforms: $(PLATFORMS)"
	@echo "This may take a while..."
	@echo ""
	cd ../.. && docker buildx build \
		--platform $(PLATFORMS) \
		--tag $(FULL_IMAGE) \
		--file src/vllm-sr/Dockerfile \
		--load \
		.
	@echo ""
	@echo "Multi-platform image built: $(FULL_IMAGE)"
	@echo ""

# Build and push multi-platform image
docker-buildx-push: docker-buildx-setup
	@echo "Building and pushing multi-platform Docker image..."
	@echo "Image: $(FULL_IMAGE)"
	@echo "Platforms: $(PLATFORMS)"
	@echo "This may take a while..."
	@echo ""
	cd ../.. && docker buildx build \
		--platform $(PLATFORMS) \
		--tag $(FULL_IMAGE) \
		--file src/vllm-sr/Dockerfile \
		--push \
		.
	@echo ""
	@echo "Multi-platform image built and pushed: $(FULL_IMAGE)"
	@echo ""

install:
	@echo "Installing vLLM Semantic Router CLI in development mode..."
	pip install -e .
	@echo "CLI installed. Try: vllm-sr --version"

# Configuration commands
validate: install
	@echo "Validating configuration..."
	python -m cli.main validate --config $(CONFIG_FILE)

config-router: install
	@echo "Printing canonical router configuration..."
	python -m cli.main config router --config $(CONFIG_FILE)

config-envoy: install
	@echo "Printing generated Envoy configuration..."
	python -m cli.main config envoy --config $(CONFIG_FILE)

# Docker operations
docker-run: $(DOCKER_RUN_DEPS)
	@echo "Starting vLLM Semantic Router container..."
	@echo "Config: $(CONFIG_FILE)"
	@echo "Topology: split"
	@echo "Router image: $(FULL_ROUTER_IMAGE)"
	@echo "Envoy image: $(FULL_ENVOY_IMAGE)"
	@echo "Dashboard image: $(FULL_DASHBOARD_IMAGE)"
	VLLM_SR_IMAGE=$(FULL_IMAGE) VLLM_SR_ROUTER_IMAGE=$(FULL_ROUTER_IMAGE) VLLM_SR_ENVOY_IMAGE=$(FULL_ENVOY_IMAGE) VLLM_SR_DASHBOARD_IMAGE=$(FULL_DASHBOARD_IMAGE) python -m cli.main serve --config $(CONFIG_FILE)

docker-stop:
	@echo "Stopping vLLM Semantic Router..."
	python -m cli.main stop

docker-logs:
	@echo "Showing router logs..."
	python -m cli.main logs router --follow

docker-shell:
	@echo "Opening shell in $(RUNTIME_CONTAINER)..."
	docker exec -it $(RUNTIME_CONTAINER) /bin/bash

test:
	@echo "Running tests..."
	pytest tests/ -v

clean:
	@echo "Cleaning up..."
	rm -rf build/ dist/ *.egg-info vllm_sr.egg-info
	rm -rf $(OUTPUT_DIR)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Cleaned"

# Check if Docker image exists
check-image:
	@if docker images -q $(FULL_IMAGE) 2>/dev/null | grep -q .; then \
		echo "Docker image exists: $(FULL_IMAGE)"; \
	else \
		echo "✗ Docker image not found: $(FULL_IMAGE)"; \
		echo "  Run: make docker-build"; \
		exit 1; \
	fi
	@if docker images -q $(FULL_ENVOY_IMAGE) 2>/dev/null | grep -q .; then \
		echo "Envoy image exists: $(FULL_ENVOY_IMAGE)"; \
	else \
		echo "✗ Envoy image not found: $(FULL_ENVOY_IMAGE)"; \
		echo "  Run: make docker-build-envoy"; \
		exit 1; \
	fi
	@if docker images -q $(FULL_DASHBOARD_IMAGE) 2>/dev/null | grep -q .; then \
		echo "Dashboard image exists: $(FULL_DASHBOARD_IMAGE)"; \
	else \
		echo "✗ Dashboard image not found: $(FULL_DASHBOARD_IMAGE)"; \
		echo "  Run: make docker-build-dashboard"; \
		exit 1; \
	fi

# Rebuild and test - Full workflow from rebuild-and-test.sh
rebuild-and-test: install
	@echo "=========================================="
	@echo "Rebuild and Test vLLM Semantic Router"
	@echo "=========================================="
	@echo ""
	@echo "This will:"
	@echo "  1. Clean up old containers"
	@echo "  2. Rebuild Docker images with all dependencies"
	@echo "  3. Start the service"
	@echo "  4. Verify multi-listener support"
	@echo ""
	@echo "Dependencies included in the image:"
	@echo "  router binary"
	@echo "  setup-mode YAML helper"
	@echo "  router health probe tooling"
	@echo ""
	@read -p "Press Enter to continue or Ctrl+C to cancel..." dummy
	@echo ""
	@echo "1. Cleaning up old containers..."
	@docker rm -f vllm-sr-router-container vllm-sr-envoy-container vllm-sr-dashboard-container vllm-sr-sim-container vllm-sr-prometheus vllm-sr-grafana vllm-sr-jaeger 2>/dev/null || echo "  No containers to remove"
	@echo ""
	@echo "2. Rebuilding Docker images..."
	@echo "  Building from: $$(cd ../.. && pwd)"
	@echo "  Note: Use 'make docker-buildx' for multi-platform builds"
	@echo ""
	@$(MAKE) docker-build docker-build-envoy docker-build-dashboard
	@echo ""
	@echo "3. Verifying dependencies in images..."
	@docker run --rm --entrypoint /bin/sh $(FULL_ROUTER_IMAGE) -c " \
		echo '  Checking router binary...'; \
		test -x /usr/local/bin/router && echo '    router binary present'; \
		echo '  Checking setup helper yaml import...'; \
		python3 -c 'import yaml; print(\"    yaml import ok\")'; \
		echo '  Checking curl...'; \
		curl --version | head -1 | sed 's/^/    /'; \
	"
	@docker run --rm --entrypoint /bin/sh $(FULL_ENVOY_IMAGE) -c " \
		echo '  Checking envoy binary...'; \
		envoy --version | head -1 | sed 's/^/    /'; \
	"
	@docker run --rm --entrypoint /bin/sh $(FULL_DASHBOARD_IMAGE) -c " \
		echo '  Checking dashboard backend...'; \
		/usr/local/bin/dashboard-backend -help >/dev/null && echo '    dashboard backend present'; \
		echo '  Checking auth bootstrap endpoint...'; \
		mkdir -p /tmp/static /tmp/data; \
		printf '<!doctype html><title>ok</title>' > /tmp/static/index.html; \
		printf 'version: v0.3\nlisteners:\n  - name: http\n    address: 0.0.0.0\n    port: 8899\n' > /tmp/config.yaml; \
		/usr/local/bin/dashboard-backend -port=8700 -static=/tmp/static -config=/tmp/config.yaml -auth-db /tmp/data/auth.db -auth-jwt-secret test-secret >/tmp/dashboard.log 2>&1 & \
		pid=\$$!; \
		for i in 1 2 3 4 5; do \
			if curl -fsS http://127.0.0.1:8700/api/auth/bootstrap/can-register >/tmp/auth.json 2>/dev/null; then break; fi; \
			sleep 1; \
		done; \
		grep -q '\"canRegister\":true' /tmp/auth.json && echo '    auth bootstrap endpoint ok' || { cat /tmp/dashboard.log; exit 1; }; \
		kill \$$pid >/dev/null 2>&1 || true; \
		wait \$$pid >/dev/null 2>&1 || true; \
		echo '  Checking Docker CLI...'; \
		docker --version | sed 's/^/    /'; \
		echo '  Checking runtime helper import...'; \
		python3 -c 'import cli.commands.runtime_support; print(\"    runtime helper import ok\")'; \
	"
	@echo ""
	@echo "4. Checking config.yaml listeners..."
	@python -c " \
import yaml; \
with open('$(CONFIG_FILE)', 'r') as f: \
    config = yaml.safe_load(f); \
    listeners = config.get('listeners', []); \
    print(f'  Found {len(listeners)} listener(s):'); \
    [print(f\"    - {l.get('name', 'unknown')}: port {l.get('port', 'unknown')}\") for l in listeners]; \
	"
	@echo ""
	@echo "5. Starting service with local images..."
	@VLLM_SR_IMAGE=$(FULL_IMAGE) VLLM_SR_ROUTER_IMAGE=$(FULL_ROUTER_IMAGE) VLLM_SR_ENVOY_IMAGE=$(FULL_ENVOY_IMAGE) VLLM_SR_DASHBOARD_IMAGE=$(FULL_DASHBOARD_IMAGE) python -m cli.main serve --config $(CONFIG_FILE) --image-pull-policy never &
	@echo ""
	@echo "6. Waiting for services to start (30 seconds)..."
	@sleep 30
	@echo ""
	@echo "7. Service status:"
	@python -m cli.main status || true
	@echo ""
	@echo "8. Managed containers:"
	@docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep 'vllm-sr-' || true
	@echo ""
	@echo "9. Recent router logs:"
	@python -m cli.main logs router || true
	@echo ""
	@echo "=========================================="
	@echo "Next Steps"
	@echo "=========================================="
	@echo ""
	@echo "Check health:"
	@echo "  curl http://localhost:8000/healthz"
	@echo "  curl http://localhost:8080/healthz"
	@echo ""
	@echo "View logs:"
	@echo "  python -m cli.main logs router --follow"
	@echo ""
	@echo "Open shell:"
	@echo "  make docker-shell RUNTIME_CONTAINER=vllm-sr-router-container"
	@echo ""
	@echo "Stop service:"
	@echo "  make docker-stop"
	@echo ""
