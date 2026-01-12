.PHONY: help build install test clean docker-build docker-run docker-stop dev-setup dev-serve dev-test rebuild-and-test

# Configuration
IMAGE_NAME ?= ghcr.io/vllm-project/semantic-router/vllm-sr
IMAGE_TAG ?= latest
FULL_IMAGE = $(IMAGE_NAME):$(IMAGE_TAG)
CONFIG_FILE ?= config.yaml
OUTPUT_DIR ?= .vllm-sr
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
	@echo "  make build        - Build Docker image ($(FULL_IMAGE))"
	@echo "  make install      - Install CLI in development mode"
	@echo "  make dev-setup    - Complete setup (build + install)"
	@echo ""
	@echo "Configuration:"
	@echo "  make generate     - Generate configurations from $(CONFIG_FILE)"
	@echo "  make validate     - Validate $(CONFIG_FILE)"
	@echo "  make show-config  - Show all configurations"
	@echo ""
	@echo "Docker Operations:"
	@echo "  make docker-build       - Build Docker image (local)"
	@echo "  make docker-buildx      - Build multi-platform image with buildx"
	@echo "  make docker-buildx-push - Build and push multi-platform image"
	@echo "  make docker-run         - Run container with $(CONFIG_FILE)"
	@echo "  make docker-stop        - Stop and remove container"
	@echo "  make docker-logs        - Show container logs"
	@echo "  make docker-shell       - Open shell in container"
	@echo ""
	@echo "Development:"
	@echo "  make dev-serve       - Start service (builds image if needed)"
	@echo "  make dev-test        - Test config generation"
	@echo "  make rebuild-and-test - Full rebuild and test workflow"
	@echo "  make test            - Run tests"
	@echo "  make clean           - Clean up build artifacts"
	@echo ""
	@echo "Variables:"
	@echo "  IMAGE_NAME=$(IMAGE_NAME)"
	@echo "  IMAGE_TAG=$(IMAGE_TAG)"
	@echo "  CONFIG_FILE=$(CONFIG_FILE)"
	@echo "  OUTPUT_DIR=$(OUTPUT_DIR)"
	@echo "  PLATFORMS=$(PLATFORMS)"
	@echo ""

# Quick development setup
dev-setup: docker-build install
	@echo ""
	@echo "✓ Development setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Edit $(CONFIG_FILE) with your configuration"
	@echo "  2. Run: make dev-serve"
	@echo ""

# Development serve - builds image if needed, then starts service
dev-serve:
	@echo "Checking Docker image..."
	@if ! docker images -q $(FULL_IMAGE) 2>/dev/null | grep -q .; then \
		echo "Docker image not found. Building..."; \
		$(MAKE) docker-build; \
	else \
		echo "✓ Docker image exists: $(FULL_IMAGE)"; \
	fi
	@echo ""
	@echo "Starting vLLM Semantic Router service..."
	@echo "Using config: $(CONFIG_FILE)"
	@echo "Docker image: $(FULL_IMAGE)"
	@echo ""
	python -m cli.main serve $(CONFIG_FILE) --image $(FULL_IMAGE)

# Development test - just generate configs
dev-test: install
	@echo "Testing configuration generation..."
	python -m cli.main generate $(CONFIG_FILE) --output-dir $(OUTPUT_DIR) --force
	@echo ""
	@echo "✓ Configuration generated successfully!"
	@echo "  Output directory: $(OUTPUT_DIR)"
	@ls -la $(OUTPUT_DIR)/
	@echo ""

build: docker-build

docker-build:
	@echo "Building vLLM Semantic Router Docker image (local platform)..."
	@echo "Image: $(FULL_IMAGE)"
	@echo "This may take a few minutes..."
	@echo ""
	cd ../.. && docker build -t $(FULL_IMAGE) -f src/vllm-sr/Dockerfile .
	@echo ""
	@echo "✓ Docker image built: $(FULL_IMAGE)"
	@echo ""

# Setup buildx builder
docker-buildx-setup:
	@echo "Setting up Docker buildx builder..."
	@if ! docker buildx ls | grep -q $(BUILDX_BUILDER); then \
		echo "Creating buildx builder: $(BUILDX_BUILDER)"; \
		docker buildx create --name $(BUILDX_BUILDER) --use; \
		docker buildx inspect --bootstrap; \
	else \
		echo "✓ Buildx builder exists: $(BUILDX_BUILDER)"; \
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
	@echo "✓ Multi-platform image built: $(FULL_IMAGE)"
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
	@echo "✓ Multi-platform image built and pushed: $(FULL_IMAGE)"
	@echo ""

install:
	@echo "Installing vLLM Semantic Router CLI in development mode..."
	pip install -e .
	@echo "✓ CLI installed. Try: vllm-sr --version"

# Configuration commands
generate: install
	@echo "Generating configurations..."
	python -m cli.main generate $(CONFIG_FILE) --output-dir $(OUTPUT_DIR) --force

validate: install
	@echo "Validating configuration..."
	python -m cli.main validate $(CONFIG_FILE)

show-config: install
	@echo "Showing configurations..."
	python -m cli.main show-config $(CONFIG_FILE) --output-dir $(OUTPUT_DIR)

# Docker operations
docker-run: docker-build
	@echo "Starting vLLM Semantic Router container..."
	@echo "Config: $(CONFIG_FILE)"
	@echo "Image: $(FULL_IMAGE)"
	python -m cli.main serve $(CONFIG_FILE) --image $(FULL_IMAGE)

docker-stop:
	@echo "Stopping vLLM Semantic Router..."
	python -m cli.main stop

docker-logs:
	@echo "Showing container logs..."
	docker logs -f vllm-sr-container

docker-shell:
	@echo "Opening shell in container..."
	docker exec -it vllm-sr-container /bin/bash

test:
	@echo "Running tests..."
	pytest tests/ -v

clean:
	@echo "Cleaning up..."
	rm -rf build/ dist/ *.egg-info vllm_sr.egg-info
	rm -rf $(OUTPUT_DIR)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleaned"

# Check if Docker image exists
check-image:
	@if docker images -q $(FULL_IMAGE) 2>/dev/null | grep -q .; then \
		echo "✓ Docker image exists: $(FULL_IMAGE)"; \
	else \
		echo "✗ Docker image not found: $(FULL_IMAGE)"; \
		echo "  Run: make docker-build"; \
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
	@echo "  2. Rebuild Docker image with all dependencies"
	@echo "  3. Start the service"
	@echo "  4. Verify multi-listener support"
	@echo ""
	@echo "Dependencies included in the image:"
	@echo "  ✓ pydantic>=2.0.0"
	@echo "  ✓ huggingface_hub[cli]>=0.20.0"
	@echo "  ✓ Multi-listener support"
	@echo ""
	@read -p "Press Enter to continue or Ctrl+C to cancel..." dummy
	@echo ""
	@echo "1. Cleaning up old containers..."
	@docker rm -f vllm-sr-container 2>/dev/null || echo "  No container to remove"
	@echo ""
	@echo "2. Rebuilding Docker image..."
	@echo "  Building from: $$(cd ../.. && pwd)"
	@echo "  Note: Use 'make docker-buildx' for multi-platform builds"
	@echo ""
	@$(MAKE) docker-build
	@echo ""
	@echo "3. Verifying dependencies in image..."
	@docker run --rm $(FULL_IMAGE) /bin/sh -c " \
		echo '  Checking pydantic...'; \
		python -c 'import pydantic; print(f\"    ✓ pydantic {pydantic.__version__}\")'; \
		echo '  Checking huggingface_hub...'; \
		python -c 'import huggingface_hub; print(f\"    ✓ huggingface_hub {huggingface_hub.__version__}\")'; \
		echo '  Checking huggingface-cli...'; \
		huggingface-cli --version | sed 's/^/    ✓ /'; \
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
	@echo "5. Starting service with local image..."
	@python -m cli.main serve $(CONFIG_FILE) --image $(FULL_IMAGE) --image-pull-policy never &
	@echo ""
	@echo "6. Waiting for services to start (30 seconds)..."
	@sleep 30
	@echo ""
	@echo "7. Container status:"
	@docker ps --filter name=vllm-sr-container --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
	@echo ""
	@echo "8. Recent logs:"
	@docker logs vllm-sr-container --tail 100
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
	@echo "  docker logs -f vllm-sr-container"
	@echo ""
	@echo "Stop service:"
	@echo "  make docker-stop"
	@echo ""

