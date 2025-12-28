.PHONY: help build install test clean docker-build docker-run docker-stop dev-setup dev-serve dev-test

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
	@echo "  make dev-setup    - Complete setup (build + install)"
	@echo "  make dev-serve    - Start service with config"
	@echo "  make dev-test     - Test configuration generation"
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
	@echo "  make dev-serve    - Start service (builds image if needed)"
	@echo "  make dev-test     - Test config generation"
	@echo "  make test         - Run tests"
	@echo "  make clean        - Clean up build artifacts"
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

