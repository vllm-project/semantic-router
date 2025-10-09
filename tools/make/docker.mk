# ======== docker.mk ========
# = Docker build and management =
# ======== docker.mk ========

# Docker image tags
DOCKER_REGISTRY ?= ghcr.io/vllm-project/semantic-router
DOCKER_TAG ?= latest

# Default docker compose environment
# Point Compose to the relocated main stack by default; override by exporting COMPOSE_FILE
export COMPOSE_FILE ?= deploy/docker-compose/docker-compose.yml
# Keep a stable project name so network/volume names are predictable across runs
export COMPOSE_PROJECT_NAME ?= semantic-router

# Build all Docker images
docker-build-all: docker-build-extproc docker-build-llm-katan docker-build-precommit
	@$(LOG_TARGET)
	@echo "All Docker images built successfully"

# Build extproc Docker image
docker-build-extproc:
	@$(LOG_TARGET)
	@echo "Building extproc Docker image..."
	@$(CONTAINER_RUNTIME) build -f Dockerfile.extproc -t $(DOCKER_REGISTRY)/extproc:$(DOCKER_TAG) .

# Build llm-katan Docker image
docker-build-llm-katan:
	@$(LOG_TARGET)
	@echo "Building llm-katan Docker image..."
	@$(CONTAINER_RUNTIME) build -f e2e-tests/llm-katan/Dockerfile -t $(DOCKER_REGISTRY)/llm-katan:$(DOCKER_TAG) e2e-tests/llm-katan/

# Build precommit Docker image
docker-build-precommit:
	@$(LOG_TARGET)
	@echo "Building precommit Docker image..."
	@$(CONTAINER_RUNTIME) build -f Dockerfile.precommit -t $(DOCKER_REGISTRY)/precommit:$(DOCKER_TAG) .

# Test llm-katan Docker image locally
docker-test-llm-katan:
	@$(LOG_TARGET)
	@echo "Testing llm-katan Docker image..."
	@curl -f http://localhost:8000/v1/models || (echo "Models endpoint failed" && exit 1)
	@echo "\n✅ llm-katan Docker image test passed"

# Run llm-katan Docker image locally
docker-run-llm-katan: docker-build-llm-katan
	@$(LOG_TARGET)
	@echo "Running llm-katan Docker image on port 8000..."
	@echo "Access the server at: http://localhost:8000"
	@echo "Press Ctrl+C to stop"
	@$(CONTAINER_RUNTIME) run --rm -p 8000:8000 $(DOCKER_REGISTRY)/llm-katan:$(DOCKER_TAG)

# Run llm-katan with custom served model name
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
docker-clean:
	@$(LOG_TARGET)
	@echo "Cleaning up Docker images..."
	@$(CONTAINER_RUNTIME) image prune -f
	@echo "Docker cleanup completed"

# Push Docker images (for CI/CD)
docker-push-all: docker-push-extproc docker-push-llm-katan
	@$(LOG_TARGET)
	@echo "All Docker images pushed successfully"

docker-push-extproc:
	@$(LOG_TARGET)
	@echo "Pushing extproc Docker image..."
	@$(CONTAINER_RUNTIME) push $(DOCKER_REGISTRY)/extproc:$(DOCKER_TAG)

docker-push-llm-katan:
	@$(LOG_TARGET)
	@echo "Pushing llm-katan Docker image..."
	@$(CONTAINER_RUNTIME) push $(DOCKER_REGISTRY)/llm-katan:$(DOCKER_TAG)

# Docker compose build flag logic
# Usage: make docker-compose-up REBUILD=1  (forces image rebuild)
BUILD_FLAG=$(if $(REBUILD),--build,)

# Docker compose shortcuts (no rebuild by default)
docker-compose-up:
	@$(LOG_TARGET)
	@echo "Starting services with docker-compose (REBUILD=$(REBUILD))..."
	@docker compose up -d $(BUILD_FLAG)

docker-compose-up-testing:
	@$(LOG_TARGET)
	@echo "Starting services with testing profile (REBUILD=$(REBUILD))..."
	@docker compose --profile testing up -d $(BUILD_FLAG)

docker-compose-up-llm-katan:
	@$(LOG_TARGET)
	@echo "Starting services with llm-katan profile (REBUILD=$(REBUILD))..."
	@docker compose --profile llm-katan up -d $(BUILD_FLAG)

# Explicit rebuild targets for convenience
docker-compose-rebuild: REBUILD=1
docker-compose-rebuild: docker-compose-up

docker-compose-rebuild-testing: REBUILD=1
docker-compose-rebuild-testing: docker-compose-up-testing

docker-compose-rebuild-llm-katan: REBUILD=1
docker-compose-rebuild-llm-katan: docker-compose-up-llm-katan

docker-compose-down:
	@$(LOG_TARGET)
	@echo "Stopping docker-compose services..."
	@docker compose down

# Help target for Docker commands
docker-help:
	@echo "Docker Make Targets:"
	@echo "  docker-build-all          - Build all Docker images"
	@echo "  docker-build-extproc      - Build extproc Docker image"
	@echo "  docker-build-llm-katan    - Build llm-katan Docker image"
	@echo "  docker-build-precommit    - Build precommit Docker image"
	@echo "  docker-test-llm-katan     - Test llm-katan Docker image"
	@echo "  docker-run-llm-katan      - Run llm-katan Docker image locally"
	@echo "  docker-run-llm-katan-custom SERVED_NAME=name - Run with custom served model name"
	@echo "  docker-clean              - Clean up Docker images"
	@echo "  docker-compose-up                    - Start services (add REBUILD=1 to rebuild)"
	@echo "  docker-compose-up-testing            - Start with testing profile (REBUILD=1 optional)"
	@echo "  docker-compose-up-llm-katan          - Start with llm-katan profile (REBUILD=1 optional)"
	@echo "  docker-compose-rebuild               - Force rebuild then start"
	@echo "  docker-compose-rebuild-testing       - Force rebuild (testing profile)"
	@echo "  docker-compose-rebuild-llm-katan     - Force rebuild (llm-katan profile)"
	@echo "  docker-compose-down                  - Stop docker-compose services"
	@echo ""
	@echo "Environment Variables:"
	@echo "  DOCKER_REGISTRY - Docker registry (default: ghcr.io/vllm-project/semantic-router)"
	@echo "  DOCKER_TAG      - Docker tag (default: latest)"
	@echo "  SERVED_NAME     - Served model name for custom runs"
