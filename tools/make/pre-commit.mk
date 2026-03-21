##@ Pre-commit

PRECOMMIT_CONTAINER := ghcr.io/vllm-project/semantic-router/precommit:latest

precommit-install: ## Install pre-commit Python package
precommit-install:
	pip install pre-commit

precommit-check: ## Run pre-commit checks on all relevant files
precommit-check:
	@echo "Running pre-commit on all tracked files..."
	@pre-commit run --all-files

# Run the full CI pre-commit pipeline in a Docker container.
# This mirrors .github/workflows/pre-commit.yml by running both
# `make agent-ci-lint` and `make precommit-check`.
#
# For interactive debugging:
#   export PRECOMMIT_CONTAINER=ghcr.io/vllm-project/semantic-router/precommit:latest
#   docker run --rm -it \
#       -v $(pwd):/app \
#       -w /app \
#       --name precommit-container ${PRECOMMIT_CONTAINER} \
#       bash
precommit-local: ## Run full CI pre-commit pipeline in a Docker/Podman container
precommit-local:
	@if command -v docker > /dev/null 2>&1; then \
		CONTAINER_CMD=docker; \
	elif command -v podman > /dev/null 2>&1; then \
		CONTAINER_CMD=podman; \
	else \
		echo "Error: Neither docker nor podman is installed. Please install one of them."; \
		exit 1; \
	fi; \
	if ! $$CONTAINER_CMD image inspect ${PRECOMMIT_CONTAINER} > /dev/null 2>&1; then \
		echo "Image not found locally. Pulling..."; \
		$$CONTAINER_CMD pull ${PRECOMMIT_CONTAINER}; \
	else \
		echo "Image found locally. Skipping pull."; \
	fi; \
	$$CONTAINER_CMD run --rm \
	    -v $(shell pwd):/app \
	    -w /app \
	    ${PRECOMMIT_CONTAINER} bash -c 'make agent-ci-lint && make precommit-check'
