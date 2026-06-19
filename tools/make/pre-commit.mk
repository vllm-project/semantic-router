##@ Pre-commit

PRECOMMIT_CONTAINER := ghcr.io/vllm-project/semantic-router/precommit:latest

AGENT_PRE_COMMIT ?= $(CURDIR)/.venv-agent/bin/pre-commit

precommit-install: ## Install the repo-local pre-commit hook into the agent harness venv
precommit-install:
	@$(MAKE) agent-venv-install
	@echo "Installing repo-local git hooks (pre-commit)..."
	@"$(AGENT_PRE_COMMIT)" install --hook-type pre-commit

precommit-branch-gate: agent-venv-install ## Run the local branch prelint bundle on demand
	@$(MAKE) agent-ci-lint AGENT_BASE_REF="$(AGENT_BASE_REF)"
	@$(MAKE) precommit-check

precommit-check: agent-venv-install ## Run pre-commit checks on all relevant files
	@echo "Running pre-commit on all tracked files..."
	@"$(AGENT_PRE_COMMIT)" run --all-files

# Run the full CI pre-commit pipeline in a Docker container.
# This mirrors .github/workflows/pre-commit.yml by running both
# `make precommit-branch-gate` and `make precommit-check` inside the
# containerized toolchain.
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
	IMAGE_SOURCE="freshly pulled image"; \
	echo "Refreshing ${PRECOMMIT_CONTAINER}..."; \
	if $$CONTAINER_CMD pull ${PRECOMMIT_CONTAINER}; then \
		:; \
	else \
		echo "Warning: failed to pull ${PRECOMMIT_CONTAINER}; checking for a cached image..." >&2; \
		IMAGE_SOURCE="cached local image"; \
	fi; \
	if ! $$CONTAINER_CMD image inspect ${PRECOMMIT_CONTAINER} > /dev/null 2>&1; then \
		echo "Error: ${PRECOMMIT_CONTAINER} is unavailable locally and could not be pulled."; \
		exit 1; \
	fi; \
	IMAGE_REF=$$($$CONTAINER_CMD image inspect ${PRECOMMIT_CONTAINER} --format '{{if .RepoDigests}}{{index .RepoDigests 0}}{{else}}{{.Id}}{{end}}' 2>/dev/null); \
	if [ -z "$$IMAGE_REF" ]; then \
		IMAGE_REF=${PRECOMMIT_CONTAINER}; \
	fi; \
	echo "Using $$IMAGE_SOURCE: $$IMAGE_REF"; \
	$$CONTAINER_CMD run --rm \
	    -e AGENT_BASE_REF="$(AGENT_BASE_REF)" \
	    -v $(shell pwd):/app \
	    -w /app \
	    ${PRECOMMIT_CONTAINER} bash -c 'make precommit-branch-gate'
