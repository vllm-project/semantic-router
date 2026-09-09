##@ Pre-commit

PRECOMMIT_CONTAINER := ghcr.io/vllm-project/semantic-router/precommit:latest

AGENT_PRE_COMMIT ?= $(AGENT_VENV)/bin/pre-commit

precommit-install: ## Install the repo-local pre-commit hook using this worktree's tooling
precommit-install:
	@$(MAKE) harness-venv-install
	@echo "Installing repo-local git hooks (pre-commit)..."
	@"$(AGENT_PRE_COMMIT)" install --hook-type pre-commit

precommit-check: harness-venv-install ## Run pre-commit checks on all relevant files
	@echo "Running pre-commit on all tracked files..."
	@"$(AGENT_PRE_COMMIT)" run --all-files

# Run the CI changed-file pre-commit pipeline in a Docker container.
#
# For interactive debugging:
#   export PRECOMMIT_CONTAINER=ghcr.io/vllm-project/semantic-router/precommit:latest
#   docker run --rm -it \
#       -v $(pwd):/app \
#       -w /app \
#       --name precommit-container ${PRECOMMIT_CONTAINER} \
#       bash
precommit-local: ## Run CI changed-file checks in a Docker/Podman container
precommit-local:
	@set -e; \
	WORKTREE="$(CURDIR)"; \
	GIT_COMMON_DIR=$$(git rev-parse --path-format=absolute --git-common-dir); \
	set --; \
	if [ -n "$(CHANGED_FILES_PATH)" ]; then \
		CHANGED_FILE_LIST=$$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$(CHANGED_FILES_PATH)"); \
		if [ ! -f "$$CHANGED_FILE_LIST" ]; then echo "Changed-file list does not exist: $$CHANGED_FILE_LIST" >&2; exit 1; fi; \
		set -- -v "$$CHANGED_FILE_LIST:/tmp/harness-changed-files.txt:ro" -e CHANGED_FILES_PATH=/tmp/harness-changed-files.txt; \
	fi; \
	if command -v docker > /dev/null 2>&1; then \
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
	$$CONTAINER_CMD run --rm "$$@" \
	    -e BASE_REF="$(BASE_REF)" \
	    -e CHANGED_FILES="$(CHANGED_FILES)" \
	    -e GIT_CONFIG_COUNT=1 \
	    -e GIT_CONFIG_KEY_0=safe.directory \
	    -e GIT_CONFIG_VALUE_0="$$WORKTREE" \
	    -e SKIP_MODEL_DEPENDENT_TESTS=true \
	    -v "$$WORKTREE:$$WORKTREE" \
	    -v "$$GIT_COMMON_DIR:$$GIT_COMMON_DIR:ro" \
	    -v "$$WORKTREE/.venv-agent" \
	    -w "$$WORKTREE" \
	    ${PRECOMMIT_CONTAINER} bash -c 'make check BASE_REF="$$BASE_REF"'
