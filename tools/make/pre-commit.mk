##@ Pre-commit

PRECOMMIT_CONTAINER := ghcr.io/vllm-project/semantic-router/precommit:latest

precommit-install: ## Install pre-commit Python package
precommit-install:
	pip install pre-commit

precommit-check: ## Run pre-commit checks on all relevant files
precommit-check:
	@FILES=$$(find . -type f \( -name "*.go" -o -name "*.rs" -o -name "*.py" -o -name "*.js" -o -name "*.sh" -o -name "*.md" -o -name "*.yaml" -o -name "*.yml" \) \
		! -path "./target/*" \
		! -path "./candle-binding/target/*" \
    ! -path "./website/node_modules/*" \
    ! -path "./frontend/node_modules/*" \
    ! -path "./website/.docusaurus/*" \
		! -path "./__pycache__/*" \
    ! -path "./.venv/*" \
    ! -path "./vendor/*" \
		! -name "*.pb.go" \
    ! -path "./.git/*" \
		| tr '\n' ' '); \
	if [ -n "$$FILES" ]; then \
		FILE_COUNT=$$(echo $$FILES | wc -w | tr -d ' '); \
		echo "Running pre-commit on $$FILE_COUNT files..."; \
		pre-commit run --files $$FILES; \
	else \
		echo "No Go, Rust, JavaScript, Shell, Markdown, Yaml, or Python files found to check"; \
	fi

# Run pre-commit hooks in a Docker container,
# and you can exec container to run bash for debug.
# export PRECOMMIT_CONTAINER=ghcr.io/vllm-project/semantic-router/precommit:latest
# docker run --rm -it \
#     -v $(pwd):/app \
#     -w /app \
#     --name precommit-container ${PRECOMMIT_CONTAINER} \
#     bash
# and then, run `pre-commit install && pre-commit run --all-files` command
precommit-local: ## Run pre-commit hooks in a Docker/Podman container
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
	    ${PRECOMMIT_CONTAINER} bash -c 'pre-commit install && pre-commit run --all-files'
