##@ Pre-commit

PRECOMMIT_CONTAINER := ghcr.io/vllm-project/semantic-router/precommit:latest

precommit-install: ## Install pre-commit Python package
	pip install pre-commit

precommit-check: ## Run pre-commit checks on all relevant files
	@FILES=$$(find . -type f \( -name "*.go" -o -name "*.rs" -o -name "*.py" -o -name "*.js" -o -name "*.md" -o -name "*.yaml" -o -name "*.yml" \) \
		! -path "./target/*" \
		! -path "./candle-binding/target/*" \
		! -path "./.git/*" \
		! -path "./node_modules/*" \
		! -path "./vendor/*" \
		! -path "./__pycache__/*" \
		! -path "./site/*" \
		! -name "*.pb.go" \
		| tr '\n' ' '); \
	if [ -n "$$FILES" ]; then \
		echo "Running pre-commit on files: $$FILES"; \
		pre-commit run --files $$FILES; \
	else \
		echo "No Go, Rust, JavaScript, Markdown, Yaml, or Python files found to check"; \
	fi

precommit-local: ## Run pre-commit hooks in a Docker/Podman container
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
