# ========================== agent.mk ==========================
# = Coding-agent entry points and gates                       =
# ============================================================

##@ Agent

ENV ?= cpu
CHANGED_FILES ?=
AGENT_BASE_REF ?=
AGENT_SERVE_CONFIG ?=
AGENT_SERVE_ARGS ?=
AGENT_SMOKE_TIMEOUT ?= 90
AGENT_STACK_NAME ?=
AGENT_PORT_OFFSET ?= 0

agent-help: ## Show help for agent-specific targets
	@echo "Agent commands:"
	@echo "  make agent-bootstrap"
	@echo "  make agent-validate"
	@echo "  make agent-scorecard"
	@echo "  make agent-dev ENV=cpu|amd"
	@echo "  make agent-serve-local ENV=cpu|amd"
	@echo "    optional: AGENT_STACK_NAME=<name> AGENT_PORT_OFFSET=<n>"
	@echo "  make agent-report ENV=cpu|amd CHANGED_FILES=\"...\""
	@echo "  make agent-lint CHANGED_FILES=\"...\""
	@echo "  make agent-fast-gate CHANGED_FILES=\"...\""
	@echo "  make agent-ci-gate CHANGED_FILES=\"...\""
	@echo "  make agent-e2e-affected CHANGED_FILES=\"...\""
	@echo "  make agent-feature-gate ENV=cpu|amd CHANGED_FILES=\"...\""

agent-bootstrap: ## Install agent validation tooling
	@$(LOG_TARGET)
	@echo "Installing agent Python tooling..."
	@python3 -m pip install -r tools/agent/requirements.txt
	@python3 -m pip install pre-commit
	@python3 -m pip install yamllint codespell
	@if command -v npm >/dev/null 2>&1; then \
		npm install -g markdownlint-cli@0.43.0 >/dev/null 2>&1 || true; \
	fi
	@if ! command -v golangci-lint >/dev/null 2>&1 && command -v go >/dev/null 2>&1; then \
		echo "Installing golangci-lint..."; \
		go install github.com/golangci/golangci-lint/cmd/golangci-lint@v2.5.0; \
	fi
	@if command -v rustup >/dev/null 2>&1; then \
		rustup component add clippy >/dev/null 2>&1 || true; \
	fi
	@echo "Agent tooling installed"

agent-validate: agent-bootstrap ## Validate the shared agent harness manifests and docs
	@$(LOG_TARGET)
	@python3 tools/agent/scripts/agent_gate.py validate

agent-scorecard: agent-bootstrap ## Show the current harness governance scorecard
	@$(LOG_TARGET)
	@python3 tools/agent/scripts/agent_gate.py scorecard --format summary

agent-dev: ## Build the canonical local development image for the selected environment
	@$(LOG_TARGET)
	@if [ "$(ENV)" = "amd" ]; then \
		$(MAKE) vllm-sr-dev VLLM_SR_PLATFORM=amd; \
	else \
		$(MAKE) vllm-sr-dev; \
	fi

agent-serve-local: ## Start vllm-sr with the canonical local image flow
	@$(LOG_TARGET)
	@DEFAULT_CONFIG="$$(python3 tools/agent/scripts/agent_gate.py resolve-env --env "$(ENV)" --field smoke_config)"; \
	CONFIG_PATH="$(AGENT_SERVE_CONFIG)"; \
	if [ -z "$$CONFIG_PATH" ]; then \
		CONFIG_PATH="$$DEFAULT_CONFIG"; \
	fi; \
	CONFIG_ARGS=""; \
	if [ -n "$$CONFIG_PATH" ]; then \
		CONFIG_ARGS="--config $$CONFIG_PATH"; \
	fi; \
	if [ "$(ENV)" = "amd" ]; then \
		echo "Starting local AMD workflow..."; \
		VLLM_SR_STACK_NAME="$(AGENT_STACK_NAME)" VLLM_SR_PORT_OFFSET="$(AGENT_PORT_OFFSET)" \
		vllm-sr serve --image-pull-policy never --platform amd $$CONFIG_ARGS $(AGENT_SERVE_ARGS); \
	else \
		echo "Starting local CPU workflow..."; \
		VLLM_SR_STACK_NAME="$(AGENT_STACK_NAME)" VLLM_SR_PORT_OFFSET="$(AGENT_PORT_OFFSET)" \
		vllm-sr serve --image-pull-policy never $$CONFIG_ARGS $(AGENT_SERVE_ARGS); \
	fi

agent-stop-local: ## Stop local vllm-sr services
	@$(LOG_TARGET)
	@VLLM_SR_STACK_NAME="$(AGENT_STACK_NAME)" VLLM_SR_PORT_OFFSET="$(AGENT_PORT_OFFSET)" vllm-sr stop || true

agent-lint: agent-bootstrap ## Run lint and structure gates for changed files
	@$(LOG_TARGET)
	@RAW_FILES="$$(python3 tools/agent/scripts/agent_gate.py changed-files --base-ref "$(AGENT_BASE_REF)" --changed-files "$(CHANGED_FILES)")"; \
	if [ -z "$$RAW_FILES" ]; then \
		echo "No changed files detected."; \
		exit 0; \
	fi; \
	FILE_ARGS="$$(printf '%s\n' "$$RAW_FILES" | paste -sd' ' -)"; \
	CSV_FILES="$$(printf '%s\n' "$$RAW_FILES" | paste -sd',' -)"; \
	echo "Running baseline pre-commit checks..."; \
	pre-commit run --files $$FILE_ARGS; \
	echo "Running Python lint..."; \
	python3 tools/agent/scripts/agent_gate.py run-python-lint --changed-files "$$CSV_FILES"; \
		echo "Running Go structural lint..."; \
		python3 tools/agent/scripts/agent_gate.py run-go-lint --base-ref "$(AGENT_BASE_REF)" --changed-files "$$CSV_FILES"; \
	echo "Running Rust lint..."; \
	python3 tools/agent/scripts/agent_gate.py run-rust-lint --changed-files "$$CSV_FILES"; \
	echo "Running structure checks..."; \
	python3 tools/agent/scripts/structure_check.py --base-ref "$(AGENT_BASE_REF)" $$FILE_ARGS

agent-fast-gate: ## Run the fast gate: manifest validation, lint, and lightweight tests
	@$(LOG_TARGET)
	@$(MAKE) agent-validate
	@$(MAKE) agent-lint CHANGED_FILES="$(CHANGED_FILES)" AGENT_BASE_REF="$(AGENT_BASE_REF)"
	@python3 tools/agent/scripts/agent_gate.py run-tests --mode fast --base-ref "$(AGENT_BASE_REF)" --changed-files "$(CHANGED_FILES)"

agent-report: ## Show primary skill, impacted surfaces, and validation commands
	@$(LOG_TARGET)
	@python3 tools/agent/scripts/agent_gate.py report --env "$(ENV)" --base-ref "$(AGENT_BASE_REF)" --changed-files "$(CHANGED_FILES)"

agent-ci-gate: ## Run the repo-standard fast CI gate
	@$(LOG_TARGET)
	@$(MAKE) agent-report ENV="$(ENV)" CHANGED_FILES="$(CHANGED_FILES)" AGENT_BASE_REF="$(AGENT_BASE_REF)"
	@python3 tools/agent/scripts/agent_gate.py resolve --base-ref "$(AGENT_BASE_REF)" --changed-files "$(CHANGED_FILES)" --format summary
	@$(MAKE) agent-fast-gate CHANGED_FILES="$(CHANGED_FILES)" AGENT_BASE_REF="$(AGENT_BASE_REF)"

agent-smoke-local: ## Validate local container, router, envoy, and dashboard health
	@$(LOG_TARGET)
	@STACK_CONTAINER="$(VLLM_SR_CONTAINER)"; \
	if [ -n "$(AGENT_STACK_NAME)" ] && [ "$(AGENT_STACK_NAME)" != "vllm-sr" ]; then \
		STACK_CONTAINER="$(AGENT_STACK_NAME)-vllm-sr-container"; \
	fi; \
	STACK_DASHBOARD_PORT=$$((8700 + $(AGENT_PORT_OFFSET))); \
	START_TIME="$$(date +%s)"; \
	while true; do \
		STATUS_OUTPUT="$$(VLLM_SR_STACK_NAME="$(AGENT_STACK_NAME)" VLLM_SR_PORT_OFFSET="$(AGENT_PORT_OFFSET)" vllm-sr status all 2>&1 || true)"; \
		if echo "$$STATUS_OUTPUT" | grep -q "Container Status: Running" && \
		   echo "$$STATUS_OUTPUT" | grep -q "Router: Running" && \
		   echo "$$STATUS_OUTPUT" | grep -q "Envoy: Running" && \
		   echo "$$STATUS_OUTPUT" | grep -q "Dashboard: Running"; then \
			echo "$$STATUS_OUTPUT"; \
			break; \
		fi; \
		NOW="$$(date +%s)"; \
		if [ $$((NOW - START_TIME)) -ge "$(AGENT_SMOKE_TIMEOUT)" ]; then \
			echo "$$STATUS_OUTPUT"; \
			echo "Timed out waiting for local smoke checks"; \
			exit 1; \
		fi; \
		sleep 5; \
	done; \
	curl -fsS "http://localhost:$$STACK_DASHBOARD_PORT" >/dev/null; \
	$(CONTAINER_RUNTIME) ps --filter "name=$$STACK_CONTAINER" --format '{{.Names}}' | grep -q "^$$STACK_CONTAINER$$"; \
	if $(CONTAINER_RUNTIME) logs $$STACK_CONTAINER 2>&1 | grep -E "Image not found locally|Failed to pull image|Container exited unexpectedly" >/dev/null; then \
		echo "Detected startup failure in container logs"; \
		exit 1; \
	fi; \
	echo "Local smoke checks passed"

agent-e2e-affected: ## Run local E2E profiles affected by the changed files
	@$(LOG_TARGET)
	@python3 tools/agent/scripts/agent_gate.py run-e2e --base-ref "$(AGENT_BASE_REF)" --changed-files "$(CHANGED_FILES)"

agent-feature-gate: ## Run lint, targeted tests, local smoke, and affected E2E profiles
	@$(LOG_TARGET)
	@set -e; \
	$(MAKE) agent-ci-gate CHANGED_FILES="$(CHANGED_FILES)" AGENT_BASE_REF="$(AGENT_BASE_REF)"; \
	python3 tools/agent/scripts/agent_gate.py run-tests --mode feature-only --base-ref "$(AGENT_BASE_REF)" --changed-files "$(CHANGED_FILES)"; \
	if [ "$$(python3 tools/agent/scripts/agent_gate.py needs-smoke --base-ref "$(AGENT_BASE_REF)" --changed-files "$(CHANGED_FILES)")" = "true" ]; then \
		trap '$(MAKE) agent-stop-local ENV=$(ENV) AGENT_STACK_NAME="$(AGENT_STACK_NAME)" AGENT_PORT_OFFSET="$(AGENT_PORT_OFFSET)" >/dev/null 2>&1 || true' EXIT; \
		$(MAKE) agent-dev ENV=$(ENV); \
		$(MAKE) agent-serve-local ENV=$(ENV) AGENT_STACK_NAME="$(AGENT_STACK_NAME)" AGENT_PORT_OFFSET="$(AGENT_PORT_OFFSET)"; \
		$(MAKE) agent-smoke-local AGENT_STACK_NAME="$(AGENT_STACK_NAME)" AGENT_PORT_OFFSET="$(AGENT_PORT_OFFSET)"; \
	fi; \
	$(MAKE) agent-e2e-affected CHANGED_FILES="$(CHANGED_FILES)" AGENT_BASE_REF="$(AGENT_BASE_REF)"; \
	python3 tools/agent/scripts/agent_gate.py report --env "$(ENV)" --base-ref "$(AGENT_BASE_REF)" --changed-files "$(CHANGED_FILES)"

.PHONY: agent-help agent-bootstrap agent-dev agent-serve-local agent-stop-local \
	agent-validate agent-lint agent-fast-gate agent-report agent-ci-gate agent-smoke-local agent-e2e-affected \
	agent-feature-gate
