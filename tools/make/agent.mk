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
AGENT_GOLANGCI_LINT_VERSION ?= 2.5.0
AGENT_REPORT_WRITE ?=
AGENT_REPORT_WRITE_PATH ?=
AGENT_REPORT_CONTEXT_DETAIL ?= compact

agent-help: ## Show help for agent-specific targets
	@echo "Agent commands:"
	@echo "  make agent-bootstrap"
	@echo "  make agent-validate"
	@echo "  make agent-scorecard"
	@echo "  make agent-ci-lint CHANGED_FILES=\"...\""
	@echo "  make agent-dev ENV=cpu|amd"
	@echo "  make agent-serve-local ENV=cpu|amd"
	@echo "    optional: AGENT_STACK_NAME=<name> AGENT_PORT_OFFSET=<n>"
	@echo "  make agent-report ENV=cpu|amd CHANGED_FILES=\"...\""
	@echo "    optional: AGENT_REPORT_CONTEXT_DETAIL=compact|full AGENT_REPORT_WRITE=1 or AGENT_REPORT_WRITE_PATH=.agent-harness/reports/custom.json"
	@echo "  make agent-lint CHANGED_FILES=\"...\""
	@echo "  make agent-fast-gate CHANGED_FILES=\"...\""
	@echo "  make agent-ci-gate CHANGED_FILES=\"...\""
	@echo "  make agent-pr-gate"
	@echo "  make test-and-build-local"
	@echo "  make agent-e2e-affected CHANGED_FILES=\"...\""
	@echo "  make agent-feature-gate ENV=cpu|amd CHANGED_FILES=\"...\""

agent-bootstrap: ## Install agent validation tooling
	@$(LOG_TARGET)
	@echo "Installing agent Python tooling..."
	@python3 -m pip install -r tools/agent/requirements.txt
	@if command -v npm >/dev/null 2>&1; then \
		npm install -g markdownlint-cli@0.43.0 >/dev/null 2>&1 || true; \
	fi
	@if command -v go >/dev/null 2>&1; then \
		GOLANGCI_BIN="$$(go env GOPATH)/bin/golangci-lint"; \
		if [ ! -x "$$GOLANGCI_BIN" ] || ! "$$GOLANGCI_BIN" version 2>/dev/null | grep -q " $(AGENT_GOLANGCI_LINT_VERSION) "; then \
			echo "Installing golangci-lint v$(AGENT_GOLANGCI_LINT_VERSION)..."; \
			go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v$(AGENT_GOLANGCI_LINT_VERSION); \
		fi; \
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
		$(MAKE) vllm-sr-dev VLLM_SR_PLATFORM=amd VLLM_SR_TOPOLOGY=$(VLLM_SR_TOPOLOGY); \
	else \
		$(MAKE) vllm-sr-dev VLLM_SR_TOPOLOGY=$(VLLM_SR_TOPOLOGY); \
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
		VLLM_SR_STACK_NAME="$(AGENT_STACK_NAME)" VLLM_SR_PORT_OFFSET="$(AGENT_PORT_OFFSET)" VLLM_SR_STATE_ROOT_DIR="$$(pwd)" VLLM_SR_TOPOLOGY="$(VLLM_SR_TOPOLOGY)" \
		vllm-sr serve --image-pull-policy never --platform amd $$CONFIG_ARGS $(AGENT_SERVE_ARGS); \
	else \
		echo "Starting local CPU workflow..."; \
		VLLM_SR_STACK_NAME="$(AGENT_STACK_NAME)" VLLM_SR_PORT_OFFSET="$(AGENT_PORT_OFFSET)" VLLM_SR_STATE_ROOT_DIR="$$(pwd)" VLLM_SR_TOPOLOGY="$(VLLM_SR_TOPOLOGY)" \
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
	pre-commit run --files $$FILE_ARGS && \
	echo "Running Python lint..." && \
	python3 tools/agent/scripts/agent_gate.py run-python-lint --changed-files "$$CSV_FILES" && \
	echo "Running Go structural lint..." && \
	python3 tools/agent/scripts/agent_gate.py run-go-lint --base-ref "$(AGENT_BASE_REF)" --changed-files "$$CSV_FILES" && \
	echo "Running reference config contract lint..." && \
	python3 tools/agent/scripts/agent_gate.py run-config-contract-lint --changed-files "$$CSV_FILES" && \
	echo "Running Rust lint..." && \
	python3 tools/agent/scripts/agent_gate.py run-rust-lint --changed-files "$$CSV_FILES" && \
	echo "Running structure checks..." && \
	python3 tools/agent/scripts/structure_check.py --base-ref "$(AGENT_BASE_REF)" $$FILE_ARGS

agent-fast-gate: ## Run the fast gate: manifest validation, lint, and lightweight tests
	@$(LOG_TARGET)
	@$(MAKE) agent-validate
	@$(MAKE) agent-lint CHANGED_FILES="$(CHANGED_FILES)" AGENT_BASE_REF="$(AGENT_BASE_REF)"
	@python3 tools/agent/scripts/agent_gate.py run-tests --mode fast --base-ref "$(AGENT_BASE_REF)" --changed-files "$(CHANGED_FILES)"

agent-ci-lint: agent-bootstrap ## Reproduce the CI changed-file lint gate locally
	@$(LOG_TARGET)
	@BASE_REF="$(AGENT_BASE_REF)"; \
	if [ -z "$$BASE_REF" ] && git rev-parse --verify origin/main >/dev/null 2>&1; then \
		BASE_REF="origin/main"; \
	fi; \
	if [ -z "$$BASE_REF" ] && git rev-parse --verify HEAD^ >/dev/null 2>&1; then \
		BASE_REF="HEAD^"; \
	fi; \
	if [ -n "$$BASE_REF" ]; then \
		echo "Using AGENT_BASE_REF=$$BASE_REF"; \
	else \
		echo "Using AGENT_BASE_REF=<empty>"; \
	fi; \
	$(MAKE) codespell-tracked && \
	$(MAKE) agent-fast-gate CHANGED_FILES="$(CHANGED_FILES)" AGENT_BASE_REF="$$BASE_REF"

agent-report: ## Show primary skill, impacted surfaces, and validation commands
	@$(LOG_TARGET)
	@if [ -n "$(AGENT_REPORT_WRITE_PATH)" ]; then \
		python3 tools/agent/scripts/agent_gate.py report --env "$(ENV)" --base-ref "$(AGENT_BASE_REF)" --changed-files "$(CHANGED_FILES)" --context-detail "$(AGENT_REPORT_CONTEXT_DETAIL)" --write "$(AGENT_REPORT_WRITE_PATH)"; \
	elif [ -n "$(AGENT_REPORT_WRITE)" ]; then \
		python3 tools/agent/scripts/agent_gate.py report --env "$(ENV)" --base-ref "$(AGENT_BASE_REF)" --changed-files "$(CHANGED_FILES)" --context-detail "$(AGENT_REPORT_CONTEXT_DETAIL)" --write-default; \
	else \
		python3 tools/agent/scripts/agent_gate.py report --env "$(ENV)" --base-ref "$(AGENT_BASE_REF)" --changed-files "$(CHANGED_FILES)" --context-detail "$(AGENT_REPORT_CONTEXT_DETAIL)"; \
	fi

agent-ci-gate: ## Run the repo-standard fast CI gate
	@$(LOG_TARGET)
	@$(MAKE) agent-report ENV="$(ENV)" CHANGED_FILES="$(CHANGED_FILES)" AGENT_BASE_REF="$(AGENT_BASE_REF)"
	@python3 tools/agent/scripts/agent_gate.py resolve --base-ref "$(AGENT_BASE_REF)" --changed-files "$(CHANGED_FILES)" --format summary
	@$(MAKE) agent-fast-gate CHANGED_FILES="$(CHANGED_FILES)" AGENT_BASE_REF="$(AGENT_BASE_REF)"

agent-smoke-local: ## Validate local container, router, envoy, and dashboard health
	@$(LOG_TARGET)
	@STACK_CONTAINER="$(VLLM_SR_CONTAINER)"; \
	STACK_ROUTER_CONTAINER="vllm-sr-router-container"; \
	STACK_ENVOY_CONTAINER="vllm-sr-envoy-container"; \
	STACK_DASHBOARD_CONTAINER="vllm-sr-dashboard-container"; \
	if [ -n "$(AGENT_STACK_NAME)" ] && [ "$(AGENT_STACK_NAME)" != "vllm-sr" ]; then \
		STACK_CONTAINER="$(AGENT_STACK_NAME)-vllm-sr-container"; \
		STACK_ROUTER_CONTAINER="$(AGENT_STACK_NAME)-vllm-sr-router-container"; \
		STACK_ENVOY_CONTAINER="$(AGENT_STACK_NAME)-vllm-sr-envoy-container"; \
		STACK_DASHBOARD_CONTAINER="$(AGENT_STACK_NAME)-vllm-sr-dashboard-container"; \
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
	if ! ( \
		$(CONTAINER_RUNTIME) ps --filter "name=$$STACK_CONTAINER" --format '{{.Names}}' | grep -q "^$$STACK_CONTAINER$$" || \
		( \
			$(CONTAINER_RUNTIME) ps --filter "name=$$STACK_ROUTER_CONTAINER" --format '{{.Names}}' | grep -q "^$$STACK_ROUTER_CONTAINER$$" && \
			$(CONTAINER_RUNTIME) ps --filter "name=$$STACK_ENVOY_CONTAINER" --format '{{.Names}}' | grep -q "^$$STACK_ENVOY_CONTAINER$$" && \
			$(CONTAINER_RUNTIME) ps --filter "name=$$STACK_DASHBOARD_CONTAINER" --format '{{.Names}}' | grep -q "^$$STACK_DASHBOARD_CONTAINER$$" \
		) \
	); then \
		echo "Managed runtime containers were not found"; \
		exit 1; \
	fi; \
	for RUNTIME_CONTAINER in \
		"$$STACK_CONTAINER" \
		"$$STACK_ROUTER_CONTAINER" \
		"$$STACK_ENVOY_CONTAINER" \
		"$$STACK_DASHBOARD_CONTAINER"; do \
		if ! $(CONTAINER_RUNTIME) ps -a --filter "name=$$RUNTIME_CONTAINER" --format '{{.Names}}' | grep -q "^$$RUNTIME_CONTAINER$$"; then \
			continue; \
		fi; \
		if $(CONTAINER_RUNTIME) logs "$$RUNTIME_CONTAINER" 2>&1 | grep -E "Image not found locally|Failed to pull image|Container exited unexpectedly" >/dev/null; then \
			echo "Detected startup failure in container logs: $$RUNTIME_CONTAINER"; \
			exit 1; \
		fi; \
	done; \
	echo "Local smoke checks passed"

agent-e2e-affected: ## Run local E2E profiles affected by the changed files
	@$(LOG_TARGET)
	@python3 tools/agent/scripts/agent_gate.py run-e2e --base-ref "$(AGENT_BASE_REF)" --changed-files "$(CHANGED_FILES)"

test-and-build-local: ## Reproduce the CI Test And Build job locally
	@$(LOG_TARGET)
	@set -e; \
	trap '$(MAKE) clean-redis >/dev/null 2>&1 || true; $(MAKE) clean-valkey >/dev/null 2>&1 || true; $(MAKE) stop-milvus >/dev/null 2>&1 || true' EXIT; \
	$(MAKE) check-go-mod-tidy; \
	$(MAKE) rust-ci; \
	$(MAKE) helm-ci-validate HELM_NAMESPACE=test-namespace; \
	python3 -m pip install -U "huggingface_hub[cli]" hf_transfer; \
	$(MAKE) start-milvus; \
	$(MAKE) start-redis; \
	$(MAKE) start-valkey; \
	CI=true CI_MINIMAL_MODELS=true CGO_ENABLED=1 LD_LIBRARY_PATH="$(CURDIR)/candle-binding/target/release" MILVUS_URI=localhost:19530 SKIP_MILVUS_TESTS=false SKIP_REDIS_TESTS=false SKIP_VALKEY_TESTS=false VALKEY_HOST=localhost VALKEY_PORT=6380 HF_TOKEN="$(HF_TOKEN)" HUGGINGFACE_HUB_TOKEN="$(HUGGINGFACE_HUB_TOKEN)" $(MAKE) test

agent-pr-gate: ## Reproduce the baseline PR requirements locally
	@$(LOG_TARGET)
	@$(MAKE) precommit-local AGENT_BASE_REF="$(AGENT_BASE_REF)"
	@$(MAKE) test-and-build-local

agent-feature-gate: ## Run lint, targeted tests, local smoke, and a final report
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
	python3 tools/agent/scripts/agent_gate.py report --env "$(ENV)" --base-ref "$(AGENT_BASE_REF)" --changed-files "$(CHANGED_FILES)"

.PHONY: agent-help agent-bootstrap agent-ci-lint agent-dev agent-serve-local agent-stop-local \
	agent-validate agent-lint agent-fast-gate agent-report agent-ci-gate agent-smoke-local agent-e2e-affected \
	test-and-build-local agent-pr-gate agent-feature-gate
