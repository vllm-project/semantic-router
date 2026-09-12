# ========================== agent.mk ==========================
# = Changed-file checks and explicit integration verification  =
# ============================================================

##@ Development checks

ENV ?= cpu
DOMAIN ?=
PROFILE ?=
CHANGED_FILES ?=
CHANGED_FILES_PATH ?=
BASE_REF ?=
HARNESS_BOOTSTRAP_DONE ?=

AGENT_GOLANGCI_LINT_VERSION ?= $(shell cat $(CURDIR)/tools/linter/go/golangci-lint.version)
AGENT_MARKDOWNLINT_VERSION ?= 0.43.0
AGENT_NODE_VERSION ?= 22.17.0

# Share the tooling environment across linked worktrees. The worktree symlink
# keeps existing pre-commit entries portable without installing dependencies
# more than once.
AGENT_GIT_COMMON_DIR ?= $(shell git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)
AGENT_PRIMARY_WORKTREE ?= $(if $(filter %/.git,$(AGENT_GIT_COMMON_DIR)),$(patsubst %/.git,%,$(AGENT_GIT_COMMON_DIR)),$(CURDIR))
AGENT_WORKTREE_VENV ?= $(CURDIR)/.venv-agent
AGENT_VENV ?= $(AGENT_PRIMARY_WORKTREE)/.venv-agent
AGENT_PYTHON ?= $(AGENT_VENV)/bin/python
AGENT_PRE_COMMIT ?= $(AGENT_VENV)/bin/pre-commit
AGENT_REQUIREMENTS_STAMP ?= $(AGENT_VENV)/.agent-requirements.txt
AGENT_NODEENV ?= $(AGENT_VENV)/nodeenv
AGENT_NODE_TOOLS ?= $(AGENT_VENV)/node-tools
AGENT_MARKDOWNLINT ?= $(AGENT_NODE_TOOLS)/node_modules/.bin/markdownlint

ifeq ($(HARNESS_BOOTSTRAP_DONE),1)
HARNESS_BOOTSTRAP_DEPS :=
HARNESS_VENV_DEPS :=
else
HARNESS_BOOTSTRAP_DEPS := harness-bootstrap
HARNESS_VENV_DEPS := harness-venv-install
endif

impact: $(HARNESS_VENV_DEPS) ## Show changed files, owners, checks, candidate CI, and available tools
	@$(LOG_TARGET)
	@"$(AGENT_PYTHON)" tools/agent/scripts/harness.py impact --env "$(ENV)" --base-ref "$(BASE_REF)" --changed-files "$(CHANGED_FILES)" --changed-files-path "$(CHANGED_FILES_PATH)"

check: $(HARNESS_BOOTSTRAP_DEPS) ## Lint changed files and run their owning domains' unit/contract checks
	@$(LOG_TARGET)
	@"$(AGENT_PYTHON)" tools/agent/scripts/harness.py check --base-ref "$(BASE_REF)" --changed-files "$(CHANGED_FILES)" --changed-files-path "$(CHANGED_FILES_PATH)"

verify: $(HARNESS_VENV_DEPS) ## Run explicitly selected integration checks (DOMAIN=... and/or PROFILE=...)
	@$(LOG_TARGET)
	@"$(AGENT_PYTHON)" tools/agent/scripts/harness.py verify --domains "$(DOMAIN)" --profiles "$(PROFILE)"

ci-full: ## Reproduce the complete baseline PR checks locally
	@$(LOG_TARGET)
	@$(MAKE) precommit-local BASE_REF="$(BASE_REF)"
	@$(MAKE) test-and-build-local

harness-check: $(HARNESS_BOOTSTRAP_DEPS) ## Validate the domain registry, workflows, and harness tests
	@$(LOG_TARGET)
	@"$(AGENT_PYTHON)" tools/agent/scripts/harness.py validate
	@"$(AGENT_PYTHON)" tools/ci/validate_workflows.py
	@"$(AGENT_PYTHON)" -m unittest discover -s tools/ci/tests -p "test_*.py"
	@"$(AGENT_PYTHON)" -m unittest discover -s tools/agent/scripts/tests -p "test_*.py"
	@if command -v actionlint >/dev/null 2>&1; then \
		actionlint -shellcheck=; \
	elif command -v go >/dev/null 2>&1; then \
		go run github.com/rhysd/actionlint/cmd/actionlint@v1.7.12 -shellcheck=; \
	else \
		echo "actionlint or Go is required for workflow validation"; \
		exit 1; \
	fi

harness-venv-install: ## Install the repository check dependencies
	@if [ ! -x "$(AGENT_PYTHON)" ]; then \
		echo "Creating $(AGENT_VENV)..."; \
		python3 -m venv "$(AGENT_VENV)"; \
	fi
	@if [ ! -f "$(AGENT_REQUIREMENTS_STAMP)" ] || \
		! cmp -s tools/agent/requirements.txt "$(AGENT_REQUIREMENTS_STAMP)"; then \
		"$(AGENT_PYTHON)" -m pip install -r tools/agent/requirements.txt && \
		cp tools/agent/requirements.txt "$(AGENT_REQUIREMENTS_STAMP)"; \
	fi
	@if [ "$(abspath $(AGENT_WORKTREE_VENV))" != "$(abspath $(AGENT_VENV))" ]; then \
		if [ -e "$(AGENT_WORKTREE_VENV)" ] && [ ! -L "$(AGENT_WORKTREE_VENV)" ]; then \
			echo "Error: $(AGENT_WORKTREE_VENV) is a local directory; move or remove it so this worktree can use $(AGENT_VENV)." >&2; \
			exit 1; \
		fi; \
		ln -sfn "$(AGENT_VENV)" "$(AGENT_WORKTREE_VENV)"; \
	fi

harness-bootstrap: harness-venv-install ## Prepare the shared Python environment
	@$(LOG_TARGET)
	@echo "Harness Python tooling ready"

harness-node-bootstrap: $(HARNESS_VENV_DEPS) ## Provide cached Node when the host has none
	@if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then \
		if [ ! -x "$(AGENT_NODEENV)/bin/node" ] || [ ! -x "$(AGENT_NODEENV)/bin/npm" ]; then \
			echo "Installing repo-local Node.js v$(AGENT_NODE_VERSION)..."; \
			"$(AGENT_PYTHON)" -m nodeenv --node="$(AGENT_NODE_VERSION)" --prebuilt "$(AGENT_NODEENV)"; \
		elif [ "$$($(AGENT_NODEENV)/bin/node --version 2>/dev/null)" != "v$(AGENT_NODE_VERSION)" ]; then \
			echo "Updating repo-local Node.js to v$(AGENT_NODE_VERSION)..."; \
			"$(AGENT_PYTHON)" -m nodeenv --force --node="$(AGENT_NODE_VERSION)" --prebuilt "$(AGENT_NODEENV)"; \
		fi; \
	fi

harness-markdown-bootstrap: harness-node-bootstrap ## Install repo-local markdownlint when needed
	@if [ ! -x "$(AGENT_MARKDOWNLINT)" ] || \
		[ "$$(PATH="$(AGENT_NODEENV)/bin:$$PATH" "$(AGENT_MARKDOWNLINT)" --version 2>/dev/null)" != "$(AGENT_MARKDOWNLINT_VERSION)" ]; then \
		NODE_PATH="$$PATH"; \
		if ! command -v npm >/dev/null 2>&1; then NODE_PATH="$(AGENT_NODEENV)/bin:$$NODE_PATH"; fi; \
		PATH="$$NODE_PATH" npm install --prefix "$(AGENT_NODE_TOOLS)" --no-audit --no-fund --loglevel=error markdownlint-cli@$(AGENT_MARKDOWNLINT_VERSION); \
	fi

harness-go-bootstrap: ## Install Go lint tooling only when Go changed
	@if command -v go >/dev/null 2>&1; then \
		GOLANGCI_BIN="$$(go env GOPATH)/bin/golangci-lint"; \
		if [ ! -x "$$GOLANGCI_BIN" ] || ! "$$GOLANGCI_BIN" version 2>/dev/null | grep -q " $(AGENT_GOLANGCI_LINT_VERSION) "; then \
			go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v$(AGENT_GOLANGCI_LINT_VERSION); \
		fi; \
	fi

harness-rust-bootstrap: ## Install Rust lint tooling only when Rust changed
	@if command -v rustup >/dev/null 2>&1; then rustup component add clippy >/dev/null 2>&1 || true; fi

test-and-build-local: ## Reproduce the CI Test And Build job locally
	@$(LOG_TARGET)
	@set -e; \
	trap '$(MAKE) clean-redis >/dev/null 2>&1 || true; $(MAKE) clean-valkey >/dev/null 2>&1 || true; $(MAKE) stop-milvus >/dev/null 2>&1 || true; $(MAKE) stop-qdrant >/dev/null 2>&1 || true' EXIT; \
	$(MAKE) check-go-mod-tidy; \
	$(MAKE) rust-ci; \
	$(MAKE) helm-ci-validate HELM_NAMESPACE=test-namespace; \
	python3 -m pip install -U "huggingface_hub[cli]" hf_transfer; \
	$(MAKE) start-milvus; \
	$(MAKE) start-qdrant; \
	$(MAKE) start-redis; \
	$(MAKE) start-valkey; \
	CI=true CI_MINIMAL_MODELS=true CGO_ENABLED=1 $(NATIVE_ENV) MILVUS_URI=localhost:19530 SKIP_MILVUS_TESTS=false SKIP_QDRANT_TESTS=false SKIP_REDIS_TESTS=false SKIP_VALKEY_TESTS=false VALKEY_HOST=localhost VALKEY_PORT=6380 HF_TOKEN="$(HF_TOKEN)" HUGGINGFACE_HUB_TOKEN="$(HUGGINGFACE_HUB_TOKEN)" $(MAKE) test

.PHONY: impact check verify ci-full harness-check harness-venv-install harness-bootstrap \
	harness-node-bootstrap harness-markdown-bootstrap harness-go-bootstrap harness-rust-bootstrap \
	test-and-build-local
