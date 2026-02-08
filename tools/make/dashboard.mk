# ========================== dashboard.mk ==========================
# = Everything For Dashboard, include Frontend and Backend          =
# ========================== dashboard.mk ==========================

DASHBOARD_DIR := dashboard
DASHBOARD_FRONTEND_DIR := $(DASHBOARD_DIR)/frontend
DASHBOARD_BACKEND_DIR := $(DASHBOARD_DIR)/backend
DASHBOARD_AGENT_DIR := $(DASHBOARD_DIR)/agent-service

##@ Dashboard

## Install and Development

dashboard-install: ## Install dashboard dependencies (frontend npm + backend go mod)
	@$(LOG_TARGET)
	@echo "Installing frontend dependencies..."
	cd $(DASHBOARD_FRONTEND_DIR) && npm install
	@echo "Tidying backend dependencies..."
	cd $(DASHBOARD_BACKEND_DIR) && go mod tidy
	@echo "✅ dashboard dependencies installed"

dashboard-dev-frontend: dashboard-install ## Start dashboard frontend in dev mode
	@$(LOG_TARGET)
	cd $(DASHBOARD_FRONTEND_DIR) && npm run dev

dashboard-dev-backend: ## Start dashboard backend in dev mode
	@$(LOG_TARGET)
	cd $(DASHBOARD_BACKEND_DIR) && go run main.go


## Build

dashboard-build-frontend: dashboard-install ## Build dashboard frontend for production
	@$(LOG_TARGET)
	cd $(DASHBOARD_FRONTEND_DIR) && npm run build
	@echo "✅ dashboard/frontend build completed"

dashboard-build-backend: ## Build dashboard backend binary
	@$(LOG_TARGET)
	@echo "Building dashboard backend..."
	cd $(DASHBOARD_BACKEND_DIR) && go build -o bin/dashboard-server ./main.go
	@echo "✅ dashboard/backend build completed"

dashboard-build: dashboard-build-frontend dashboard-build-backend ## Build dashboard (frontend + backend)
	@$(LOG_TARGET)
	@echo "✅ dashboard build completed (frontend + backend)"

## Lint and Type Check

dashboard-lint: ## Lint dashboard frontend and backend
	@$(LOG_TARGET)
	@echo "Running ESLint for dashboard frontend..."
	cd $(DASHBOARD_FRONTEND_DIR) && npm install 2>/dev/null && npm run lint
	@echo "✅ dashboard/frontend lint passed"
	@echo "Running golangci-lint for dashboard backend..."
	@cd $(DASHBOARD_BACKEND_DIR) && \
		export GOROOT=$$(dirname $$(dirname $$(readlink -f $$(which go)))) && \
		export GOPATH=$${GOPATH:-$$HOME/go} && \
		golangci-lint run ./... --config ../../tools/linter/go/.golangci.yml
	@echo "✅ dashboard/backend lint passed"

dashboard-lint-fix: ## Auto-fix lint issues in dashboard (frontend + backend)
	@$(LOG_TARGET)
	@echo "Running ESLint fix for dashboard frontend..."
	cd $(DASHBOARD_FRONTEND_DIR) && npm install 2>/dev/null && npm run lint -- --fix || true
	@echo "✅ dashboard/frontend lint fix applied"
	@echo "Running golangci-lint fix for dashboard backend..."
	@cd $(DASHBOARD_BACKEND_DIR) && \
		export GOROOT=$$(dirname $$(dirname $$(readlink -f $$(which go)))) && \
		export GOPATH=$${GOPATH:-$$HOME/go} && \
		golangci-lint run ./... --fix --config ../../tools/linter/go/.golangci.yml
	@echo "✅ dashboard/backend lint fix applied"

dashboard-type-check: ## Run TypeScript type checking for dashboard frontend
	@$(LOG_TARGET)
	cd $(DASHBOARD_FRONTEND_DIR) && npm install 2>/dev/null && npm run type-check
	@echo "✅ dashboard/frontend type-check passed"

dashboard-go-mod-tidy: ## Check go mod tidy for dashboard backend
	@$(LOG_TARGET)
	@echo "Checking dashboard/backend..."
	@cd $(DASHBOARD_BACKEND_DIR) && go mod tidy && \
		if ! git diff --exit-code go.mod go.sum 2>/dev/null; then \
			echo "ERROR: go.mod or go.sum files are not tidy in dashboard/backend. Please run 'go mod tidy' in dashboard/backend directory and commit the changes."; \
			git diff go.mod go.sum; \
			exit 1; \
		fi
	@echo "✅ dashboard/backend go mod tidy check passed"

dashboard-check: dashboard-lint dashboard-type-check dashboard-go-mod-tidy ## Run all dashboard checks (lint, type-check, go mod tidy)
	@$(LOG_TARGET)
	@echo "✅ All dashboard checks passed"

## Agent Service (E2B Computer-Use)

dashboard-agent-install: ## Install agent service Python dependencies
	@$(LOG_TARGET)
	@echo "Installing agent service dependencies..."
	@if [ ! -d "$(DASHBOARD_AGENT_DIR)/venv" ]; then \
		python3 -m venv $(DASHBOARD_AGENT_DIR)/venv; \
	fi
	@$(DASHBOARD_AGENT_DIR)/venv/bin/pip install --upgrade pip
	@$(DASHBOARD_AGENT_DIR)/venv/bin/pip install -e $(DASHBOARD_AGENT_DIR)[dev]
	@echo "✅ agent service dependencies installed"

dashboard-agent-dev: ## Start agent service in dev mode (requires E2B_API_KEY)
	@$(LOG_TARGET)
	@if [ ! -f "$(DASHBOARD_AGENT_DIR)/venv/bin/python" ]; then \
		echo "❌ Agent service virtualenv not found. Run 'make dashboard-agent-install' first."; \
		exit 1; \
	fi
	@echo "Starting agent service on port 8000..."
	@echo "Ensure E2B_API_KEY and HF_TOKEN environment variables are set."
	cd $(DASHBOARD_AGENT_DIR) && ./venv/bin/python -m cua_agent.main

dashboard-dev-all: ## Start all dashboard services (frontend + backend + agent)
	@$(LOG_TARGET)
	@echo "Starting all dashboard services..."
	@echo "Run these commands in separate terminals:"
	@echo "  Terminal 1: make dashboard-dev-frontend"
	@echo "  Terminal 2: make dashboard-dev-backend"
	@echo "  Terminal 3: make dashboard-agent-dev"
	@echo ""
	@echo "Or use a process manager like 'concurrently' or 'tmux'"

## Clean

dashboard-clean: ## Clean dashboard build artifacts (frontend dist + backend bin + agent venv)
	@$(LOG_TARGET)
	rm -rf $(DASHBOARD_FRONTEND_DIR)/dist
	rm -rf $(DASHBOARD_FRONTEND_DIR)/node_modules
	rm -rf $(DASHBOARD_BACKEND_DIR)/bin
	rm -rf $(DASHBOARD_AGENT_DIR)/venv
	rm -rf $(DASHBOARD_AGENT_DIR)/*.egg-info
	@echo "✅ dashboard cleaned"

.PHONY: dashboard-install dashboard-dev-frontend dashboard-dev-backend \
	dashboard-build dashboard-build-frontend dashboard-build-backend \
	dashboard-lint dashboard-lint-fix dashboard-type-check dashboard-go-mod-tidy \
	dashboard-check dashboard-clean \
	dashboard-agent-install dashboard-agent-dev dashboard-dev-all

