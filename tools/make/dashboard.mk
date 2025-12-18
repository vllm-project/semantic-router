# ========================== dashboard.mk ==========================
# = Everything For Dashboard, include Frontend and Backend          =
# ========================== dashboard.mk ==========================

DASHBOARD_DIR := dashboard
DASHBOARD_FRONTEND_DIR := $(DASHBOARD_DIR)/frontend
DASHBOARD_BACKEND_DIR := $(DASHBOARD_DIR)/backend

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

## Clean

dashboard-clean: ## Clean dashboard build artifacts (frontend dist + backend bin)
	@$(LOG_TARGET)
	rm -rf $(DASHBOARD_FRONTEND_DIR)/dist
	rm -rf $(DASHBOARD_FRONTEND_DIR)/node_modules
	rm -rf $(DASHBOARD_BACKEND_DIR)/bin
	@echo "✅ dashboard cleaned"

.PHONY: dashboard-install dashboard-dev-frontend dashboard-dev-backend \
	dashboard-build dashboard-build-frontend dashboard-build-backend \
	dashboard-lint dashboard-lint-fix dashboard-type-check dashboard-go-mod-tidy \
	dashboard-check dashboard-clean

