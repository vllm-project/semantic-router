# ======== golang.mk ========
# = Everything For Golang   =
# ======== golang.mk ========

##@ Golang

go-lint: ## Run golangci-lint for src/semantic-router
	@$(LOG_TARGET)
	@echo "Running golangci-lint for src/semantic-router..."
	@cd src/semantic-router/ && \
		export GOROOT=$$(dirname $$(dirname $$(readlink -f $$(which go)))) && \
		export GOPATH=$$(go env GOPATH 2>/dev/null || echo "$$HOME/go") && \
		export PATH="$$GOPATH/bin:$$PATH" && \
		golangci-lint run ./... --config ../../tools/linter/go/.golangci.yml
	@echo "src/semantic-router go module lint passed"

go-lint-fix: ## Auto-fix lint issues in src/semantic-router (may need manual fix)
	@$(LOG_TARGET)
	@echo "Running golangci-lint fix for src/semantic-router..."
	@cd src/semantic-router/ && \
		export GOROOT=$$(dirname $$(dirname $$(readlink -f $$(which go)))) && \
		export GOPATH=$$(go env GOPATH 2>/dev/null || echo "$$HOME/go") && \
		export PATH="$$GOPATH/bin:$$PATH" && \
		golangci-lint run ./... --fix --config ../../tools/linter/go/.golangci.yml
	@echo "src/semantic-router go module lint fix applied"

vet: $(if $(CI),rust-ci,rust) ## Run go vet for all Go modules (build Rust library first)
	@$(LOG_TARGET)
	@cd candle-binding && go vet ./...
	@cd src/semantic-router && go vet ./...

check-go-mod-tidy: ## Check go mod tidy for all Go modules
	@$(LOG_TARGET)
	@echo "Checking go mod tidy for all Go modules..."
	@echo "Checking candle-binding..."
	@cd candle-binding && go mod tidy && \
		(git diff --exit-code go.mod 2>/dev/null || (echo "ERROR: go.mod file is not tidy in candle-binding. Please run 'go mod tidy' in candle-binding directory and commit the changes." && git diff go.mod && exit 1)) && \
		(test ! -f go.sum || git diff --exit-code go.sum 2>/dev/null || (echo "ERROR: go.sum file is not tidy in candle-binding. Please run 'go mod tidy' in candle-binding directory and commit the changes." && git diff go.sum && exit 1))
	@echo "candle-binding go mod tidy check passed"
	@echo "Checking src/semantic-router..."
	@cd src/semantic-router && go mod tidy && \
		if ! git diff --exit-code go.mod go.sum; then \
			echo "ERROR: go.mod or go.sum files are not tidy in src/semantic-router. Please run 'go mod tidy' in src/semantic-router directory and commit the changes."; \
			git diff go.mod go.sum; \
			exit 1; \
		fi
	@echo "src/semantic-router go mod tidy check passed"
	@echo "All go mod tidy checks passed"
