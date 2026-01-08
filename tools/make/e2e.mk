# ======== e2e.mk ========
# = E2E Testing Framework =
# ======== e2e.mk ========

##@ E2E Testing

# E2E test configuration
E2E_PROFILE ?= ai-gateway
E2E_CLUSTER_NAME ?= semantic-router-e2e
E2E_IMAGE_TAG ?= e2e-test
E2E_KEEP_CLUSTER ?= false
E2E_USE_EXISTING_CLUSTER ?= false
E2E_VERBOSE ?= true
E2E_PARALLEL ?= false
E2E_TESTS ?=
E2E_SETUP_ONLY ?= false
E2E_SKIP_SETUP ?= false

# Build the E2E test binary
build-e2e: ## Build the E2E test binary
	@$(LOG_TARGET)
	@echo "Building E2E test binary..."
	@cd e2e && go build -o ../bin/e2e ./cmd/e2e

# Run E2E tests
e2e-test: ## Run E2E tests (PROFILE=ai-gateway by default)
e2e-test: build-e2e
	@$(LOG_TARGET)
	@echo "Running E2E tests with profile: $(E2E_PROFILE)"
	@./bin/e2e \
		-profile=$(E2E_PROFILE) \
		-cluster=$(E2E_CLUSTER_NAME) \
		-image-tag=$(E2E_IMAGE_TAG) \
		-keep-cluster=$(E2E_KEEP_CLUSTER) \
		-use-existing-cluster=$(E2E_USE_EXISTING_CLUSTER) \
		-verbose=$(E2E_VERBOSE) \
		-parallel=$(E2E_PARALLEL) \
		-setup-only=$(E2E_SETUP_ONLY) \
		-skip-setup=$(E2E_SKIP_SETUP) \
		$(if $(E2E_TESTS),-tests=$(E2E_TESTS),)

# Run E2E tests with AI Gateway profile
e2e-test-ai-gateway: ## Run E2E tests with AI Gateway profile
e2e-test-ai-gateway: E2E_PROFILE=ai-gateway
e2e-test-ai-gateway: e2e-test

# Run E2E tests with Dynamo profile (requires GPU)
e2e-test-dynamo: ## Run E2E tests with Dynamo profile (requires 3+ GPUs)
e2e-test-dynamo: E2E_PROFILE=dynamo
e2e-test-dynamo: e2e-test

# Run E2E tests and keep cluster for debugging
e2e-test-debug: ## Run E2E tests and keep cluster for debugging
e2e-test-debug: E2E_KEEP_CLUSTER=true
e2e-test-debug: E2E_VERBOSE=true
e2e-test-debug: e2e-test

# Setup profile only without running tests
e2e-setup: ## Setup profile only without running tests
e2e-setup: E2E_SETUP_ONLY=true
e2e-setup: e2e-test

# Run tests only without setup (assumes environment is already deployed)
e2e-test-only: ## Run tests only without setup (assumes environment is already deployed)
e2e-test-only: E2E_USE_EXISTING_CLUSTER=true
e2e-test-only: E2E_SKIP_SETUP=true
e2e-test-only: e2e-test

# Run specific E2E test cases
e2e-test-specific: ## Run specific E2E test cases (E2E_TESTS="test1,test2")
e2e-test-specific:
	@if [ -z "$(E2E_TESTS)" ]; then \
		echo "Error: E2E_TESTS is not set"; \
		echo "Usage: make e2e-test-specific E2E_TESTS=\"basic-health-check,chat-completions-request\""; \
		exit 1; \
	fi
	@$(MAKE) e2e-test E2E_TESTS=$(E2E_TESTS)

# Clean up E2E test cluster
e2e-cleanup: ## Clean up E2E test cluster
	@$(LOG_TARGET)
	@echo "Cleaning up E2E test cluster: $(E2E_CLUSTER_NAME)"
	@kind delete cluster --name $(E2E_CLUSTER_NAME) || true

# Download E2E test dependencies
e2e-deps: ## Download E2E test dependencies
	@$(LOG_TARGET)
	@echo "Downloading E2E test dependencies..."
	@cd e2e && go mod download

# Tidy E2E test dependencies
e2e-tidy: ## Tidy E2E test dependencies
	@$(LOG_TARGET)
	@echo "Tidying E2E test dependencies..."
	@cd e2e && go mod tidy

# Help for E2E testing
e2e-help: ## Show help for E2E testing
	@echo "E2E Testing Framework"
	@echo ""
	@echo "Available Profiles:"
	@echo "  ai-gateway       - Test Semantic Router with Envoy AI Gateway"
	@echo "  aibrix           - Test Semantic Router with vLLM AIBrix"
	@echo "  dynamo           - Test Semantic Router with Nvidia Dynamo (requires 3+ GPUs)"
	@echo "  istio            - Test Semantic Router with Istio service mesh"
	@echo "  llm-d            - Test Semantic Router with LLM-D"
	@echo "  production-stack - Test Semantic Router in production-like stack (HA/LB/Obs)"
	@echo "  response-api     - Test Response API endpoints (POST/GET/DELETE /v1/responses)"
	@echo ""
	@echo "Environment Variables:"
	@echo "  E2E_PROFILE              - Test profile to run (default: ai-gateway)"
	@echo "  E2E_CLUSTER_NAME         - Kind cluster name (default: semantic-router-e2e)"
	@echo "  E2E_IMAGE_TAG            - Docker image tag (default: e2e-test)"
	@echo "  E2E_KEEP_CLUSTER         - Keep cluster after tests (default: false)"
	@echo "  E2E_USE_EXISTING_CLUSTER - Use existing cluster (default: false)"
	@echo "  E2E_VERBOSE              - Enable verbose logging (default: true)"
	@echo "  E2E_PARALLEL             - Run tests in parallel (default: false)"
	@echo "  E2E_TESTS                - Comma-separated list of test cases to run"
	@echo "  E2E_SETUP_ONLY           - Only setup profile without running tests (default: false)"
	@echo "  E2E_SKIP_SETUP           - Skip setup and only run tests (default: false)"
	@echo ""
	@echo "Common Commands:"
	@echo "  make e2e-test                                    # Run all tests with default profile"
	@echo "  make e2e-test E2E_PROFILE=ai-gateway             # Run AI Gateway tests"
	@echo "  make e2e-test-dynamo                             # Run Dynamo tests (requires GPU)"
	@echo "  make e2e-test-debug                              # Run tests and keep cluster"
	@echo "  make e2e-test-specific E2E_TESTS=\"test1,test2\"   # Run specific tests"
	@echo "  make e2e-cleanup                                 # Clean up test cluster"
	@echo ""
	@echo "Advanced Workflows:"
	@echo "  make e2e-setup                                   # Setup environment only (no tests)"
	@echo "  make e2e-test-only                               # Run tests only (skip setup)"
	@echo "  make e2e-test-only E2E_TESTS=\"test1\"             # Run specific test (skip setup)"
	@echo ""
	@echo "Example Workflow (Setup once, run tests multiple times):"
	@echo "  1. make e2e-setup                                # Setup environment"
	@echo "  2. make e2e-test-only                            # Run all tests"
	@echo "  3. make e2e-test-only E2E_TESTS=\"test1\"          # Run specific test"
	@echo "  4. make e2e-cleanup                              # Clean up when done"
