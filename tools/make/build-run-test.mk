# ============== build-run-test.mk ==============
# =   Project build, run and test related       =
# =============== build-run-test.mk =============

##@ Build/Test

# Build the Rust library and Golang binding
build: rust build-router ## Build the Rust library and Golang binding

# Build router
build-router: rust ## Build the router binary
	@$(LOG_TARGET)
	@mkdir -p bin
	@cd src/semantic-router && go build --tags=milvus -o ../../bin/router cmd/main.go

# Run the router
run-router: build-router download-models ## Run the router with the specified config
	@echo "Running router with config: ${CONFIG_FILE}"
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release && \
		./bin/router -config=${CONFIG_FILE} --enable-system-prompt-api=true

# Run the router with e2e config for testing
run-router-e2e: build-router download-models ## Run the router with e2e config for testing
	@echo "Running router with e2e config: config/config.e2e.yaml"
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release && \
		./bin/router -config=config/config.e2e.yaml

# Unit test semantic-router
# By default, Milvus tests are skipped. To enable them, set SKIP_MILVUS_TESTS=false
# Example: make test-semantic-router SKIP_MILVUS_TESTS=false
test-semantic-router: build-router ## Run unit tests for semantic-router (set SKIP_MILVUS_TESTS=false to enable Milvus tests)
	@$(LOG_TARGET)
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release && \
	export SKIP_MILVUS_TESTS=$${SKIP_MILVUS_TESTS:-true} && \
		cd src/semantic-router && CGO_ENABLED=1 go test -v ./...

# Test the Rust library and the Go binding
test: vet go-lint check-go-mod-tidy download-models test-binding test-semantic-router ## Run all tests (Go, Rust, binding)

# Clean built artifacts
clean: ## Clean built artifacts
	@echo "Cleaning build artifacts..."
	cd candle-binding && cargo clean
	rm -f bin/router

# Test the Envoy extproc
test-auto-prompt-reasoning: ## Test Envoy extproc with a math prompt (curl)
	@echo "Testing Envoy extproc with curl (Math)..."
	curl -X POST http://localhost:8801/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "auto", "messages": [{"role": "system", "content": "You are a professional math teacher. Explain math concepts clearly and show step-by-step solutions to problems."}, {"role": "user", "content": "What is the derivative of f(x) = x^3 + 2x^2 - 5x + 7?"}]}'
