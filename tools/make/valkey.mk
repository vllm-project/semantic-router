start-valkey: ## Start Valkey bundle container with search module
stop-valkey: ## Stop and remove Valkey container
restart-valkey: ## Restart Valkey container
valkey-status: ## Show status of Valkey container
clean-valkey: ## Clean up Valkey data
test-valkey-vectorstore: ## Test vector store with Valkey backend
test-valkey-vectorstore-no-container: ## Test vector store against existing Valkey (VALKEY_PORT=6379)
test-valkey-memory-unit: ## Run ValkeyStore unit tests (no live Valkey required)
test-valkey-memory: ## Test ValkeyStore integration against containerised Valkey
test-valkey-memory-no-container: ## Test ValkeyStore integration against existing Valkey (VALKEY_PORT=6379)
test-valkey-cache: ## Test semantic cache with Valkey backend
test-semantic-router-valkey: ## Test semantic-router with Valkey cache backend
test-valkey-all: ## Test all Valkey backends
run-valkey-cache-example: ## Run the Valkey cache example
run-valkey-cache-example-no-container: ## Run the Valkey cache example using existing Valkey server
run-valkey-vectorstore-example: ## Run the Valkey vector store example
run-valkey-vectorstore-example-no-container: ## Run the Valkey vector store example using existing Valkey server
run-valkey-example: ## [Deprecated] Use run-valkey-cache-example
test-valkey-vectorstore-external: ## Alias for test-valkey-vectorstore-no-container
run-valkey-example-external: ## Alias for run-valkey-vectorstore-example-no-container
verify-valkey: ## Verify Valkey installation and vector search capability
valkey-info: ## Show Valkey information and cache statistics
valkey-cli: ## Open Valkey CLI for interactive commands
benchmark-valkey: ## Run Valkey cache performance benchmark

# Hold the shared legacy datastore lease across setup, tests, and cleanup.
VALKEY_PUBLIC_TARGETS := start-valkey stop-valkey restart-valkey valkey-status clean-valkey test-valkey-vectorstore test-valkey-vectorstore-no-container test-valkey-memory-unit test-valkey-memory test-valkey-memory-no-container test-valkey-cache test-semantic-router-valkey test-valkey-all run-valkey-cache-example run-valkey-cache-example-no-container run-valkey-vectorstore-example run-valkey-vectorstore-example-no-container run-valkey-example test-valkey-vectorstore-external run-valkey-example-external verify-valkey valkey-info valkey-cli benchmark-valkey
.PHONY: $(VALKEY_PUBLIC_TARGETS) $(addsuffix -unlocked,$(VALKEY_PUBLIC_TARGETS))
$(VALKEY_PUBLIC_TARGETS):
	@python3 tools/dev/with_test_resources.py --resource legacy-test-datastores -- $(MAKE) $@-unlocked

# ======== valkey.mk ========
# = Everything For Valkey   =
# ======== valkey.mk ========

##@ Valkey

# Valkey container management with valkey-search module
start-valkey-unlocked:
	@$(LOG_TARGET)
	@if $(CONTAINER_RUNTIME) ps --filter "name=valkey-semantic-cache" --format "{{.Names}}" | grep -q valkey-semantic-cache; then \
		echo "Valkey container is already running"; \
	elif $(CONTAINER_RUNTIME) ps -a --filter "name=valkey-semantic-cache" --format "{{.Names}}" | grep -q valkey-semantic-cache; then \
		echo "Starting existing Valkey container..."; \
		$(CONTAINER_RUNTIME) start valkey-semantic-cache; \
	else \
		mkdir -p /tmp/valkey-data; \
		echo "Starting Valkey bundle with search module on port 6380..."; \
		$(CONTAINER_RUNTIME) run -d \
			--name valkey-semantic-cache \
			-p 6380:6379 \
			-v /tmp/valkey-data:/data \
			valkey/valkey-bundle:unstable; \
		echo "Waiting for Valkey to be ready..."; \
		sleep 5; \
	fi
	@echo "Valkey with search module available at localhost:6380"
	@echo ""
	@echo "Note: valkey-bundle:unstable is required because valkey-search 1.2.0-rc3"
	@echo "      (included in this bundle) adds text search support needed by this project."
	@echo "      The search module is GA; the bundle tag will move to :latest once a stable"
	@echo "      release includes valkey-search >= 1.2.0."

stop-valkey-unlocked:
	@$(LOG_TARGET)
	@$(CONTAINER_RUNTIME) stop valkey-semantic-cache || true
	@$(CONTAINER_RUNTIME) rm valkey-semantic-cache || true
	@echo "Valkey container stopped and removed"

restart-valkey-unlocked:
	@$(MAKE) stop-valkey
	@$(MAKE) start-valkey

valkey-status-unlocked:
	@$(LOG_TARGET)
	@if $(CONTAINER_RUNTIME) ps --filter "name=valkey-semantic-cache" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -q valkey-semantic-cache; then \
		echo "Valkey container is running:"; \
		$(CONTAINER_RUNTIME) ps --filter "name=valkey-semantic-cache" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"; \
	else \
		echo "Valkey container is not running"; \
		echo "Run 'make start-valkey' to start it"; \
	fi

clean-valkey-unlocked: stop-valkey
	@$(LOG_TARGET)
	@echo "Cleaning up Valkey data..."
	@rm -rf /tmp/valkey-data 2>/dev/null || sudo -n rm -rf /tmp/valkey-data
	@echo "Valkey data directory cleaned"

# ---------------------------------------------------------------------------
# Vector Store Tests
# ---------------------------------------------------------------------------

# Test vector store with Valkey backend
test-valkey-vectorstore-unlocked: start-valkey rust
	@$(LOG_TARGET)
	@echo "Testing vector store with Valkey backend..."
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/nlp-binding/target/release && \
	export SR_TEST_MODE=true && \
	export VALKEY_HOST=localhost && \
	export VALKEY_PORT=6380 && \
	export SKIP_VALKEY_TESTS=false && \
		cd src/semantic-router && CGO_ENABLED=1 go test -v ./pkg/vectorstore/ -run TestVectorStore -- --focus="ValkeyBackend"
	@echo "Consider running 'make stop-valkey' when done testing"

# Test vector store against an already-running Valkey instance (no container management)
test-valkey-vectorstore-no-container-unlocked: rust
	@$(LOG_TARGET)
	@echo "Testing vector store against existing Valkey on port $${VALKEY_PORT:-6379}..."
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/nlp-binding/target/release && \
	export SR_TEST_MODE=true && \
	export VALKEY_HOST=$${VALKEY_HOST:-localhost} && \
	export VALKEY_PORT=$${VALKEY_PORT:-6379} && \
	export SKIP_VALKEY_TESTS=false && \
		cd src/semantic-router && CGO_ENABLED=1 go test -v ./pkg/vectorstore/ -run TestVectorStore -- --focus="ValkeyBackend"

# ---------------------------------------------------------------------------
# Memory Tests
# ---------------------------------------------------------------------------

# Test Valkey memory store (unit tests only, no container needed)
test-valkey-memory-unit-unlocked: rust
	@$(LOG_TARGET)
	@echo "Running ValkeyStore unit tests..."
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/nlp-binding/target/release && \
	export SR_TEST_MODE=true && \
		cd src/semantic-router && CGO_ENABLED=1 go test -v ./pkg/memory/ -run "TestValkey|TestMemoryValkeyConfig|TestNewValkeyStore|TestCachingStore|TestNewCachingStore" -count=1

# Test Valkey memory store integration (requires live Valkey with Search module)
test-valkey-memory-unlocked: start-valkey rust
	@$(LOG_TARGET)
	@echo "Testing Valkey memory store integration..."
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/nlp-binding/target/release && \
	export SR_TEST_MODE=true && \
	export VALKEY_HOST=localhost && \
	export VALKEY_PORT=6380 && \
	export SKIP_VALKEY_TESTS=false && \
		cd src/semantic-router && CGO_ENABLED=1 go test -v ./pkg/memory/ -run "TestValkeyStoreInteg" -count=1 -timeout 120s
	@echo "Consider running 'make stop-valkey' when done testing"

# Test Valkey memory store against an already-running Valkey instance
test-valkey-memory-no-container-unlocked: rust
	@$(LOG_TARGET)
	@echo "Testing Valkey memory store against existing Valkey on port ${VALKEY_PORT:-6379}..."
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/nlp-binding/target/release && \
	export SR_TEST_MODE=true && \
	export VALKEY_HOST=${VALKEY_HOST:-localhost} && \
	export VALKEY_PORT=${VALKEY_PORT:-6379} && \
	export SKIP_VALKEY_TESTS=false && \
		cd src/semantic-router && CGO_ENABLED=1 go test -v ./pkg/memory/ -run "TestValkeyStoreInteg" -count=1 -timeout 120s

# ---------------------------------------------------------------------------
# Cache Tests
# ---------------------------------------------------------------------------

# Test semantic cache with Valkey backend
test-valkey-cache-unlocked: start-valkey rust
	@$(LOG_TARGET)
	@echo "Testing semantic cache with Valkey backend..."
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/nlp-binding/target/release && \
	export SR_TEST_MODE=true && \
	export VALKEY_HOST=localhost && \
	export VALKEY_PORT=6380 && \
		cd src/semantic-router && CGO_ENABLED=1 go test -v ./pkg/cache/ -run TestValkeyCache
	@echo "Consider running 'make stop-valkey' when done testing"

# Test semantic-router with Valkey enabled
test-semantic-router-valkey-unlocked: build-router start-valkey
	@$(LOG_TARGET)
	@echo "Testing semantic-router with Valkey cache backend..."
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/nlp-binding/target/release && \
	export SR_TEST_MODE=true && \
	export VALKEY_HOST=localhost && \
	export VALKEY_PORT=6380 && \
		cd src/semantic-router && CGO_ENABLED=1 go test -v ./...
	@echo "Consider running 'make stop-valkey' when done testing"

# ---------------------------------------------------------------------------
# Combined Tests
# ---------------------------------------------------------------------------

# Test all Valkey backends (cache + vector store + memory)
test-valkey-all-unlocked: start-valkey rust
	@$(LOG_TARGET)
	@echo "Testing all Valkey backends..."
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/nlp-binding/target/release && \
	export SR_TEST_MODE=true && \
	export VALKEY_HOST=localhost && \
	export VALKEY_PORT=6380 && \
	export SKIP_VALKEY_TESTS=false && \
		cd src/semantic-router && CGO_ENABLED=1 go test -v ./pkg/vectorstore/ -run TestVectorStore -- --focus="ValkeyBackend" && \
		cd src/semantic-router && CGO_ENABLED=1 go test -v ./pkg/cache/ -run TestValkeyCache && \
		cd src/semantic-router && CGO_ENABLED=1 go test -v ./pkg/memory/ -run "TestValkeyStoreInteg" -count=1 -timeout 120s
	@echo "Consider running 'make stop-valkey' when done testing"

# ---------------------------------------------------------------------------
# Example Apps
# ---------------------------------------------------------------------------

# Run Valkey cache example
run-valkey-cache-example-unlocked: start-valkey rust
	@$(LOG_TARGET)
	@echo "Running Valkey cache example..."
	@cd src/semantic-router && \
		export LD_LIBRARY_PATH=${PWD}/../../candle-binding/target/release:${PWD}/../../nlp-binding/target/release && \
		export VALKEY_HOST=localhost && \
		export VALKEY_PORT=6380 && \
		go run ../../tools/valkey/valkey-cache.go
	@echo ""
	@echo "Example complete! Check Valkey using:"
	@echo "  • docker exec -it valkey-semantic-cache valkey-cli"

# Run Valkey cache example without starting container (use existing Valkey server)
run-valkey-cache-example-no-container-unlocked: rust
	@$(LOG_TARGET)
	@echo "Running Valkey cache example (using existing server)..."
	@echo "Note: Expects Valkey server at VALKEY_HOST:VALKEY_PORT (default: localhost:6379)"
	@cd src/semantic-router && \
		export LD_LIBRARY_PATH=${PWD}/../../candle-binding/target/release:${PWD}/../../nlp-binding/target/release && \
		export VALKEY_HOST=$${VALKEY_HOST:-localhost} && \
		export VALKEY_PORT=$${VALKEY_PORT:-6379} && \
		go run ../../tools/valkey/valkey-cache.go
	@echo ""
	@echo "Example complete!"

# Run Valkey vector store example
run-valkey-vectorstore-example-unlocked: start-valkey rust
	@$(LOG_TARGET)
	@echo "Running Valkey vector store example..."
	@cd src/semantic-router && \
		export LD_LIBRARY_PATH=${PWD}/../../candle-binding/target/release:${PWD}/../../nlp-binding/target/release && \
		go run ../../tools/valkey/valkey-vectorstore.go
	@echo ""
	@echo "Example complete! Inspect Valkey using:"
	@echo "  • make valkey-cli"
	@echo "  • make valkey-info"

# Run Valkey vector store example without starting container (use existing Valkey server)
run-valkey-vectorstore-example-no-container-unlocked: rust
	@$(LOG_TARGET)
	@echo "Running Valkey vector store example (using existing server)..."
	@cd src/semantic-router && \
		export LD_LIBRARY_PATH=${PWD}/../../candle-binding/target/release:${PWD}/../../nlp-binding/target/release && \
		go run ../../tools/valkey/valkey-vectorstore.go

# ---------------------------------------------------------------------------
# Deprecated / Compatibility Aliases
# ---------------------------------------------------------------------------

# run-valkey-example was renamed to run-valkey-cache-example; keep the old
# name as a forwarding alias so existing scripts don't break.
run-valkey-example-unlocked: run-valkey-cache-example

# Aliases matching the target names cited in the PR description.
test-valkey-vectorstore-external-unlocked: test-valkey-vectorstore-no-container
run-valkey-example-external-unlocked: run-valkey-vectorstore-example-no-container

# ---------------------------------------------------------------------------
# Verification & Utilities
# ---------------------------------------------------------------------------

# Verify Valkey installation
verify-valkey-unlocked: start-valkey
	@$(LOG_TARGET)
	@echo "Verifying Valkey installation..."
	@echo ""
	@echo "1. Testing basic connectivity..."
	@$(CONTAINER_RUNTIME) exec valkey-semantic-cache valkey-cli PING || \
		(echo "❌ Valkey connectivity failed" && exit 1)
	@echo "✓ Valkey is responding"
	@echo ""
	@echo "2. Checking Valkey version..."
	@$(CONTAINER_RUNTIME) exec valkey-semantic-cache valkey-cli INFO server | grep -E "valkey_version|redis_version" || true
	@echo ""
	@echo "3. Checking for valkey-search module..."
	@$(CONTAINER_RUNTIME) exec valkey-semantic-cache valkey-cli MODULE LIST | grep -i search || \
		(echo "❌ valkey-search module not found" && exit 1)
	@echo "✓ valkey-search module is loaded"
	@echo ""
	@echo "Valkey with search module is ready!"
	@echo ""
	@echo "Access Valkey:"
	@echo "  • CLI: docker exec -it valkey-semantic-cache valkey-cli"
	@echo "  • Port: localhost:6380"

# Check Valkey data
valkey-info-unlocked:
	@$(LOG_TARGET)
	@echo "Valkey Server Information:"
	@echo "════════════════════════════════════════"
	@$(CONTAINER_RUNTIME) exec valkey-semantic-cache valkey-cli INFO server | grep -E "valkey_version|redis_version|os|process_id|uptime" || true
	@echo ""
	@echo "Memory Usage:"
	@echo "════════════════════════════════════════"
	@$(CONTAINER_RUNTIME) exec valkey-semantic-cache valkey-cli INFO memory | grep -E "used_memory_human|used_memory_peak_human" || true
	@echo ""
	@echo "Cache Statistics:"
	@echo "════════════════════════════════════════"
	@$(CONTAINER_RUNTIME) exec valkey-semantic-cache valkey-cli DBSIZE
	@echo ""
	@echo "Check for semantic cache index:"
	@$(CONTAINER_RUNTIME) exec valkey-semantic-cache valkey-cli FT._LIST || echo "No indexes found"

# Valkey CLI access
valkey-cli-unlocked:
	@$(LOG_TARGET)
	@echo "Opening Valkey CLI (type 'exit' to quit)..."
	@echo ""
	@echo "Useful commands:"
	@echo "  KEYS doc:*               - List all cached documents"
	@echo "  FT.INFO semantic_cache_idx  - Show index info"
	@echo "  DBSIZE                   - Count total keys"
	@echo "  FLUSHDB                  - Clear all data (careful!)"
	@echo "  MODULE LIST              - Show loaded modules"
	@echo ""
	@$(CONTAINER_RUNTIME) exec -it valkey-semantic-cache valkey-cli

# Benchmark Valkey cache performance
benchmark-valkey-unlocked: rust start-valkey
	@$(LOG_TARGET)
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  Valkey Cache Performance Benchmark"
	@echo "  Testing cache operations with 1000 entries"
	@echo "═══════════════════════════════════════════════════════════"
	@echo ""
	@mkdir -p benchmark_results/valkey
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/nlp-binding/target/release && \
		export USE_CPU=${USE_CPU:-false} && \
		export SR_BENCHMARK_MODE=true && \
		export VALKEY_HOST=localhost && \
		export VALKEY_PORT=6380 && \
		cd src/semantic-router/pkg/cache && \
		out=../../../../benchmark_results/valkey/results.txt && \
		CGO_ENABLED=1 go test -v -timeout 30m \
		-run='^$$' -bench=BenchmarkValkeyCache \
		-benchtime=100x -benchmem . > "$$out" 2>&1; status=$$?; \
		cat "$$out"; \
		[ $$status -eq 0 ] || exit $$status; \
		grep -q 'ns/op' "$$out" || { echo "ERROR: -bench=BenchmarkValkeyCache matched no benchmark (silent-pass guard tripped)"; exit 1; }
	@echo ""
	@echo "Benchmark complete! Results in: benchmark_results/valkey/results.txt"
