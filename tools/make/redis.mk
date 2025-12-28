# ======== redis.mk ========
# = Everything For Redis   =
# ======== redis.mk ========

##@ Redis

# Redis container management
start-redis: ## Start Redis Stack container for testing
	@$(LOG_TARGET)
	@if $(CONTAINER_RUNTIME) ps --filter "name=redis-semantic-cache" --format "{{.Names}}" | grep -q redis-semantic-cache; then \
		echo "Redis container is already running"; \
	else \
		mkdir -p /tmp/redis-data; \
		$(CONTAINER_RUNTIME) run -d \
			--name redis-semantic-cache \
			-p 6379:6379 \
			-p 8001:8001 \
			-v /tmp/redis-data:/data \
			-e REDIS_ARGS="--save 60 1 --appendonly yes" \
			redis/redis-stack:latest; \
		echo "Waiting for Redis to be ready..."; \
		sleep 5; \
		echo "Redis should be available at localhost:6379"; \
		echo "RedisInsight UI available at http://localhost:8001"; \
	fi

stop-redis: ## Stop and remove Redis container
	@$(LOG_TARGET)
	@$(CONTAINER_RUNTIME) stop redis-semantic-cache || true
	@$(CONTAINER_RUNTIME) rm redis-semantic-cache || true
	@echo "Redis container stopped and removed"

restart-redis: stop-redis start-redis ## Restart Redis container

redis-status: ## Show status of Redis container
	@$(LOG_TARGET)
	@if $(CONTAINER_RUNTIME) ps --filter "name=redis-semantic-cache" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -q redis-semantic-cache; then \
		echo "Redis container is running:"; \
		$(CONTAINER_RUNTIME) ps --filter "name=redis-semantic-cache" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"; \
	else \
		echo "Redis container is not running"; \
		echo "Run 'make start-redis' to start it"; \
	fi

clean-redis: stop-redis ## Clean up Redis data
	@$(LOG_TARGET)
	@echo "Cleaning up Redis data..."
	@sudo rm -rf /tmp/redis-data || rm -rf /tmp/redis-data
	@echo "Redis data directory cleaned"

# Test semantic cache with Redis backend
test-redis-cache: start-redis rust ## Test semantic cache with Redis backend
	@$(LOG_TARGET)
	@echo "Testing semantic cache with Redis backend..."
	@export LD_LIBRARY_PATH=$${PWD}/candle-binding/target/release && \
	export SR_TEST_MODE=true && \
		cd src/semantic-router && CGO_ENABLED=1 go test -v ./pkg/cache/ -run TestRedisCache
	@echo "Consider running 'make stop-redis' when done testing"

# Test semantic-router with Redis enabled
test-semantic-router-redis: build-router start-redis ## Test semantic-router with Redis cache backend
	@$(LOG_TARGET)
	@echo "Testing semantic-router with Redis cache backend..."
	@export LD_LIBRARY_PATH=$${PWD}/candle-binding/target/release && \
	export SR_TEST_MODE=true && \
		cd src/semantic-router && CGO_ENABLED=1 go test -v ./...
	@echo "Consider running 'make stop-redis' when done testing"

# Run Redis cache example
run-redis-example: start-redis rust ## Run the Redis cache example
	@$(LOG_TARGET)
	@echo "Running Redis cache example..."
	@cd src/semantic-router && \
		export LD_LIBRARY_PATH=$${PWD}/../../candle-binding/target/release && \
		go run ../../deploy/examples/redis-cache-example.go
	@echo ""
	@echo "Example complete! Check Redis using:"
	@echo "  • redis-cli (command line)"
	@echo "  • http://localhost:8001 (RedisInsight UI)"

# Verify Redis installation
verify-redis: start-redis ## Verify Redis installation and vector search capability
	@$(LOG_TARGET)
	@echo "Verifying Redis installation..."
	@echo ""
	@echo "1. Testing basic connectivity..."
	@$(CONTAINER_RUNTIME) exec redis-semantic-cache redis-cli PING || \
		(echo "❌ Redis connectivity failed" && exit 1)
	@echo "✓ Redis is responding"
	@echo ""
	@echo "2. Checking RediSearch module..."
	@$(CONTAINER_RUNTIME) exec redis-semantic-cache redis-cli MODULE LIST | grep -q search || \
		(echo "❌ RediSearch module not found" && exit 1)
	@echo "✓ RediSearch module is loaded"
	@echo ""
	@echo "3. Checking RedisJSON module..."
	@$(CONTAINER_RUNTIME) exec redis-semantic-cache redis-cli MODULE LIST | grep -q ReJSON || \
		echo "⚠ RedisJSON module not found (optional)"
	@echo ""
	@echo "✓ Redis Stack is ready for semantic caching!"
	@echo ""
	@echo "Access Redis:"
	@echo "  • CLI: docker exec -it redis-semantic-cache redis-cli"
	@echo "  • UI:  http://localhost:8001"

# Check Redis data
redis-info: ## Show Redis information and cache statistics
	@$(LOG_TARGET)
	@echo "Redis Server Information:"
	@echo "════════════════════════════════════════"
	@$(CONTAINER_RUNTIME) exec redis-semantic-cache redis-cli INFO server | grep -E "redis_version|os|process_id|uptime"
	@echo ""
	@echo "Memory Usage:"
	@echo "════════════════════════════════════════"
	@$(CONTAINER_RUNTIME) exec redis-semantic-cache redis-cli INFO memory | grep -E "used_memory_human|used_memory_peak_human"
	@echo ""
	@echo "Cache Statistics:"
	@echo "════════════════════════════════════════"
	@$(CONTAINER_RUNTIME) exec redis-semantic-cache redis-cli DBSIZE
	@echo ""
	@echo "Check for semantic cache index:"
	@$(CONTAINER_RUNTIME) exec redis-semantic-cache redis-cli FT._LIST || echo "No indexes found"

# Redis CLI access
redis-cli: ## Open Redis CLI for interactive commands
	@$(LOG_TARGET)
	@echo "Opening Redis CLI (type 'exit' to quit)..."
	@echo ""
	@echo "Useful commands:"
	@echo "  KEYS doc:*               - List all cached documents"
	@echo "  FT.INFO semantic_cache_idx  - Show index info"
	@echo "  DBSIZE                   - Count total keys"
	@echo "  FLUSHDB                  - Clear all data (careful!)"
	@echo ""
	@$(CONTAINER_RUNTIME) exec -it redis-semantic-cache redis-cli

# Benchmark Redis cache performance
benchmark-redis: rust start-redis ## Run Redis cache performance benchmark
	@$(LOG_TARGET)
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  Redis Cache Performance Benchmark"
	@echo "  Testing cache operations with 1000 entries"
	@echo "═══════════════════════════════════════════════════════════"
	@echo ""
	@mkdir -p benchmark_results/redis
	@export LD_LIBRARY_PATH=$${PWD}/candle-binding/target/release && \
		export USE_CPU=$${USE_CPU:-false} && \
		export SR_BENCHMARK_MODE=true && \
		cd src/semantic-router/pkg/cache && \
		CGO_ENABLED=1 go test -v -timeout 30m \
		-run='^$$' -bench=BenchmarkRedisCache \
		-benchtime=100x -benchmem . | tee ../../../../benchmark_results/redis/results.txt
	@echo ""
	@echo "Benchmark complete! Results in: benchmark_results/redis/results.txt"

# Compare Redis vs Milvus vs In-Memory
benchmark-cache-comparison: rust start-redis start-milvus ## Compare all cache backends
	@$(LOG_TARGET)
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  Cache Backend Comparison Benchmark"
	@echo "  Testing: In-Memory, Redis, Milvus"
	@echo "═══════════════════════════════════════════════════════════"
	@echo ""
	@mkdir -p benchmark_results/comparison
	@export LD_LIBRARY_PATH=$${PWD}/candle-binding/target/release && \
		export USE_CPU=$${USE_CPU:-false} && \
		export SR_BENCHMARK_MODE=true && \
		cd src/semantic-router/pkg/cache && \
		CGO_ENABLED=1 go test -v -timeout 60m -tags=milvus \
		-run='^$$' -bench='BenchmarkCacheComparison' \
		-benchtime=50x -benchmem . | tee ../../../../benchmark_results/comparison/results.txt
	@echo ""
	@echo "Comparison complete! Results in: benchmark_results/comparison/results.txt"
	@echo ""
	@echo "To clean up:"
	@echo "  make stop-redis"
	@echo "  make stop-milvus"

