# ======== milvus.mk ========
# = Everything For milvus   =
# ======== milvus.mk ========

##@ Milvus

# Milvus container management
start-milvus: ## Start Milvus container for testing
	@$(LOG_TARGET)
	@mkdir -p /tmp/milvus-data
	@$(CONTAINER_RUNTIME) run -d \
		--name milvus-semantic-cache \
		--security-opt seccomp:unconfined \
		-e ETCD_USE_EMBED=true \
		-e ETCD_DATA_DIR=/var/lib/milvus/etcd \
		-e ETCD_CONFIG_PATH=/milvus/configs/advanced/etcd.yaml \
		-e COMMON_STORAGETYPE=local \
		-e CLUSTER_ENABLED=false \
		-p 19530:19530 \
		-p 9091:9091 \
		-v /tmp/milvus-data:/var/lib/milvus:z \
		milvusdb/milvus:v2.3.3 \
		milvus run standalone
	@echo "Waiting for Milvus to be ready..."
	@sleep 15
	@echo "Milvus should be available at localhost:19530"

stop-milvus: ## Stop and remove Milvus container
	@$(LOG_TARGET)
	@$(CONTAINER_RUNTIME) stop milvus-semantic-cache || true
	@$(CONTAINER_RUNTIME) rm milvus-semantic-cache || true
	@sudo rm -rf /tmp/milvus-data || true
	@echo "Milvus container stopped and removed"

restart-milvus: stop-milvus start-milvus ## Restart Milvus container

milvus-status: ## Show status of Milvus container
	@$(LOG_TARGET)
	@if $(CONTAINER_RUNTIME) ps --filter "name=milvus-semantic-cache" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -q milvus-semantic-cache; then \
		echo "Milvus container is running:"; \
		$(CONTAINER_RUNTIME) ps --filter "name=milvus-semantic-cache" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"; \
	else \
		echo "Milvus container is not running"; \
		echo "Run 'make start-milvus' to start it"; \
	fi

clean-milvus: stop-milvus ## Clean up Milvus data
	@$(LOG_TARGET)
	@echo "Cleaning up Milvus data..."
	@sudo rm -rf milvus-data || rm -rf milvus-data
	@echo "Milvus data directory cleaned"

# Test semantic cache with Milvus backend
test-milvus-cache: start-milvus rust
	@$(LOG_TARGET)
	@echo "Testing semantic cache with Milvus backend..."
	@export LD_LIBRARY_PATH=$${PWD}/candle-binding/target/release && \
	export SR_TEST_MODE=true && \
		cd src/semantic-router && CGO_ENABLED=1 go test -tags=milvus -v ./pkg/cache/
	@echo "Consider running 'make stop-milvus' when done testing"

# Test semantic-router with Milvus enabled
test-semantic-router-milvus: build-router start-milvus
	@$(LOG_TARGET)
	@echo "Testing semantic-router with Milvus cache backend..."
	@export LD_LIBRARY_PATH=$${PWD}/candle-binding/target/release && \
	export SR_TEST_MODE=true && \
		cd src/semantic-router && CGO_ENABLED=1 go test -tags=milvus -v ./...
	@echo "Consider running 'make stop-milvus' when done testing"

# Milvus UI (Attu) management
start-milvus-ui: ## Start Attu UI to browse Milvus data
	@$(LOG_TARGET)
	@echo "Starting Attu (Milvus UI) with $(CONTAINER_RUNTIME)..."
	@$(CONTAINER_RUNTIME) run -d \
		--name milvus-ui \
		--add-host=host.docker.internal:host-gateway \
		-e MILVUS_URL=host.docker.internal:19530 \
		-p 18000:3000 \
		zilliz/attu:v2.3.5
	@echo "Waiting for Attu to be ready..."
	@sleep 3
	@echo "Open UI: http://localhost:18000 (Milvus at host.docker.internal:19530)"

stop-milvus-ui:
	@$(LOG_TARGET)
	@echo "Stopping Attu (Milvus UI) container..."
	@$(CONTAINER_RUNTIME) stop milvus-ui || true
	@$(CONTAINER_RUNTIME) rm milvus-ui || true
	@echo "Attu container stopped and removed"

# Hybrid vs Milvus Benchmarks
benchmark-hybrid-vs-milvus: rust start-milvus ## Run comprehensive Hybrid Cache vs Milvus benchmarks
	@$(LOG_TARGET)
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  Hybrid Cache vs Milvus Benchmark Suite"
	@echo "  Validating claims from hybrid HNSW storage paper"
	@echo "  Cache sizes: 10K, 50K, 100K entries"
	@echo "═══════════════════════════════════════════════════════════"
	@echo ""
	@echo "GPU Usage:"
	@echo "  • To use GPU: USE_CPU=false make benchmark-hybrid-vs-milvus"
	@echo "  • Select GPUs: CUDA_VISIBLE_DEVICES=2,3 USE_CPU=false make benchmark-hybrid-vs-milvus"
	@echo "  • Default: Uses GPU if available (USE_CPU=false)"
	@echo ""
	@bash scripts/run_hybrid_vs_milvus_benchmarks.sh
	@echo ""
	@echo "Benchmarks complete! Results in: benchmark_results/hybrid_vs_milvus/"
	@echo ""
	@echo "Next steps:"
	@echo "  make analyze-hybrid-benchmarks    # Analyze results"
	@echo "  make plot-hybrid-benchmarks       # Generate plots"
	@echo "  make stop-milvus                  # Clean up"

analyze-hybrid-benchmarks: ## Analyze Hybrid vs Milvus benchmark results
	@$(LOG_TARGET)
	@echo "Checking for CSV results in benchmark_results/hybrid_vs_milvus/..."
	@if ls benchmark_results/hybrid_vs_milvus/results_*.csv >/dev/null 2>&1; then \
		echo "Found CSV results, analyzing..."; \
		python3 scripts/analyze_hybrid_benchmarks.py; \
	elif [ -f /tmp/benchmark_batch_fixed.log ]; then \
		echo "No CSV found, parsing from log file..."; \
		python3 scripts/parse_hybrid_benchmark_log.py /tmp/benchmark_batch_fixed.log; \
	else \
		echo "$(shell tput setaf 3)No benchmark results found. Run 'make benchmark-hybrid-quick' first.$(shell tput sgr0)"; \
		exit 1; \
	fi

plot-hybrid-benchmarks: ## Generate plots from Hybrid vs Milvus benchmarks
	@$(LOG_TARGET)
	@python3 scripts/plot_hybrid_comparison.py

benchmark-hybrid-quick: rust ## Run quick Hybrid vs Milvus benchmark (smaller scale)
	@$(LOG_TARGET)
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  Quick Hybrid vs Milvus Benchmark (10K entries only)"
	@echo "  Estimated time: 7-10 minutes"
	@echo "═══════════════════════════════════════════════════════════"
	@echo ""
	@echo "Cleaning and restarting Milvus..."
	@$(CONTAINER_RUNTIME) stop milvus-semantic-cache 2>/dev/null || true
	@$(CONTAINER_RUNTIME) rm milvus-semantic-cache 2>/dev/null || true
	@sudo rm -rf /tmp/milvus-data 2>/dev/null || true
	@$(MAKE) start-milvus
	@sleep 5
	@echo ""
	@echo "GPU Usage:"
	@echo "  • To use GPU: USE_CPU=false make benchmark-hybrid-quick"
	@echo "  • Select GPUs: CUDA_VISIBLE_DEVICES=2,3 USE_CPU=false make benchmark-hybrid-quick"
	@echo ""
	@echo "Test Options:"
	@echo "  • Hybrid only: SKIP_MILVUS=true make benchmark-hybrid-quick"
	@echo "  • Both caches: make benchmark-hybrid-quick (default)"
	@echo ""
	@mkdir -p benchmark_results/hybrid_vs_milvus
	@export LD_LIBRARY_PATH=$${PWD}/candle-binding/target/release && \
		export USE_CPU=$${USE_CPU:-false} && \
		export SKIP_MILVUS=$${SKIP_MILVUS:-false} && \
		export SR_BENCHMARK_MODE=true && \
		echo "Using GPU mode: USE_CPU=$$USE_CPU" && \
		echo "Skip Milvus: SKIP_MILVUS=$$SKIP_MILVUS" && \
		cd src/semantic-router/pkg/cache && \
		CGO_ENABLED=1 go test -v -timeout 60m -tags=milvus \
		-run='^$$' -bench='^BenchmarkHybridVsMilvus/CacheSize_10000$$' \
		-benchtime=50x -benchmem .
	@echo ""
	@echo "Quick benchmark complete!"
	@echo "Results in: benchmark_results/hybrid_vs_milvus/"

benchmark-hybrid-only: rust ## Run ONLY Hybrid cache benchmark (skip Milvus for faster testing)
	@$(LOG_TARGET)
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  Hybrid Cache ONLY Benchmark (10K entries)"
	@echo "  Estimated time: 3-5 minutes"
	@echo "═══════════════════════════════════════════════════════════"
	@echo ""
	@echo "Cleaning and restarting Milvus..."
	@$(CONTAINER_RUNTIME) stop milvus-semantic-cache 2>/dev/null || true
	@$(CONTAINER_RUNTIME) rm milvus-semantic-cache 2>/dev/null || true
	@sudo rm -rf /tmp/milvus-data 2>/dev/null || true
	@$(MAKE) start-milvus
	@sleep 5
	@echo ""
	@echo "GPU Usage:"
	@echo "  • To use GPU: USE_CPU=false make benchmark-hybrid-only"
	@echo "  • Select GPUs: CUDA_VISIBLE_DEVICES=2,3 USE_CPU=false make benchmark-hybrid-only"
	@echo ""
	@mkdir -p benchmark_results/hybrid_vs_milvus
	@export LD_LIBRARY_PATH=$${PWD}/candle-binding/target/release && \
		export USE_CPU=$${USE_CPU:-false} && \
		export SKIP_MILVUS=true && \
		export SR_BENCHMARK_MODE=true && \
		echo "Using GPU mode: USE_CPU=$$USE_CPU" && \
		echo "Testing HYBRID CACHE ONLY (Milvus skipped)" && \
		cd src/semantic-router/pkg/cache && \
		CGO_ENABLED=1 go test -v -timeout 60m -tags=milvus \
		-run='^$$' -bench='^BenchmarkHybridVsMilvus/CacheSize_10000$$' \
		-benchtime=50x -benchmem .
	@echo ""
	@echo "Hybrid-only benchmark complete!"
	@echo "Results in: benchmark_results/hybrid_vs_milvus/"
