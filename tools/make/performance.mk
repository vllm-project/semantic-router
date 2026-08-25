# ============== performance.mk ==============
# =   Performance testing related targets   =
# ============== performance.mk ==============

##@ Performance Testing

ACCESS_CAPACITY_KEYS ?= 10000
ACCESS_CAPACITY_REPLICAS ?= 4
ACCESS_CAPACITY_CONCURRENCY ?= 64
ACCESS_CAPACITY_REQUEST_LIMIT ?= 12
ACCESS_CAPACITY_TIMEOUT ?= 20m
ACCESS_CAPACITY_USAGE_DRAIN_TIMEOUT ?= 30s
ACCESS_CAPACITY_MAX_ADMISSION_P99 ?= 100ms
ACCESS_CAPACITY_MAX_USAGE_LAG_P99 ?= 5s
ACCESS_CAPACITY_MIN_PROJECTION_KEYS_PER_SECOND ?= 100
ACCESS_CAPACITY_MAX_PROJECTION_BYTES_PER_KEY ?= 32768
ACCESS_CAPACITY_MAX_EVENT_BYTES ?= 16384
ACCESS_CAPACITY_OUTPUT_ROOT ?= $(CURDIR)/.agent-harness/access-capacity

# Create reports directory if it doesn't exist
.PHONY: ensure-reports-dir
ensure-reports-dir:
	@mkdir -p reports

# Run all performance benchmarks
perf-bench: ## Run all performance benchmarks
perf-bench: build-router ensure-reports-dir
	@$(LOG_TARGET)
	@echo "Running performance benchmarks..."
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/ml-binding/target/release:${PWD}/nlp-binding/target/release && \
	cd perf && go test -bench=. -benchmem -benchtime=10s ./benchmarks/... \
	  -cpuprofile=../reports/cpu.prof \
	  -memprofile=../reports/mem.prof \
	  -timeout=30m

# Run quick performance benchmarks (shorter benchtime for faster iteration)
perf-bench-quick: ## Run quick performance benchmarks (3s benchtime)
perf-bench-quick: build-router ensure-reports-dir
	@$(LOG_TARGET)
	@echo "Running quick performance benchmarks..."
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/ml-binding/target/release:${PWD}/nlp-binding/target/release && \
	cd perf && go test -bench=. -benchmem -benchtime=3s ./benchmarks/... \
	  -timeout=15m

# Run specific benchmark suite
perf-bench-classification: ## Run classification benchmarks
perf-bench-classification: build-router ensure-reports-dir
	@$(LOG_TARGET)
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/ml-binding/target/release:${PWD}/nlp-binding/target/release && \
	cd perf && go test -bench=BenchmarkClassify.* -benchmem -benchtime=10s ./benchmarks/

perf-bench-decision: ## Run decision engine benchmarks
perf-bench-decision: build-router ensure-reports-dir
	@$(LOG_TARGET)
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/ml-binding/target/release:${PWD}/nlp-binding/target/release && \
	cd perf && go test -bench=BenchmarkEvaluate.* -benchmem -benchtime=10s ./benchmarks/

perf-bench-cache: ## Run cache benchmarks
perf-bench-cache: build-router ensure-reports-dir
	@$(LOG_TARGET)
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/ml-binding/target/release:${PWD}/nlp-binding/target/release && \
	cd perf && go test -bench=BenchmarkCache.* -benchmem -benchtime=10s ./benchmarks/

# Run Looper family benchmarks. Unlike the other component benchmarks, the
# Looper/Fusion/Flow/ReMoM benches live in the main module (src/semantic-router)
# next to the unexported hot-path functions they measure, so go test runs there.
perf-bench-looper: ## Run Looper family (ReMoM/Fusion/Flow/Base) benchmarks
perf-bench-looper: build-router ensure-reports-dir
	@$(LOG_TARGET)
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/ml-binding/target/release:${PWD}/nlp-binding/target/release && \
	cd src/semantic-router && go test -bench='^Benchmark(ReMoM|Fusion|Flow|Base)' -benchmem -benchtime=10s ./pkg/looper/... \
	  | tee ../../reports/bench-results-looper.txt

# Run E2E performance tests
perf-e2e: ## Run E2E performance tests
perf-e2e: build-e2e ensure-reports-dir
	@$(LOG_TARGET)
	@echo "Running E2E performance tests..."
	@./bin/e2e -profile=envoy-ai-gateway \
	  -tests=performance-throughput,performance-latency,performance-resource

# This opt-in gate uses the production access publication/runtime packages and
# an isolated prefix in a caller-supplied Redis/Valkey. It is not an HTTP E2E
# and never serializes the connection URL into its report.
.PHONY: perf-access-capacity
perf-access-capacity: ## Run the 10,000-key Router access-runtime capacity gate
	@$(LOG_TARGET)
	@test -n "$${ACCESS_CAPACITY_REDIS_URL:-}" || { \
		echo "ACCESS_CAPACITY_REDIS_URL is required (use an isolated Redis/Valkey)"; \
		exit 1; \
	}
	@cd src/semantic-router && go run ./cmd/access-capacity-gate \
		--keys "$(ACCESS_CAPACITY_KEYS)" \
		--replicas "$(ACCESS_CAPACITY_REPLICAS)" \
		--concurrency "$(ACCESS_CAPACITY_CONCURRENCY)" \
		--request-limit "$(ACCESS_CAPACITY_REQUEST_LIMIT)" \
		--timeout "$(ACCESS_CAPACITY_TIMEOUT)" \
		--usage-drain-timeout "$(ACCESS_CAPACITY_USAGE_DRAIN_TIMEOUT)" \
		--max-admission-p99 "$(ACCESS_CAPACITY_MAX_ADMISSION_P99)" \
		--max-usage-lag-p99 "$(ACCESS_CAPACITY_MAX_USAGE_LAG_P99)" \
		--min-projection-keys-per-second "$(ACCESS_CAPACITY_MIN_PROJECTION_KEYS_PER_SECOND)" \
		--max-projection-bytes-per-key "$(ACCESS_CAPACITY_MAX_PROJECTION_BYTES_PER_KEY)" \
		--max-event-bytes "$(ACCESS_CAPACITY_MAX_EVENT_BYTES)" \
		--output-root "$(ACCESS_CAPACITY_OUTPUT_ROOT)"

# Compare against baseline (report only; use perf-check to fail on regression).
# Consumes reports/bench-output.txt (a captured benchmark run — 'make perf-check'
# writes it, or tee one there yourself; CI also produces it), turns it into
# reports/current.json, and diffs it against every per-suite baseline in
# perf/testdata/baselines.
perf-compare: ## Compare current benchmark output against baseline (report only)
perf-compare: ensure-reports-dir
	@$(LOG_TARGET)
	@test -f reports/bench-output.txt || { \
	  echo "reports/bench-output.txt not found — run 'make perf-check' (runs benchmarks + gates), or capture a run first: make perf-bench-quick 2>&1 | tee reports/bench-output.txt"; \
	  exit 1; }
	@echo "Building current results from benchmark output..."
	@cd perf && go run cmd/perftest/main.go \
	  --parse-bench=../reports/bench-output.txt \
	  --output=../reports/current.json
	@echo "Comparing performance against baseline..."
	@cd perf && go run cmd/perftest/main.go \
	  --compare-baseline=testdata/baselines/ \
	  --current=../reports/current.json \
	  --threshold-file=config/thresholds.yaml \
	  --output=../reports/comparison.json

# Run benchmarks with CPU profiling
perf-profile-cpu: ## Run benchmarks with CPU profiling and open pprof
perf-profile-cpu: perf-bench
	@$(LOG_TARGET)
	@echo "Opening CPU profile..."
	@go tool pprof -http=:8080 reports/cpu.prof

# Run benchmarks with memory profiling
perf-profile-mem: ## Run benchmarks with memory profiling and open pprof
perf-profile-mem: perf-bench
	@$(LOG_TARGET)
	@echo "Opening memory profile..."
	@go tool pprof -http=:8080 reports/mem.prof

# Generate CPU flame graph
perf-flamegraph: ## Generate CPU flame graph
perf-flamegraph: perf-bench
	@$(LOG_TARGET)
	@echo "Generating CPU flame graph..."
	@go tool pprof -http=:8080 reports/cpu.prof &

# Update performance baselines
perf-baseline-update: ## Update performance baselines
perf-baseline-update: ensure-reports-dir
	@$(LOG_TARGET)
	@echo "Running benchmarks to update baseline..."
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/ml-binding/target/release:${PWD}/nlp-binding/target/release && \
	cd perf && go test -bench=. -benchmem -benchtime=30s ./benchmarks/... \
	  | tee ../reports/bench-results.txt
	@echo "Running Looper family benchmarks to update baseline..."
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/ml-binding/target/release:${PWD}/nlp-binding/target/release && \
	cd src/semantic-router && go test -bench='^Benchmark(ReMoM|Fusion|Flow|Base)' -benchmem -benchtime=30s ./pkg/looper/... \
	  | tee -a ../../reports/bench-results.txt
	@echo "Updating baselines..."
	@cd perf/scripts && ./update-baseline.sh

# Generate performance report
perf-report: ## Generate performance report (requires comparison.json)
perf-report: ensure-reports-dir
	@$(LOG_TARGET)
	@echo "Generating performance report..."
	@cd perf && go run cmd/perftest/main.go \
	  --generate-report \
	  --input=../reports/comparison.json \
	  --output=../reports/perf-report.html

# Clean performance test artifacts
perf-clean: ## Clean performance test artifacts
	@$(LOG_TARGET)
	@echo "Cleaning performance test artifacts..."
	@rm -rf reports/*.prof reports/*.json reports/*.html reports/*.md
	@echo "Performance artifacts cleaned"

# Run continuous performance monitoring (for local development)
perf-watch: ## Continuously run quick benchmarks on file changes
	@echo "Watching for changes and running quick benchmarks..."
	@while true; do \
		make perf-bench-quick; \
		echo "Waiting for changes... (Ctrl+C to stop)"; \
		sleep 30; \
	done

# Performance test with specific concurrency
perf-bench-concurrency: ## Run benchmarks with specific concurrency (e.g., CONCURRENCY=4)
perf-bench-concurrency: build-router ensure-reports-dir
	@$(LOG_TARGET)
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/ml-binding/target/release:${PWD}/nlp-binding/target/release && \
	export GOMAXPROCS=$${CONCURRENCY:-4} && \
	cd perf && go test -bench=.*Parallel -benchmem -benchtime=10s ./benchmarks/...

# Run benchmarks and fail if any benchmark regresses beyond its threshold.
# Compares against the committed baselines in perf/testdata/baselines — refresh
# them for your hardware with 'make perf-baseline-update' before trusting a
# local pass/fail, since absolute ns/op is machine-dependent. CI self-baselines
# against main in the same runner (see .github/workflows/performance-test.yml),
# so it needs no committed baseline.
perf-check: ## Run benchmarks and fail if regressions exceed thresholds
perf-check: build-router ensure-reports-dir
	@$(LOG_TARGET)
	@echo "Running benchmarks for regression check..."
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/ml-binding/target/release:${PWD}/nlp-binding/target/release && \
	cd perf && go test -bench=. -benchmem -benchtime=10s ./benchmarks/... | tee ../reports/bench-output.txt
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/ml-binding/target/release:${PWD}/nlp-binding/target/release && \
	cd src/semantic-router && go test -bench='^Benchmark(ReMoM|Fusion|Flow|Base)' -benchmem -benchtime=10s ./pkg/looper/... | tee -a ../../reports/bench-output.txt
	@echo "Building current results and comparing against baseline..."
	@cd perf && go run cmd/perftest/main.go \
	  --parse-bench=../reports/bench-output.txt \
	  --output=../reports/current.json
	@cd perf && go run cmd/perftest/main.go \
	  --compare-baseline=testdata/baselines/ \
	  --current=../reports/current.json \
	  --threshold-file=config/thresholds.yaml \
	  --output=../reports/comparison.json \
	  --fail-on-regression

# Show performance test help
perf-help: ## Show performance testing help
	@echo "Performance Testing Targets:"
	@echo ""
	@echo "Quick Start:"
	@echo "  make perf-bench              - Run all benchmarks (10s per test)"
	@echo "  make perf-bench-quick        - Run quick benchmarks (3s per test)"
	@echo "  make perf-compare            - Compare against baseline"
	@echo "  make perf-check              - Run benchmarks and fail on regression"
	@echo ""
	@echo "Component Benchmarks:"
	@echo "  make perf-bench-classification - Benchmark classification"
	@echo "  make perf-bench-decision       - Benchmark decision engine"
	@echo "  make perf-bench-cache          - Benchmark cache"
	@echo ""
	@echo "Profiling:"
	@echo "  make perf-profile-cpu        - Profile CPU usage"
	@echo "  make perf-profile-mem        - Profile memory usage"
	@echo "  make perf-flamegraph         - Generate flame graph"
	@echo ""
	@echo "E2E Performance:"
	@echo "  make perf-e2e                - Run E2E performance tests"
	@echo "  make perf-access-capacity    - Run opt-in 10,000-key access-runtime gate"
	@echo ""
	@echo "Baselines & Reports:"
	@echo "  make perf-baseline-update    - Update performance baselines"
	@echo "  make perf-report             - Generate HTML report"
	@echo ""
	@echo "Cleanup:"
	@echo "  make perf-clean              - Clean performance artifacts"
