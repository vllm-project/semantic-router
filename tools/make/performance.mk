# ============== performance.mk ==============
# =   Performance testing related targets   =
# ============== performance.mk ==============

PERF_BASE_REF ?= HEAD^
PERF_MODEL_BASELINE_DIR ?= $(CURDIR)/reports
PERF_MODEL_BASELINE ?= $(PERF_MODEL_BASELINE_DIR)/model-baseline.json
export VLLM_SR_MODEL_MANIFEST ?= $(PERF_MODEL_MANIFEST)

perf-model-baseline: download-models-perf ## Measure the same Vela artifacts against PERF_BASE_REF
	@export $(NATIVE_ENV) && bash perf/scripts/compare-model-baseline.sh \
		"$(PERF_BASE_REF)" "$(PERF_MODEL_BASELINE_DIR)"

.PHONY: perf-model-baseline

##@ Performance Testing

perf-test-unit: ## Test benchmark parsing, artifact identity, and regression contracts
	@$(LOG_TARGET)
	@cd perf && go test ./pkg/... ./cmd/...

.PHONY: perf-test-unit

perf-test-benchmark-contracts: $(if $(CI),rust-ci,rust) ## Test benchmark owners and device selection without checkpoint inference
	@export $(NATIVE_ENV) && cd perf && go test -count=1 ./benchmarks/...

.PHONY: perf-test-benchmark-contracts

# Create reports directory if it doesn't exist
.PHONY: ensure-reports-dir
ensure-reports-dir:
	@mkdir -p reports

# Run all performance benchmarks
perf-bench: ## Run all performance benchmarks
perf-bench: $(if $(CI),rust-ci,rust) ensure-reports-dir
	@$(LOG_TARGET)
	@echo "Running performance benchmarks..."
	@export $(NATIVE_ENV) && \
	cd perf && go test -run "^$$" -bench=. -benchmem -benchtime=10s ./benchmarks/... \
	  -cpuprofile=../reports/cpu.prof \
	  -memprofile=../reports/mem.prof \
	  -timeout=30m

# Run quick performance benchmarks (shorter benchtime for faster iteration)
perf-bench-quick: ## Run quick performance benchmarks (3s benchtime)
perf-bench-quick: $(if $(CI),rust-ci,rust) ensure-reports-dir
	@$(LOG_TARGET)
	@echo "Running quick performance benchmarks..."
	@export $(NATIVE_ENV) && \
	cd perf && go test -run "^$$" -bench=. -benchmem -benchtime=3s ./benchmarks/... \
	  -timeout=15m

# Run specific benchmark suite
perf-bench-classification: ## Run classification benchmarks
perf-bench-classification: $(if $(CI),rust-ci,rust) ensure-reports-dir
	@$(LOG_TARGET)
	@export $(NATIVE_ENV) && \
	cd perf && go test -run "^$$" -bench=BenchmarkClassify.* -benchmem -benchtime=10s ./benchmarks/

perf-bench-decision: ## Run decision engine benchmarks
perf-bench-decision: $(if $(CI),rust-ci,rust) ensure-reports-dir
	@$(LOG_TARGET)
	@export $(NATIVE_ENV) && \
	cd perf && go test -run "^$$" -bench=BenchmarkEvaluate.* -benchmem -benchtime=10s ./benchmarks/

perf-bench-cache: ## Run cache benchmarks
perf-bench-cache: $(if $(CI),rust-ci,rust) ensure-reports-dir
	@$(LOG_TARGET)
	@export $(NATIVE_ENV) && \
	cd perf && go test -run "^$$" -bench=BenchmarkCache.* -benchmem -benchtime=10s ./benchmarks/

# Run Looper family benchmarks. Unlike the other component benchmarks, the
# Looper/Fusion/Flow/ReMoM benches live in the main module (src/semantic-router)
# next to the unexported hot-path functions they measure, so go test runs there.
perf-bench-looper: ## Run Looper family (ReMoM/Fusion/Flow/Base) benchmarks
perf-bench-looper: $(if $(CI),rust-ci,rust) ensure-reports-dir
	@$(LOG_TARGET)
	@export $(NATIVE_ENV) && \
	cd src/semantic-router && bash -o pipefail -c \
	  'go test -run "^$$" -bench="^Benchmark(ReMoM|Fusion|Flow|Base)" -benchmem -benchtime=10s ./pkg/looper/... | tee ../../reports/bench-results-looper.txt'

# Run E2E performance tests
perf-e2e: ## Run E2E performance tests
perf-e2e: build-e2e ensure-reports-dir
	@$(LOG_TARGET)
	@echo "Running E2E performance tests..."
	@./bin/e2e -profile=envoy-ai-gateway \
	  -tests=performance-throughput,performance-latency,performance-resource

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
	  --model-baseline="$(PERF_MODEL_BASELINE)" \
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
	@export $(NATIVE_ENV) && \
	cd perf && bash -o pipefail -c \
	  'go test -run "^$$" -bench=. -benchmem -benchtime=30s ./benchmarks/... | tee ../reports/bench-results.txt'
	@echo "Running Looper family benchmarks to update baseline..."
	@export $(NATIVE_ENV) && \
	cd src/semantic-router && bash -o pipefail -c \
	  'go test -run "^$$" -bench="^Benchmark(ReMoM|Fusion|Flow|Base)" -benchmem -benchtime=30s ./pkg/looper/... | tee -a ../../reports/bench-results.txt'
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
perf-bench-concurrency: $(if $(CI),rust-ci,rust) ensure-reports-dir
	@$(LOG_TARGET)
	@export $(NATIVE_ENV) && \
	export GOMAXPROCS=$${CONCURRENCY:-4} && \
	cd perf && go test -run "^$$" -bench=.*Parallel -benchmem -benchtime=10s ./benchmarks/...

# Run benchmarks and fail if any benchmark regresses beyond its threshold.
# Compares against the committed baselines in perf/testdata/baselines — refresh
# them for your hardware with 'make perf-baseline-update' before trusting a
# local pass/fail. Model paths use the same current harness and artifacts on
# PERF_BASE_REF; other components retain reviewed committed baselines. Time is
# advisory; the gate measures Go allocations, not native allocator/RSS memory.
perf-check: ## Run benchmarks and fail if regressions exceed thresholds
perf-check: $(if $(CI),rust-ci,rust) ensure-reports-dir perf-model-baseline
	@$(LOG_TARGET)
	@echo "Running benchmarks for regression check..."
	@export $(NATIVE_ENV) && \
	cd perf && bash -o pipefail -c \
	  'go test -run "^$$" -bench=. -benchmem -benchtime=10s ./benchmarks/... | tee ../reports/bench-output.txt'
	@export $(NATIVE_ENV) && \
	cd src/semantic-router && bash -o pipefail -c \
	  'go test -run "^$$" -bench="^Benchmark(ReMoM|Fusion|Flow|Base)" -benchmem -benchtime=10s ./pkg/looper/... | tee -a ../../reports/bench-output.txt'
	@echo "Building current results and comparing against baseline..."
	@cd perf && go run cmd/perftest/main.go \
	  --parse-bench=../reports/bench-output.txt \
	  --output=../reports/current.json
	@cd perf && go run cmd/perftest/main.go \
	  --compare-baseline=testdata/baselines/ \
	  --model-baseline="$(PERF_MODEL_BASELINE)" \
	  --current=../reports/current.json \
	  --threshold-file=config/thresholds.yaml \
	  --output=../reports/comparison.json \
	  --inventory=config/benchmark-inventory.json \
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
	@echo ""
	@echo "Baselines & Reports:"
	@echo "  make perf-baseline-update    - Update performance baselines"
	@echo "  make perf-report             - Generate HTML report"
	@echo ""
	@echo "Cleanup:"
	@echo "  make perf-clean              - Clean performance artifacts"
