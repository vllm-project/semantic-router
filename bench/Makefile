.PHONY: all bench bench-quick bench-full bench-concurrent bench-memory bench-compare clean help

# Default target
all: help

# Run quick benchmarks (shorter time)
bench-quick:
	@echo "Running quick jailbreak benchmarks..."
	go test -bench=. -benchmem -benchtime=3s

# Run full benchmarks (longer time for more accuracy)
bench-full:
	@echo "Running full jailbreak benchmarks..."
	@mkdir -p results
	go test -bench=. -benchmem -benchtime=10s | tee results/bench_$$(date +%Y%m%d_%H%M%S).txt

# Run only concurrent benchmarks
bench-concurrent:
	@echo "Running concurrency benchmarks..."
	go test -bench=Concurrent -benchmem -benchtime=30s

# Run with memory profiling
bench-memory:
	@echo "Running benchmarks with memory profiling..."
	@mkdir -p results
	go test -bench=. -benchmem -memprofile=results/mem.prof -cpuprofile=results/cpu.prof
	@echo "View memory profile: go tool pprof results/mem.prof"
	@echo "View CPU profile: go tool pprof results/cpu.prof"

# Compare with previous results (requires benchstat)
bench-compare:
	@if ! command -v benchstat >/dev/null 2>&1; then \
		echo "Installing benchstat..."; \
		go install golang.org/x/perf/cmd/benchstat@latest; \
	fi
	@if [ -z "$$(ls -t results/bench_*.txt 2>/dev/null | head -2 | tail -1)" ]; then \
		echo "No previous results found. Run 'make bench-full' first."; \
		exit 1; \
	fi
	@echo "Comparing with previous results..."
	@OLD=$$(ls -t results/bench_*.txt | head -2 | tail -1); \
	 NEW=$$(ls -t results/bench_*.txt | head -1); \
	 echo "Old: $$OLD"; \
	 echo "New: $$NEW"; \
	 benchstat $$OLD $$NEW

# Run benchmarks for specific model
bench-modernbert:
	@echo "Running ModernBERT benchmarks..."
	go test -bench=BenchmarkModernBert -benchmem

bench-deberta:
	@echo "Running DeBERTa benchmarks..."
	go test -bench=BenchmarkDeberta -benchmem

bench-unified:
	@echo "Running Unified classifier benchmarks..."
	go test -bench=BenchmarkUnified -benchmem

# Run only initialization benchmarks
bench-init:
	@echo "Running initialization benchmarks..."
	go test -bench=BenchmarkInit -benchmem

# Run benchmarks with race detector (slower but checks for race conditions)
bench-race:
	@echo "Running benchmarks with race detector..."
	go test -race -bench=Concurrent -benchtime=5s

# Clean results
clean:
	@echo "Cleaning benchmark results..."
	rm -rf results/*.txt results/*.prof
	rm -f /tmp/jailbreak_bench.test

# Help target
help:
	@echo "Jailbreak Classifier Benchmarks"
	@echo ""
	@echo "Available targets:"
	@echo "  make bench-quick       - Run quick benchmarks (3s each)"
	@echo "  make bench-full        - Run full benchmarks (10s each, save results)"
	@echo "  make bench-concurrent  - Run concurrency benchmarks only"
	@echo "  make bench-memory      - Run with CPU and memory profiling"
	@echo "  make bench-compare     - Compare with previous results"
	@echo ""
	@echo "Model-specific benchmarks:"
	@echo "  make bench-modernbert  - Benchmark ModernBERT only"
	@echo "  make bench-deberta     - Benchmark DeBERTa only"
	@echo "  make bench-unified     - Benchmark Unified classifier only"
	@echo ""
	@echo "Other targets:"
	@echo "  make bench-init        - Benchmark initialization only"
	@echo "  make bench-race        - Run with race detector"
	@echo "  make clean             - Clean benchmark results"
	@echo "  make help              - Show this help message"

