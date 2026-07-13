# Performance Testing

This directory contains the performance testing infrastructure for vLLM Semantic Router.

## Overview

The performance testing framework provides:

- **Component Benchmarks**: Fast Go benchmarks for individual components (classification, decision engine, cache)
- **E2E Performance Tests**: Full-stack load testing integrated with the e2e framework
- **Profiling**: pprof integration for CPU, memory, and goroutine profiling
- **Baseline Comparison**: Automated regression detection against performance baselines
- **CI/CD Integration**: Performance tests run on every PR with regression blocking

## Quick Start

### Running Benchmarks

```bash
# Run all benchmarks
make perf-bench

# Run quick benchmarks (faster iteration)
make perf-bench-quick

# Run specific component benchmarks
make perf-bench-classification
make perf-bench-decision
make perf-bench-cache
```

### Profiling

```bash
# Run benchmarks with profiling
make perf-bench

# Analyze CPU profile
go tool pprof -http=:8080 reports/cpu.prof

# Analyze memory profile
go tool pprof -http=:8080 reports/mem.prof

# Or use shortcuts
make perf-profile-cpu
make perf-profile-mem
```

### Baseline Comparison

```bash
# Compare against baseline (report only). Consumes reports/bench-output.txt, so
# run a benchmark target that tees there first (or use make perf-check).
make perf-compare

# Update baselines (run this on main branch after verifying improvements)
make perf-baseline-update
```

### Regression Detection

```bash
# Run benchmarks and fail if any benchmark regresses beyond its threshold
make perf-check
```

## Directory Structure

```
perf/
├── cmd/perftest/           # CLI tool for performance testing
├── pkg/
│   ├── benchmark/          # Benchmark orchestration and reporting
│   ├── profiler/           # pprof profiling utilities
│   └── metrics/            # Runtime metrics collection
├── benchmarks/             # Benchmark test files
│   ├── classification_bench_test.go
│   ├── decision_bench_test.go
│   ├── cache_bench_test.go
│   └── extproc_bench_test.go
├── config/                 # Configuration files
│   ├── perf.yaml          # Performance test configuration
│   └── thresholds.yaml    # Performance SLOs and thresholds
├── testdata/baselines/     # Performance baselines
└── scripts/                # Utility scripts
```

## Component Benchmarks

### Classification Benchmarks

Test classification performance with different batch sizes:

- `BenchmarkClassifyBatch_Size1` - Single text classification
- `BenchmarkClassifyBatch_Size10` - Batch of 10
- `BenchmarkClassifyBatch_Size50` - Batch of 50
- `BenchmarkClassifyBatch_Size100` - Batch of 100
- `BenchmarkClassifyCategory` - Category classification
- `BenchmarkClassifyPII` - PII detection
- `BenchmarkClassifyJailbreak` - Jailbreak detection

### Decision Engine Benchmarks

Test decision evaluation performance:

- `BenchmarkEvaluateDecisions_SingleDomain` - Single domain
- `BenchmarkEvaluateDecisions_MultipleDomains` - Multiple domains
- `BenchmarkEvaluateDecisions_WithKeywords` - With keyword matching
- `BenchmarkPrioritySelection` - Decision priority selection

### Cache Benchmarks

Test semantic cache performance (wraps existing cache benchmark tool):

- `BenchmarkCacheSearch_1000Entries` - Search in 1K entries
- `BenchmarkCacheSearch_10000Entries` - Search in 10K entries
- `BenchmarkCacheSearch_HNSW` - HNSW index performance
- `BenchmarkCacheSearch_Linear` - Linear search performance
- `BenchmarkCacheConcurrency_*` - Different concurrency levels

## Performance Metrics

### Tracked Metrics

**Latency**:

- P50, P90, P95, P99 percentiles
- Average and max latency

**Throughput**:

- Requests per second (QPS)
- Batch processing efficiency

**Resource Usage**:

- CPU usage (cores)
- Memory usage (MB)
- Goroutine count
- Heap allocations

**Component-Specific**:

- Classification: CGO call overhead
- Cache: Hit rate, HNSW vs linear speedup
- Decision: Rule matching time

### Performance Thresholds

Defined in `config/thresholds.yaml`:

| Component | Metric | Threshold |
|-----------|--------|-----------|
| Classification (batch=1) | P95 latency | < 10ms |
| Classification (batch=10) | P95 latency | < 50ms |
| Decision Engine | P95 latency | < 1ms |
| Cache (1K entries) | P95 latency | < 5ms |
| Cache | Hit rate | > 80% |

Regression thresholds are per-benchmark `ns/op` bounds (5% for the cheap,
deterministic decision engine; 25% for the noisier Looper micro-benchmarks; 10%
default). They are matched by benchmark name — see
[Thresholds Config](#thresholds-config-configthresholdsyaml).

## E2E Performance Tests

E2E tests measure full-stack performance:

```bash
# Run E2E performance tests
make perf-e2e
```

Test cases:

- `performance-throughput` - Sustained QPS measurement
- `performance-latency` - End-to-end latency distribution
- `performance-resource` - Resource utilization monitoring

## CI/CD Integration

### PR regression gate

`performance-test.yml` runs on every PR that touches the router, bindings, or
`perf/`:

1. **Run benchmarks** — component suites + the Looper family, tee'd to
   `reports/bench-output.txt`.
2. **Generate current results** — `perftest --parse-bench` turns that raw output
   into `reports/current.json`.
3. **Compare and gate** — `perftest --compare-baseline ... --fail-on-regression`
   diffs `current.json` against the committed per-suite baselines and **exits
   non-zero if any benchmark regresses beyond its threshold**, turning the check
   red on the PR.
4. **Comment on the PR** — a summary comment reports each suite's status and the
   regression-gate result.

### Baseline lifecycle

The gate compares **absolute `ns/op`**, which is machine-dependent, so the
committed baselines in `testdata/baselines/` must come from the **same
ubuntu-latest runner class** the PR gate runs on. The **Nightly Performance
Baseline** workflow (`performance-nightly.yml`) refreshes and commits them on
that runner.

> **Seeding (one-time):** the baselines currently in the repo were captured on
> developer machines. Trigger the Nightly workflow once (`workflow_dispatch`) to
> replace them with ubuntu-latest numbers before relying on the gate; otherwise
> the first PRs will see cross-machine noise. Suites whose baseline is empty
> (e.g. classification before models are cached) are simply skipped by the gate,
> never falsely failed.

For **local** runs, `make perf-check` compares against these committed baselines
too — refresh your own with `make perf-baseline-update` first, since your
hardware differs from CI.

### Scope

This gate establishes the single-runner numeric-regression mechanism. Large-scale
**cross-hardware / cross-backend** coverage (Candle/ONNX × NVIDIA × AMD,
backend-specific baselines) is tracked separately by **#1510**. The gap this work
closes was surfaced by the router-quality audit in **#2375** and filed as
**#2455**.

## Configuration

### Performance Test Config (`config/perf.yaml`)

```yaml
benchmark_config:
  classification:
    batch_sizes: [1, 10, 50, 100]
    iterations: 1000

  cache:
    cache_sizes: [1000, 10000]
    concurrency_levels: [1, 10, 50]
```

### Thresholds Config (`config/thresholds.yaml`)

Regression thresholds are matched **per benchmark** by name. Each entry maps a
Go-regexp pattern to the maximum tolerated `ns/op` regression (percent slower
than baseline); the **first matching pattern wins**, so entries are ordered
most-specific first, and any benchmark matching nothing uses `default`:

```yaml
component_benchmarks:
  default:
    max_regression_percent: 10
  benchmarks:
    - name: classification
      pattern: "^BenchmarkClassify"
      max_regression_percent: 15
    - name: decision_engine
      pattern: "^Benchmark(EvaluateDecisions|PrioritySelection|Rule)"
      max_regression_percent: 5
    - name: looper
      pattern: "^Benchmark(ReMoM|Fusion|Flow|Base)"
      max_regression_percent: 25
```

## Troubleshooting

### Benchmarks fail to run

Ensure the Rust library is built and in the library path:

```bash
make rust
export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release
```

### Models not found

Download models before running benchmarks:

```bash
make download-models
```

### High variance in results

- Increase `benchtime` for more stable results
- Run benchmarks multiple times and average
- Ensure no other CPU-intensive processes are running

### Memory profiling shows high allocations

Use the memory profile to identify hot spots:

```bash
go tool pprof -http=:8080 reports/mem.prof
```

Look for:

- String/slice allocations in classification
- CGO marshalling overhead
- Cache entry allocations

## Adding New Benchmarks

1. Create benchmark function in appropriate file:

```go
func BenchmarkMyFeature(b *testing.B) {
    // Setup
    setupMyFeature(b)

    b.ResetTimer()
    b.ReportAllocs()

    for i := 0; i < b.N; i++ {
        // Test code
    }
}
```

2. Update thresholds in `config/thresholds.yaml`

3. Run the benchmark:

```bash
cd perf
go test -bench=BenchmarkMyFeature -benchmem ./benchmarks/
```

4. Update baseline:

```bash
make perf-baseline-update
```

## Best Practices

1. **Always warm up** - Run warmup iterations before measuring
2. **Report allocations** - Use `b.ReportAllocs()` to track memory
3. **Reset timer** - Use `b.ResetTimer()` after setup
4. **Use realistic data** - Test with production-like inputs
5. **Control variance** - Use fixed seeds for random data
6. **Measure what matters** - Focus on user-facing metrics

## Resources

- [Go Benchmarking Guide](https://dave.cheney.net/2013/06/30/how-to-write-benchmarks-in-go)
- [pprof Documentation](https://github.com/google/pprof/blob/master/doc/README.md)
- [Performance Best Practices](https://go.dev/doc/effective_go#performance)
