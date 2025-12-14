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
# Compare current performance against baseline
make perf-compare

# Update baselines (run this on main branch after verifying improvements)
make perf-baseline-update
```

### Regression Detection

```bash
# Run benchmarks and fail if regressions detected
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

Regression thresholds: 10-20% depending on component.

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

Performance tests run automatically on every PR:

1. **PR Opened** → Run component benchmarks (5 min)
2. **Compare Against Baseline** → Calculate % changes
3. **Post Results to PR** → Automatic comment with metrics table
4. **Block if Regression** → Fail CI if thresholds exceeded

Nightly jobs update baselines on the main branch.

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

```yaml
component_benchmarks:
  classification:
    batch_size_1:
      max_p95_latency_ms: 10.0
      max_regression_percent: 10
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
