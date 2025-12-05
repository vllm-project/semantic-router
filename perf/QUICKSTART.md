# Performance Testing Quick Start Guide

This guide walks you through running performance tests for the first time.

## Prerequisites

- Go 1.24+
- Rust 1.90+
- HuggingFace CLI (`pip install huggingface_hub`)
- Make
- At least 10GB free disk space (for models)

## Step-by-Step Instructions

### Step 1: Download Models

```bash
make download-models
```

**What it does:**

- Downloads ML models needed for classification and embeddings
- Stores models in `models/` directory
- Takes 5-30 minutes depending on network speed

**Quick alternative (minimal models):**

```bash
CI_MINIMAL_MODELS=true make download-models
```

**Expected output:**

```
Downloading models...
✓ ModernBERT classification models downloaded
✓ Qwen3 embedding model downloaded
Models ready in models/
```

---

### Step 2: Build

```bash
make build
```

**What it does:**

- Compiles Rust library (candle-binding)
- Builds Go semantic router binary
- Creates `bin/router` executable

**Expected output:**

```
Building Rust library...
   Compiling candle-binding...
   Finished release [optimized] target(s)
Building router...
✓ Build complete: bin/router
```

**Troubleshooting:**

- If Rust fails: `make clean && make rust`
- If Go fails: `cd src/semantic-router && go mod tidy`

---

### Step 3: Run Benchmarks (Quick Mode)

```bash
make perf-bench-quick
```

**What it does:**

- Runs all component benchmarks with 3s benchtime (fast)
- Tests classification, decision engine, and cache
- Generates CPU and memory profiles
- Takes 3-5 minutes

**Expected output:**

```
Running performance benchmarks...
goos: linux
goarch: amd64

BenchmarkClassifyBatch_Size1-8           100  12345678 ns/op  234 B/op  5 allocs/op
BenchmarkClassifyBatch_Size10-8           50  23456789 ns/op  456 B/op 10 allocs/op
BenchmarkEvaluateDecisions_Single-8     5000    234567 ns/op   89 B/op  3 allocs/op
BenchmarkCacheSearch_1000Entries-8      1000   1234567 ns/op  123 B/op  4 allocs/op

PASS
ok      github.com/vllm-project/semantic-router/perf/benchmarks  45.678s
```

**Run specific benchmarks:**

```bash
make perf-bench-classification  # Classification only
make perf-bench-decision        # Decision engine only
make perf-bench-cache           # Cache only
```

---

### Step 4: View CPU Profile

```bash
make perf-profile-cpu
```

**What it does:**

- Opens pprof web interface at http://localhost:8080
- Shows CPU flame graph and call tree
- Identifies performance hot spots

**Expected behavior:**

1. Browser opens automatically
2. Shows interactive flame graph
3. Click on functions to drill down
4. View call graph, top functions, etc.

**Manual analysis:**

```bash
# Generate flame graph
go tool pprof -http=:8080 reports/cpu.prof

# View top CPU consumers
go tool pprof -top reports/cpu.prof

# Interactive mode
go tool pprof reports/cpu.prof
```

**Memory profile:**

```bash
make perf-profile-mem
# or manually:
go tool pprof -http=:8080 reports/mem.prof
```

---

### Step 5: Update Baseline (on main branch)

```bash
# IMPORTANT: Only run on main branch after verifying performance is good!
git checkout main
make perf-baseline-update
```

**What it does:**

- Runs comprehensive benchmarks (30s benchtime)
- Generates baseline JSON files
- Stores in `perf/testdata/baselines/`
- Takes 10-15 minutes

**Expected output:**

```
Running benchmarks to update baseline...
Running for 30s each...

Updating baselines...
✓ Baseline files updated successfully
  Git commit: abc123def
  Timestamp: 2025-12-04T10:00:00Z

Baselines saved to:
  perf/testdata/baselines/classification.json
  perf/testdata/baselines/decision.json
  perf/testdata/baselines/cache.json
```

**Commit baselines:**

```bash
git add perf/testdata/baselines/
git commit -m "chore: update performance baselines"
git push
```

---

## Additional Commands

### Compare Against Baseline

```bash
make perf-compare
```

Shows performance changes vs baseline with % differences.

### Run with Regression Check

```bash
make perf-check
```

Exits with error code 1 if regressions detected (useful in CI).

### Full Benchmarks (10s benchtime)

```bash
make perf-bench
```

More thorough than quick mode, takes 10-15 minutes.

### E2E Performance Tests

```bash
make perf-e2e
```

Runs full-stack load tests with Kubernetes (requires Kind cluster).

### Clean Artifacts

```bash
make perf-clean
```

Removes all profile and report files.

---

## Understanding Results

### Benchmark Output Format

```
BenchmarkName-8    N   ns/op    B/op   allocs/op
                  │    │        │      │
                  │    │        │      └─ Allocations per operation
                  │    │        └─ Bytes allocated per operation
                  │    └─ Nanoseconds per operation
                  └─ Number of iterations
```

### Good Performance Indicators

✅ **Classification (batch=1):** < 10ms (10,000,000 ns/op)
✅ **Classification (batch=10):** < 50ms (50,000,000 ns/op)
✅ **Decision Engine:** < 1ms (1,000,000 ns/op)
✅ **Cache Search (1K):** < 5ms (5,000,000 ns/op)
✅ **Low allocations:** < 10 allocs/op per request

### Profile Interpretation

In pprof web UI:

- **Red = hot** (most CPU time)
- **Focus on wide bars** (cumulative time)
- **Look for unexpected calls** (e.g., lots of allocations)
- **Check CGO overhead** (C.* functions)

---

## Troubleshooting

### Models not found

```bash
# Re-download models
make download-models

# Check models exist
ls -la models/
```

### Library path error

```bash
# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release

# Or use the Makefile (handles this automatically)
make perf-bench-quick
```

### Benchmarks fail

```bash
# Rebuild everything
make clean
make build

# Check config exists
ls config/testing/config.e2e.yaml
```

### High variance in results

- Ensure no other CPU-intensive processes running
- Run multiple times: `make perf-bench-quick && make perf-bench-quick`
- Use longer benchtime: `make perf-bench` (10s instead of 3s)

---

## Next Steps

1. **Set up CI**: Push your branch to enable performance testing on PRs
2. **Optimize**: Use profiles to identify and fix bottlenecks
3. **Track trends**: Compare results over time
4. **Add tests**: Create new benchmarks for your components

## Learn More

- [Full Performance Testing README](README.md)
- [Profiling Guide](../docs/performance/profiling.md) (when created)
- [Go Benchmarking](https://dave.cheney.net/2013/06/30/how-to-write-benchmarks-in-go)
- [pprof Guide](https://github.com/google/pprof/blob/master/doc/README.md)
