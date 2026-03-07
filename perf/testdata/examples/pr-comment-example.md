# Example GitHub PR Comment

This is what will automatically appear as a comment on your PR when performance tests run in CI.

---

## 🔥 Performance Benchmark Results

**Commit:** `816dbec26397` | **Branch:** `perf_test` | **Run:** [#1234](https://github.com/vllm-project/semantic-router/actions/runs/1234)

### Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| Total Benchmarks | 32 | 100% |
| ⚠️ Regressions | 1 | 3.1% |
| 🚀 Improvements | 8 | 25.0% |
| ➡️ No Change | 23 | 71.9% |

---

### 📊 Key Performance Changes

| Component | Metric | Baseline | Current | Change | Status |
|-----------|--------|----------|---------|--------|--------|
| **Classification** (batch=1) | P95 Latency | 10.50ms | 10.12ms | -3.62% | |
| **Classification** (batch=10) | Throughput | 19.10 qps | 19.52 qps | +2.20% | 🚀 |
| **Decision Engine** (complex) | P95 Latency | 0.46ms | 0.52ms | **+13.04%** | ⚠️ |
| **Decision Engine** (complex) | Throughput | 2189 qps | 1952 qps | **-10.83%** | ⚠️ |
| **Cache** (1K entries) | P95 Latency | 4.23ms | 4.15ms | -1.89% | |
| **Cache** (concurrency=50) | Throughput | 1267 qps | 1322 qps | +4.34% | 🚀 |

---

### ⚠️ Regressions Detected

**1 regression exceeds threshold (10%):**

#### `BenchmarkEvaluateDecisions_ComplexScenario`

- **Latency:** 0.46ms → 0.52ms (+13.04%) ⚠️
- **Throughput:** 2189 qps → 1952 qps (-10.83%) ⚠️
- **Threshold:** 10% (exceeded by 3.04%)

**Action Required:**

- Review complex decision evaluation logic
- Run `make perf-profile-cpu` locally to identify bottleneck
- Consider optimizing rule matching for multi-domain scenarios

---

### 🚀 Notable Improvements

1. **Cache Concurrency** (+4.34% throughput)
   - Better performance under high concurrent load
   - Improved from 1267 qps to 1322 qps

2. **Classification Latency** (-3.62% P95)
   - Single-text classification now faster
   - Reduced from 10.50ms to 10.12ms

3. **Request Processing** (-2.43%)
   - ExtProc handler optimization showing results

---

### 📁 Artifacts

- [Full Benchmark Results](https://github.com/vllm-project/semantic-router/actions/runs/1234/artifacts)
- [CPU Profile](https://github.com/vllm-project/semantic-router/actions/runs/1234/artifacts/cpu.prof)
- [Memory Profile](https://github.com/vllm-project/semantic-router/actions/runs/1234/artifacts/mem.prof)

---

### 💡 Next Steps

To investigate the regression locally:

```bash
# Run benchmarks with profiling
make perf-bench

# View CPU profile
make perf-profile-cpu

# Compare against baseline
make perf-compare
```

---

<details>
<summary>📋 View All Benchmark Results</summary>

| Benchmark | ns/op | Change | Status |
|-----------|-------|--------|--------|
| BenchmarkClassifyBatch_Size1 | 10,123,456 | -1.19% | |
| BenchmarkClassifyBatch_Size10 | 51,234,567 | -2.12% | 🚀 |
| BenchmarkClassifyBatch_Size50 | 212,345,678 | -1.54% | |
| BenchmarkClassifyBatch_Size100 | 410,234,567 | -0.51% | |
| BenchmarkClassifyCategory | 8,654,321 | -1.27% | |
| BenchmarkClassifyPII | 10,089,123 | -0.34% | |
| BenchmarkClassifyJailbreak | 9,823,456 | -0.54% | |
| BenchmarkCGOOverhead | 3,423,456 | -0.96% | |
| BenchmarkEvaluateDecisions_Single | 229,876 | -2.00% | 🚀 |
| BenchmarkEvaluateDecisions_Multiple | 342,123 | -1.03% | |
| BenchmarkEvaluateDecisions_WithKeywords | 265,432 | -0.92% | |
| BenchmarkEvaluateDecisions_Complex | 512,345 | **+12.16%** | ⚠️ |
| BenchmarkRuleEvaluation_AND | 195,432 | -1.68% | |
| BenchmarkRuleEvaluation_OR | 174,321 | -1.26% | |
| BenchmarkPrioritySelection | 286,789 | -0.77% | |
| BenchmarkCacheSearch_1000 | 3,389,012 | -1.96% | 🚀 |
| BenchmarkCacheSearch_10000 | 7,823,456 | -0.84% | |
| BenchmarkCacheSearch_HNSW | 2,312,345 | -1.42% | |
| BenchmarkCacheSearch_Linear | 5,623,456 | -0.98% | |
| BenchmarkCacheConcurrency_1 | 2,856,789 | -1.15% | |
| BenchmarkCacheConcurrency_10 | 1,212,345 | -1.80% | 🚀 |
| BenchmarkCacheConcurrency_50 | 756,234 | -4.16% | 🚀 |
| BenchmarkProcessRequest | 445,678 | -2.43% | 🚀 |
| BenchmarkProcessRequestBody | 671,234 | -1.13% | |
| BenchmarkHeaderProcessing | 231,234 | -1.42% | |
| BenchmarkFullRequestFlow | 878,901 | -1.26% | |

</details>

---

*Performance testing powered by [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) • Generated at 2025-12-04 16:30:00 UTC*
