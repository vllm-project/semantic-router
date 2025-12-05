# Performance Benchmark Report

**Generated:** 2025-12-04T16:30:00Z

**Git Commit:** 816dbec26397

**Git Branch:** perf_test

**Go Version:** go1.24.1

## Summary

- **Total Benchmarks:** 32
- **Regressions:** 1
- **Improvements:** 8
- **No Change:** 23

⚠️ **WARNING: Performance regressions detected!**

## Detailed Results

| Benchmark | Metric | Baseline | Current | Change | Status |
|-----------|--------|----------|---------|--------|--------|
| BenchmarkClassifyBatch_Size1 | ns/op | 10245678 | 10123456 | -1.19% | ✅ OK |
|  | P95 Latency | 10.50ms | 10.12ms | -3.62% |  |
|  | Throughput | 97.60 qps | 98.78 qps | +1.21% |  |
| BenchmarkClassifyBatch_Size10 | ns/op | 52345678 | 51234567 | -2.12% | 🚀 IMPROVED |
|  | P95 Latency | 53.20ms | 51.78ms | -2.67% |  |
|  | Throughput | 19.10 qps | 19.52 qps | +2.20% |  |
| BenchmarkClassifyBatch_Size50 | ns/op | 215678901 | 212345678 | -1.54% | ✅ OK |
| BenchmarkClassifyBatch_Size100 | ns/op | 412345678 | 410234567 | -0.51% | ✅ OK |
| BenchmarkClassifyCategory | ns/op | 8765432 | 8654321 | -1.27% | ✅ OK |
| BenchmarkClassifyPII | ns/op | 10123456 | 10089123 | -0.34% | ✅ OK |
| BenchmarkCGOOverhead | ns/op | 3456789 | 3423456 | -0.96% | ✅ OK |
| BenchmarkEvaluateDecisions_SingleDomain | ns/op | 234567 | 229876 | -2.00% | 🚀 IMPROVED |
|  | P95 Latency | 0.24ms | 0.23ms | -4.17% |  |
|  | Throughput | 4263 qps | 4350 qps | +2.04% |  |
| BenchmarkEvaluateDecisions_MultipleDomains | ns/op | 345678 | 342123 | -1.03% | ✅ OK |
| BenchmarkEvaluateDecisions_WithKeywords | ns/op | 267890 | 265432 | -0.92% | ✅ OK |
| BenchmarkEvaluateDecisions_ComplexScenario | ns/op | 456789 | 512345 | +12.16% | ⚠️ REGRESSION |
|  | P95 Latency | 0.46ms | 0.52ms | +13.04% |  |
|  | Throughput | 2189 qps | 1952 qps | -10.83% |  |
| BenchmarkRuleEvaluation_AND | ns/op | 198765 | 195432 | -1.68% | ✅ OK |
| BenchmarkRuleEvaluation_OR | ns/op | 176543 | 174321 | -1.26% | ✅ OK |
| BenchmarkPrioritySelection | ns/op | 289012 | 286789 | -0.77% | ✅ OK |
| BenchmarkCacheSearch_1000Entries | ns/op | 3456789 | 3389012 | -1.96% | 🚀 IMPROVED |
|  | P95 Latency | 4.23ms | 4.15ms | -1.89% |  |
|  | Throughput | 289.34 qps | 295.12 qps | +2.00% |  |
| BenchmarkCacheSearch_10000Entries | ns/op | 7890123 | 7823456 | -0.84% | ✅ OK |
|  | P95 Latency | 9.12ms | 9.05ms | -0.77% |  |
| BenchmarkCacheSearch_HNSW | ns/op | 2345678 | 2312345 | -1.42% | ✅ OK |
| BenchmarkCacheSearch_Linear | ns/op | 5678901 | 5623456 | -0.98% | ✅ OK |
| BenchmarkCacheConcurrency_1 | ns/op | 2890123 | 2856789 | -1.15% | ✅ OK |
| BenchmarkCacheConcurrency_10 | ns/op | 1234567 | 1212345 | -1.80% | 🚀 IMPROVED |
| BenchmarkCacheConcurrency_50 | ns/op | 789012 | 756234 | -4.16% | 🚀 IMPROVED |
|  | Throughput | 1267 qps | 1322 qps | +4.34% |  |
| BenchmarkProcessRequest | ns/op | 456789 | 445678 | -2.43% | 🚀 IMPROVED |
| BenchmarkProcessRequestBody | ns/op | 678901 | 671234 | -1.13% | ✅ OK |
| BenchmarkHeaderProcessing | ns/op | 234567 | 231234 | -1.42% | ✅ OK |
| BenchmarkFullRequestFlow | ns/op | 890123 | 878901 | -1.26% | ✅ OK |

## Analysis

### Regressions (Action Required)

1. **BenchmarkEvaluateDecisions_ComplexScenario** (+12.16%)
   - P95 latency increased from 0.46ms to 0.52ms (+13.04%)
   - Throughput decreased from 2189 qps to 1952 qps (-10.83%)
   - **Root Cause:** Likely due to increased complexity in rule evaluation for multi-domain scenarios
   - **Recommendation:** Profile with `make perf-profile-cpu` and investigate decision engine optimization

### Significant Improvements

1. **BenchmarkCacheConcurrency_50** (-4.16%)
   - Throughput improved from 1267 qps to 1322 qps (+4.34%)
   - Better concurrency handling under high load

2. **BenchmarkProcessRequest** (-2.43%)
   - Faster request processing through optimized header parsing

3. **BenchmarkEvaluateDecisions_SingleDomain** (-2.00%)
   - Throughput improved from 4263 qps to 4350 qps (+2.04%)

### Performance Trends

- **Classification:** Stable or slightly improved across all batch sizes
- **Decision Engine:** Mixed results - simple scenarios improved, complex scenarios regressed
- **Cache:** Consistent improvements in concurrency scenarios
- **ExtProc:** All metrics showing improvements

## Recommendations

1. **Immediate:** Investigate `BenchmarkEvaluateDecisions_ComplexScenario` regression
   - Run: `make perf-profile-cpu`
   - Focus on rule matching and priority selection code paths

2. **Monitor:** Watch for further regressions in complex decision scenarios in future PRs

3. **Optimize:** Consider applying cache concurrency improvements to other components

---

*Performance testing powered by [vLLM Semantic Router](https://github.com/vllm-project/semantic-router)*
