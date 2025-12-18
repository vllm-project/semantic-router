# Performance Testing Report Examples

This directory contains example outputs showing what you'll see when running performance tests.

## 📁 Files in This Directory

### 1. **benchmark-output-example.txt**

Raw benchmark output from `make perf-bench-quick`

**Shows:**

- ns/op (nanoseconds per operation)
- Throughput (operations per second)
- Memory allocations
- P50/P90/P95/P99 latencies
- Cache hit rates

**Example line:**

```
BenchmarkClassifyBatch_Size1-8    100  10245678 ns/op  10.25 ms/op  2456 B/op  45 allocs/op
```

---

### 2. **comparison-example.txt**

Baseline comparison output from `make perf-compare`

**Shows:**

- Benchmark vs baseline comparison
- Percentage changes
- Regression detection
- Performance improvements
- Actionable recommendations

**Example:**

```
⚠️  BenchmarkEvaluateDecisions_Complex: +12.16% (threshold: 10%)
    - P95 latency increased by 13.04%
    - ACTION REQUIRED: Investigate
```

---

### 3. **example-report.json**

Machine-readable JSON report

**Use for:**

- CI/CD automation
- Programmatic analysis
- Data visualization
- Trend tracking

**Structure:**

```json
{
  "metadata": {...},
  "comparisons": [...],
  "has_regressions": true,
  "summary": {...}
}
```

---

### 4. **example-report.md**

Human-readable Markdown report

**Use for:**

- Documentation
- Sharing results
- GitHub issues
- Performance reviews

**Includes:**

- Executive summary
- Detailed comparison tables
- Analysis and recommendations
- Trend insights

---

### 5. **example-report.html**

Beautiful HTML report with styling

**Features:**

- Professional design
- Color-coded metrics
- Interactive elements (when fully implemented)
- Visual summary cards
- Detailed tables

**Open in browser:**

```bash
open perf/testdata/examples/example-report.html
```

---

### 6. **pr-comment-example.md**

GitHub PR comment format

**Shows:**

- What appears on your PRs automatically
- Summary table
- Key changes highlighted
- Regression warnings
- Expandable full results

**Triggered by:** CI workflow on PR

---

### 7. **pprof-example.txt**

CPU profiling output and interpretation

**Shows:**

- Top CPU consuming functions
- Flame graph visualization
- Memory allocation patterns
- Optimization opportunities
- Hot spot analysis

**View interactively:**

```bash
make perf-profile-cpu  # Opens browser at localhost:8080
```

---

## 🚀 Quick Examples

### Scenario 1: Everything is Good ✅

```
Summary:
  Total Benchmarks: 32
  Regressions: 0
  Improvements: 5
  No Change: 27

✓ No regressions detected
```

### Scenario 2: Regression Detected ⚠️

```
⚠️  WARNING: 1 regression(s) detected!

BenchmarkEvaluateDecisions_Complex: +12.16%
  - P95 latency: 0.46ms → 0.52ms (+13.04%)
  - Throughput: 2189 qps → 1952 qps (-10.83%)
  - BLOCKS PR (exceeds 10% threshold)
```

### Scenario 3: Great Improvements 🚀

```
Significant Improvements:
  1. Cache Concurrency: +4.34% throughput
  2. Classification: -3.62% P95 latency
  3. Request Processing: -2.43% overall
```

---

## 📊 Understanding the Reports

### Performance Metrics Glossary

| Metric | Description | Good Value |
|--------|-------------|------------|
| **ns/op** | Nanoseconds per operation | Lower is better |
| **P50** | 50th percentile latency | < threshold |
| **P95** | 95th percentile latency | Most important metric |
| **P99** | 99th percentile latency | Worst-case performance |
| **QPS** | Queries per second | Higher is better |
| **allocs/op** | Allocations per operation | Lower is better |
| **B/op** | Bytes allocated per operation | Lower is better |

### Status Indicators

- ✅ **OK**: Within acceptable range
- 🚀 **IMPROVED**: Significant improvement (> 5%)
- ⚠️ **REGRESSION**: Performance degraded beyond threshold
- ➡️ **NO CHANGE**: Minimal difference (< 1%)

### Change Interpretation

| Change | Meaning |
|--------|---------|
| -10% ns/op | 10% faster (good) |
| +10% ns/op | 10% slower (bad) |
| +10% QPS | 10% more throughput (good) |
| -10% QPS | 10% less throughput (bad) |

---

## 🎯 How to Use These Examples

### For New Users

1. Read `benchmark-output-example.txt` to understand raw output
2. Check `comparison-example.txt` to see regression detection
3. View `example-report.html` in browser for full experience

### For CI Integration

1. Reference `pr-comment-example.md` for expected PR comments
2. Use `example-report.json` structure for automation
3. Set up thresholds based on example values

### For Performance Optimization

1. Study `pprof-example.txt` for profiling insights
2. Focus on functions > 5% CPU time
3. Reduce allocations in hot paths
4. Run `make perf-profile-cpu` for your code

---

## 🔍 Real vs Example Data

**Note:** These examples use realistic but fictional data. Your actual results will vary based on:

- Hardware (CPU, memory)
- Model sizes
- Batch sizes
- Concurrency levels
- Code changes

**To generate real reports:**

```bash
# Run benchmarks
make perf-bench-quick

# Compare with baseline
make perf-compare

# Generate reports
make perf-report
```

---

## 📚 Learn More

- [Performance Testing README](../../README.md)
- [Quick Start Guide](../../QUICKSTART.md)
- [Configuration Reference](../../config/thresholds.yaml)
- [Makefile Targets](../../../tools/make/performance.mk)

---

*Examples created to help you understand performance testing outputs before running actual tests.*
