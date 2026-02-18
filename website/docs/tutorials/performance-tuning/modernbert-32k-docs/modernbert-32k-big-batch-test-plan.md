# Big Batch Test Plan (High Concurrency C=50+)

**Required**: NVIDIA A100 GPU (40GB+ VRAM)

---

## Overview

This test plan covers validation of ModernBERT-base-32k under high concurrency (big batch) scenarios. These tests cannot be completed with the current environment (NVIDIA L4 GPU, 23GB VRAM) due to memory fragmentation issues and require an A100 GPU with 40GB+ VRAM.

**Infrastructure Status**: Ready - All tools and test frameworks are prepared

---

## Test Requirements

### Hardware Requirements

- **GPU**: NVIDIA A100 (40GB+ VRAM) - **Required**
- **System RAM**: 64GB+ recommended
- **CUDA**: Version 12.0+
- **Driver**: Latest NVIDIA driver

### Software Requirements

- `benchmark_concurrent.rs` - Supports C=50, C=100 (currently fails due to memory)
- `benchmark_performance.rs` - Performance profiling tool
- Flash Attention 2 enabled
- All dependencies installed

---

## Test Cases

### 1. High Concurrency Testing (C=50, C=100)

#### 1.1 Low Context Length (1K-4K tokens)

| Context Length | Concurrency | Expected Success Rate | Expected Latency |
|----------------|-------------|----------------------|------------------|
| 1024 tokens    | C=50        | ≥ 90%                | < 2000ms (p95)   |
| 1024 tokens    | C=100       | ≥ 80%                | < 3000ms (p95)   |
| 4096 tokens    | C=50        | ≥ 80%                | < 15000ms (p95)  |
| 4096 tokens    | C=100       | ≥ 70%                | < 20000ms (p95)  |

**Test Steps:**
1. Run `benchmark_concurrent.rs` with C=50, C=100
2. Test with 1K, 4K tokens
3. Measure latency (mean, p50, p95, p99)
4. Track success/error rates
5. Document memory usage

**Deliverables:**
- Latency statistics for each concurrency level
- Success/error rates
- Memory usage profiles
- Throughput measurements

---

#### 1.2 Medium Context Length (8K tokens)

| Context Length | Concurrency | Expected Success Rate | Expected Latency |
|----------------|-------------|----------------------|------------------|
| 8192 tokens    | C=50        | ≥ 70%                | < 25000ms (p95)  |
| 8192 tokens    | C=100       | ≥ 60%                | < 35000ms (p95)  |

**Test Steps:**
1. Run `benchmark_concurrent.rs` with C=50, C=100
2. Test with 8K tokens
3. Measure latency (mean, p50, p95, p99)
4. Track success/error rates
5. Document memory usage and fragmentation

**Deliverables:**
- Latency statistics
- Success/error rates
- Memory fragmentation analysis
- Recommendations for production

---

### 2. Throughput Analysis

#### 2.1 Requests Per Second (RPS)

**Test Steps:**
1. Measure throughput at different concurrency levels
2. Calculate RPS for C=50, C=100
3. Compare with C=1, C=10 baseline
4. Document throughput scaling

**Deliverables:**
- RPS measurements
- Throughput scaling analysis
- Bottleneck identification

---

### 3. Memory Fragmentation Analysis

#### 3.1 Memory Management Under High Concurrency

**Test Steps:**
1. Monitor GPU memory usage during high concurrency tests
2. Track memory allocation/deallocation patterns
3. Identify memory fragmentation issues
4. Test memory cleanup strategies
5. Document recommendations

**Deliverables:**
- Memory usage profiles
- Fragmentation analysis
- Cleanup strategy recommendations

---

### 4. Latency Distribution Analysis

#### 4.1 P95/P99 Latency Under High Concurrency

**Test Steps:**
1. Measure latency distribution at C=50, C=100
2. Analyze p50, p95, p99 percentiles
3. Identify latency spikes
4. Document tail latency behavior

**Deliverables:**
- Latency distribution charts
- P95/P99 analysis
- Tail latency recommendations

---

### 5. Error Rate Analysis

#### 5.1 OOM Error Patterns

**Test Steps:**
1. Track OOM errors at different concurrency levels
2. Analyze error patterns
3. Identify failure modes
4. Document error recovery strategies

**Deliverables:**
- Error rate analysis
- Failure mode documentation
- Recovery strategy recommendations

---

## Expected Outcomes

### Success Criteria

1. **C=50**:
   - 1K tokens: ≥ 90% success rate
   - 4K tokens: ≥ 80% success rate
   - 8K tokens: ≥ 70% success rate

2. **C=100**:
   - 1K tokens: ≥ 80% success rate
   - 4K tokens: ≥ 70% success rate
   - 8K tokens: ≥ 60% success rate

3. **Latency**:
   - P95 latency within acceptable limits
   - No excessive tail latency

4. **Memory**:
   - Memory usage within GPU limits
   - No memory leaks
   - Acceptable fragmentation

---

## Infrastructure Readiness

### Tools Ready 

- `benchmark_concurrent.rs` - Supports C=50, C=100 (currently fails due to memory)
- `benchmark_performance.rs` - Performance profiling ready
- Flash Attention 2 enabled
- All dependencies installed

### How to Run Tests

1. **Ensure sufficient GPU memory** (A100 40GB+)

2. **Run benchmark with high concurrency**:
   ```bash
   cargo run --example benchmark_concurrent --release --features cuda,flash-attn
   ```

3. **Monitor memory usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

---

## Resource Estimates

### Time Estimates

- **High Concurrency Testing**: 6-8 hours
- **Throughput Analysis**: 2-3 hours
- **Memory Fragmentation Analysis**: 3-4 hours
- **Latency Distribution Analysis**: 2-3 hours
- **Error Rate Analysis**: 2-3 hours

**Total**: ~15-21 hours

### Resource Requirements

- **GPU**: A100 (40GB VRAM) - **Required**
- **System RAM**: 64GB+ recommended
- **Storage**: ~10GB for test data and results


---

## Deliverables

1. **Test Results Report**
   - Latency measurements (C=50, C=100)
   - Success/error rates
   - Throughput measurements
   - Memory usage profiles

2. **Performance Analysis**
   - Throughput scaling analysis
   - Memory fragmentation analysis
   - Latency distribution analysis
   - Error rate patterns

3. **Recommendations**
   - Production concurrency limits
   - Memory management strategies
   - Scaling recommendations
   - Error handling strategies

---

## Known Issues (From Current Testing)

### Memory Fragmentation

- **Issue**: GPU memory not released between tests
- **Impact**: Subsequent high-concurrency tests fail
- **Workaround**: Test with fresh GPU state or longer wait times
- **Solution**: Investigate memory cleanup strategies on A100

### OOM Errors at High Concurrency

- **Issue**: OOM errors at C=50+ for 8K+ tokens
- **Impact**: Cannot test high concurrency with current environment
- **Solution**: A100 with 40GB+ VRAM required

---

## References

- **Performance Validation**: [Performance Validation](./modernbert-32k-performance-validation.md)
- **Deployment Guide**: [Deployment Guide](./modernbert-32k-deployment-guide.md)
- **Benchmark Tool**: `candle-binding/examples/benchmark_concurrent.rs`
- **Performance Tool**: `candle-binding/examples/benchmark_performance.rs`

---
