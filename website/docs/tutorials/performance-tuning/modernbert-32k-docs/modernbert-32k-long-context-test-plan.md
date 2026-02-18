# Long Context Test Plan (16K-32K Tokens)

**Project**: Issue #995 - ModernBERT-base-32k Integration  
**Required**: NVIDIA A100 GPU (40GB+ VRAM)

---

## Overview

This test plan covers validation of ModernBERT-base-32k for long context sequences (16K-32K tokens). These tests cannot be completed with the current environment (NVIDIA L4 GPU, 23GB VRAM) and require an A100 GPU with 40GB+ VRAM.

**Infrastructure Status**: Ready - All tools and test frameworks are prepared

---

## Test Requirements

### Hardware Requirements

- **GPU**: NVIDIA A100 (40GB+ VRAM) - **Required**
- **System RAM**: 64GB+ recommended
- **CUDA**: Version 12.0+
- **Driver**: Latest NVIDIA driver

### Software Requirements

- `benchmark_concurrent.rs` - Supports 16K/32K (currently commented out)
- `benchmark_performance.rs` - Performance profiling tool
- Flash Attention 2 enabled
- All dependencies installed

---

## Test Cases

### 1. Basic Inference Testing

#### 1.1 Single Request Latency (C=1)

| Context Length | Expected Latency | Success Criteria |
|----------------|------------------|------------------|
| 16384 tokens   | < 10s            | Latency < 10s |
| 24576 tokens   | < 15s            | Latency < 15s |
| 32768 tokens   | < 20s            | Latency < 20s |

**Test Steps:**
1. Load ModernBERT-base-32k model
2. Create test sequences of 16K, 24K, 32K tokens
3. Measure inference latency for each
4. Verify no OOM errors
5. Document results

**Deliverables:**
- Latency measurements for each context length
- Memory usage profiles
- Success/failure status

---

### 2. Concurrent Request Testing

#### 2.1 Low Concurrency (C=1, C=10)

| Context Length | Concurrency | Expected Success Rate |
|----------------|-------------|----------------------|
| 16384 tokens   | C=1         | 100%                 |
| 16384 tokens   | C=10        | ≥ 80%                |
| 32768 tokens   | C=1         | 100%                 |
| 32768 tokens   | C=10        | ≥ 80%                |

**Test Steps:**
1. Run `benchmark_concurrent.rs` with 16K, 32K tokens
2. Test with C=1 and C=10
3. Measure latency (mean, p50, p95, p99)
4. Track success/error rates
5. Document memory usage

**Deliverables:**
- Latency statistics for each concurrency level
- Success/error rates
- Memory usage profiles

---

### 3. Performance Profiling

#### 3.1 Component Breakdown

**Test Steps:**
1. Run `benchmark_performance.rs` for 16K, 32K tokens
2. Measure:
   - Tokenization time
   - Tensor creation time
   - Forward pass time
   - Total latency
3. Compare with Flash Attention 2 enabled/disabled
4. Document performance breakdown

**Deliverables:**
- Performance breakdown by component
- Flash Attention 2 impact
- Bottleneck identification

---

### 4. Memory Profiling

#### 4.1 Memory Usage Analysis

**Test Steps:**
1. Measure GPU memory usage for each context length
2. Track memory allocation patterns
3. Identify memory peaks
4. Document memory requirements

**Deliverables:**
- Memory usage profiles
- Peak memory requirements
- Memory efficiency metrics

---

### 5. Accuracy Validation

#### 5.1 Signal Extraction Accuracy

**Test Steps:**
1. Test domain classification accuracy at 16K, 32K tokens
2. Test PII detection accuracy at 16K, 32K tokens
3. Test jailbreak detection accuracy at 16K, 32K tokens
4. Compare with baseline (512 tokens)
5. Document accuracy degradation (if any)

**Deliverables:**
- Accuracy measurements for each signal type
- Comparison with baseline
- Accuracy degradation analysis

---

### 6. Position Accuracy Testing

#### 6.1 Information Retrieval at Different Positions

**Test Steps:**
1. Place test information at beginning, middle, end of sequence
2. Test with 16K, 32K tokens
3. Measure retrieval accuracy for each position
4. Compare with model card baseline
5. Document position accuracy

**Deliverables:**
- Position accuracy results
- Comparison with baseline
- Recommendations

---

## Expected Outcomes

### Success Criteria

1. **16K tokens**: 
   - C=1 latency < 10s
   - C=10 success rate ≥ 80%
   - No OOM errors

2. **32K tokens**:
   - C=1 latency < 20s
   - C=10 success rate ≥ 80%
   - No OOM errors

3. **Accuracy**:
   - Signal extraction accuracy maintained (≥ 0.90 for domain, ≥ 0.85 for PII)
   - Position accuracy comparable to model card baseline

4. **Memory**:
   - Memory usage within GPU limits
   - No memory leaks

---

## Infrastructure Readiness

### Tools Ready 

- `benchmark_concurrent.rs` - Supports 16K/32K (uncomment test cases)
- `benchmark_performance.rs` - Performance profiling ready
- Flash Attention 2 enabled
- All dependencies installed

### How to Run Tests

1. **Uncomment test cases** in `benchmark_concurrent.rs`:
   ```rust
   let context_lengths = vec![
       1024,
       4096,
       8192,
       16384,  // Uncomment this
       32768,  // Uncomment this
   ];
   ```

2. **Run benchmark**:
   ```bash
   cargo run --example benchmark_concurrent --release --features cuda,flash-attn
   ```

3. **Run performance profiling**:
   ```bash
   cargo run --example benchmark_performance --release --features cuda,flash-attn
   ```

---

## Resource Estimates

### Time Estimates

- **Basic Inference Testing**: 2-3 hours
- **Concurrent Request Testing**: 4-6 hours
- **Performance Profiling**: 2-3 hours
- **Memory Profiling**: 1-2 hours
- **Accuracy Validation**: 3-4 hours
- **Position Accuracy Testing**: 2-3 hours

**Total**: ~14-21 hours

### Resource Requirements

- **GPU**: A100 (40GB VRAM) - **Required**
- **System RAM**: 64GB+ recommended
- **Storage**: ~10GB for test data and results

---

## Deliverables

1. **Test Results Report**
   - Latency measurements (16K, 32K tokens)
   - Concurrency results
   - Memory usage profiles
   - Accuracy measurements

2. **Performance Analysis**
   - Component breakdown
   - Flash Attention 2 impact
   - Bottleneck identification

3. **Recommendations**
   - Deployment recommendations for 16K-32K tokens
   - Chunking strategy recommendations
   - Resource requirements

---

## References

- **Performance Validation**: [Performance Validation](./modernbert-32k-performance-validation.md)
- **Deployment Guide**: [Deployment Guide](./modernbert-32k-deployment-guide.md)
- **Benchmark Tool**: `candle-binding/examples/benchmark_concurrent.rs`
- **Performance Tool**: `candle-binding/examples/benchmark_performance.rs`

---

