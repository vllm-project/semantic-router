---
sidebar_position: 1
---

# ModernBERT-base-32k Performance Benchmark Results

This tutorial provides benchmark results and performance tuning guidance for ModernBERT-base-32k integration. Use these results to provision hardware and adjust workload expectations for your deployment.

## Overview

ModernBERT-base-32k extends the context window from 512 tokens (BERT-base) to 32,768 tokens, enabling processing of long documents and conversations. This guide presents empirical benchmark results from comprehensive testing across different context lengths and concurrency levels.

**Test Environment:**

- **GPU**: NVIDIA L4 (23GB VRAM)
- **Flash Attention 2**: Enabled
- **Model**: `llm-semantic-router/modernbert-base-32k`
- **Test Tool**: `candle-binding/examples/benchmark_concurrent.rs`

---

## Benchmark Results

### Single Request Latency (C=1)

| Context Length | Mean Latency | p50 Latency | p95 Latency | p99 Latency | Status |
|----------------|--------------|-------------|-------------|-------------|--------|
| 1,024 tokens   | 90.98ms      | 94.18ms     | 94.24ms     | 94.24ms     | Pass |
| 4,096 tokens   | 899.87ms     | 955.05ms    | 955.93ms    | 955.93ms    | Pass |
| 8,192 tokens   | 3,299.92ms   | 3,524.62ms  | 3,526.34ms  | 3,526.34ms  | Pass |

**Notes:**

- 1K tokens: Stable performance with mean ≈ p50
- 4K and 8K tokens: Stable performance with mean ≈ p50

### Concurrent Requests (C=10)

| Context Length | Mean Latency | p50 Latency | p95 Latency | Success Rate | Status |
|----------------|--------------|-------------|-------------|--------------|--------|
| 1,024 tokens   | 1,001.22ms   | 970.65ms    | 1,379.32ms  | 100%         | Pass |
| 4,096 tokens   | 9,323.45ms   | 9,389.28ms  | 10,349.11ms | 93%          | Partial |
| 8,192 tokens   | N/A          | N/A         | N/A         | 0%           | Fail |

**Notes:**

- 1K tokens: Excellent performance with 100% success rate
- 4K tokens: 93% success rate (7 OOM errors out of 100 requests)
- 8K tokens: Failed due to insufficient GPU memory

### High Concurrency (C=50, C=100)

All high concurrency tests (C=50+) failed due to hardware limitations. The current test environment (NVIDIA L4 GPU with 23GB VRAM) does not provide sufficient memory for high concurrency workloads with larger context lengths. Testing high concurrency (C=50+) requires a GPU with 40GB+ VRAM (e.g., NVIDIA A100) as documented in the [Big Batch Test Plan](./modernbert-32k-docs/modernbert-32k-big-batch-test-plan.md).

---

## Hardware Provisioning Guide

### Minimum Requirements

| Context Length | GPU VRAM | System RAM | Recommended GPU |
|----------------|----------|------------|-----------------|
| ≤ 1K tokens    | ≥ 5GB    | ≥ 16GB     | NVIDIA T4, L4   |
| ≤ 4K tokens    | ≥ 10GB   | ≥ 32GB     | NVIDIA L4, A10G |
| ≤ 8K tokens    | ≥ 23GB   | ≥ 32GB     | NVIDIA L4, A10G |
| 16K+ tokens    | ≥ 40GB   | ≥ 64GB     | NVIDIA A100     |

### Recommended Configuration

**For Production (1K-8K tokens):**

- **GPU**: NVIDIA L4 (23GB VRAM) or better
- **System RAM**: 32GB+
- **CUDA**: Version 12.0+
- **Flash Attention 2**: Enabled (provides 1.75x-11.9x speedup)

**For Long Context (16K-32K tokens):**

- **GPU**: NVIDIA A100 (40GB+ VRAM) - **Required**
- **System RAM**: 64GB+
- See [Long Context Test Plan](./modernbert-32k-docs/modernbert-32k-long-context-test-plan.md) for details

---

## Workload Expectations

### Concurrency Limits by Context Length

| Context Length | Max Concurrency | Expected Throughput | Notes |
|----------------|-----------------|---------------------|-------|
| 1,024 tokens   | C=10            | ~10 req/s           | Tested and reliable |
| 4,096 tokens   | C=10            | ~1 req/s            | 88% success rate |
| 8,192 tokens   | C=1             | ~0.3 req/s          | Only C=1 works reliably |
| 16,384+ tokens | C=1 (with chunking) | Variable      | Requires A100 or chunking |

### Latency Expectations

**Single Request (C=1):**

- 1K tokens: ~100ms (p50)
- 4K tokens: ~950ms (p50)
- 8K tokens: ~3,500ms (p50)

**Concurrent Requests (C=10):**

- 1K tokens: ~1,000ms (mean)
- 4K tokens: ~9,400ms (mean, 93% success)
- 8K tokens: Not supported (OOM)

### Memory Usage

| Context Length | GPU Memory per Request | Notes                        |
|----------------|------------------------|------------------------------|
| 512 tokens     | ~5MB                   | Very efficient               |
| 1K tokens      | ~11MB                  | Very efficient               |
| 4K tokens      | ~estimated             | Moderate                     |
| 8K tokens      | ~estimated             | High (requires 22GB+ VRAM)   |

---

## Configuration Guide

### Enabling ModernBERT-base-32k

To use ModernBERT-base-32k in your semantic router configuration:

```yaml
classifier:
  category_model:
    model_id: "models/mom-domain-classifier"
    use_modernbert: true  # Enable ModernBERT-base-32k
    threshold: 0.6
    use_cpu: false  # Use GPU for better performance
    category_mapping_path: "models/mom-domain-classifier/category_mapping.json"
  
  pii_model:
    model_id: "models/mom-pii-classifier"
    use_modernbert: true  # Enable ModernBERT-base-32k
    threshold: 0.7
    use_cpu: false
    pii_mapping_path: "models/mom-pii-classifier/pii_type_mapping.json"

prompt_guard:
  model_id: "models/mom-jailbreak-classifier"
  use_modernbert: true  # Enable ModernBERT-base-32k
  threshold: 0.7
  use_cpu: false
```

### Flash Attention 2

Flash Attention 2 provides significant performance improvements (1.75x-11.9x speedup). Ensure it's enabled when building:

```bash
cargo build --release --features cuda,flash-attn
```

---

## Performance Tuning Recommendations

### 1. Context Length Selection

Choose context length based on your use case:

- **Short queries (≤1K tokens)**: Best performance, supports C=10
- **Medium documents (1K-4K tokens)**: Good performance, supports C=10 with 88% success
- **Long documents (4K-8K tokens)**: Acceptable performance, only C=1 supported
- **Very long documents (8K+ tokens)**: Requires chunking or A100 GPU

### 2. Concurrency Tuning

**Start Conservative:**

1. Begin with C=1 for all context lengths
2. Gradually increase to C=10 for 1K-4K tokens
3. Monitor GPU memory and error rates
4. Reduce concurrency if OOM errors occur

**Production Settings:**

```yaml
# Recommended concurrency limits
concurrency_limits:
  1024: 10   # 1K tokens: C=10
  4096: 10   # 4K tokens: C=10 (monitor for OOM)
  8192: 1    # 8K tokens: C=1 only
```

### 3. Memory Management

- **Monitor GPU memory** using `nvidia-smi`
- **Ensure sufficient free memory** before processing large batches
- **Use chunking** for sequences > 8K tokens
- **Restart service periodically** if memory constraints occur after extended use

### 4. Device Selection

**Always prefer GPU:**

- GPU provides 45x speedup for 512 tokens
- Flash Attention 2 provides additional 1.75x-11.9x speedup
- CPU only suitable as fallback for very short sequences

---

## Running Benchmarks

### Prerequisites

1. **Install Rust and CUDA:**

   ```bash
   # Install Rust
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   
   # Install CUDA toolkit
   # See: https://developer.nvidia.com/cuda-downloads
   ```

2. **Build with Flash Attention 2:**

   ```bash
   cd candle-binding
   cargo build --example benchmark_concurrent --release --features cuda,flash-attn
   ```

### Running Concurrent Request Benchmark

```bash
# Run benchmark for 1K-8K tokens
cargo run --example benchmark_concurrent --release --features cuda,flash-attn
```

### Testing Long Context (16K-32K)

For testing 16K-32K tokens (requires A100 GPU):

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

2. **Run benchmark:**

   ```bash
   cargo run --example benchmark_concurrent --release --features cuda,flash-attn
   ```

3. **See**: [Long Context Test Plan](./modernbert-32k-docs/modernbert-32k-long-context-test-plan.md) for detailed test plan

---

## Troubleshooting

### Out of Memory (OOM) Errors

**Symptoms:**

- `CUDA_ERROR_OUT_OF_MEMORY` errors
- Requests failing at high concurrency

**Solutions:**

1. Reduce concurrency (C=10 → C=1)
2. Use chunking for sequences > 8K tokens
3. Increase wait time between requests
4. Use GPU with more VRAM (A100 40GB+)

### High Latency

**Symptoms:**

- Latency higher than expected
- p95/p99 latency spikes

**Solutions:**

1. Enable Flash Attention 2
2. Reduce concurrency
3. Use chunking for long sequences
4. Monitor GPU utilization

### Memory Constraints

**Symptoms:**

- Tests pass initially, then fail
- Memory not released between requests
- OOM errors after initial successful tests

**Solutions:**

1. **Upgrade hardware**: Use GPU with more VRAM (A100 40GB+ for high concurrency)
2. Add explicit memory cleanup between requests
3. Increase wait time between requests
4. Restart service periodically to clear memory
5. Use memory pool management if available
6. **Reduce concurrency**: Lower concurrency levels reduce memory pressure

---

## Key Findings

### What Works

- **1K-4K tokens**: Reliable with C=1 and C=10
- **8K tokens**: Reliable with C=1
- **Flash Attention 2**: 1.75x-11.9x speedup
- **Memory efficiency**: ~5-11MB per request

### Limitations

- **4K tokens**: C=10 has 88% success rate (12 OOM errors)
- **8K tokens**: C=10+ not supported (OOM)
- **16K+ tokens**: Cannot test with 23GB VRAM (requires A100)

### Future Work

- **16K-32K tokens**: Test plan ready, waiting for A100 environment (40GB+ VRAM)
- **High concurrency (C=50+)**: Test plan ready, waiting for A100 environment (40GB+ VRAM)

See:

- [Long Context Test Plan](./modernbert-32k-docs/modernbert-32k-long-context-test-plan.md)
- [Big Batch Test Plan](./modernbert-32k-docs/modernbert-32k-big-batch-test-plan.md)

---

## References

- **Benchmark Tool**: `candle-binding/examples/benchmark_concurrent.rs`
- **Performance Tool**: `candle-binding/examples/benchmark_performance.rs`
- **Full Results**: [Performance Validation](./modernbert-32k-docs/modernbert-32k-performance-validation.md)
- **Deployment Guide**: [Deployment Guide](./modernbert-32k-docs/modernbert-32k-deployment-guide.md)

---
