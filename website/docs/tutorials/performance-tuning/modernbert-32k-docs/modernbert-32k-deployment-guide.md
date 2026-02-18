# End User Deployment Guide

**Project**: Issue #995 - ModernBERT-base-32k Integration  
**Based on**: Performance & Functionality Validation Results (1K-8K tokens)

---

## Overview

This guide provides deployment recommendations for ModernBERT-base-32k based on empirical testing results. All recommendations are based on validated test results for context lengths from 512 tokens to 8K tokens.

**For long context (16K-32K) and big batch testing, see separate test plans:**

- [Long Context Test Plan](./modernbert-32k-long-context-test-plan.md)
- [Big Batch Test Plan](./modernbert-32k-big-batch-test-plan.md)

---

## 1. Performance Recommendations

### 1.1 Latency Expectations

| Context Length | GPU Latency (C=1) | GPU Latency (C=10)       | CPU Latency |
|----------------|-------------------|--------------------------|-------------|
| 512 tokens     | 163ms             | N/A                      | 7367ms      |
| 1K tokens      | 785ms             | 996ms                    | 806ms       |
| 4K tokens      | 896ms             | 9066ms (88% success)      | N/A         |
| 8K tokens      | 3294ms            | N/A (fails)              | N/A         |

**Recommendations:**

- Use GPU for all production deployments (45x faster for 512 tokens)
- Flash Attention 2 provides 1.75x-11.9x speedup (highly recommended)
- CPU only suitable for 512 tokens (similar performance to GPU for 1K+)

### 1.2 Memory Requirements

| Context Length | GPU Memory per Request | Total Memory (C=10)  |
|----------------|------------------------|----------------------|
| 512 tokens     | ~5MB                   | ~50MB                |
| 1K tokens      | ~11MB                  | ~110MB               |
| 4K tokens      | ~estimated             | ~estimated           |
| 8K tokens      | ~estimated             | ~estimated           |

**Recommendations:**

- Memory usage is very efficient (~5-11MB per request)
- For 8K tokens with C=1, ensure at least 2GB free GPU memory
- For 4K tokens with C=10, ensure at least 1GB free GPU memory

---

## 2. Concurrency Limits

### 2.1 Recommended Concurrency by Context Length

| Context Length | Max Concurrency      | Notes                              |
|----------------|----------------------|------------------------------------|
| 1024 tokens    | C=10                 | Tested and works reliably           |
| 4096 tokens    | C=10                 | 88% success rate (12 OOM errors)   |
| 8192 tokens    | C=1                  | Only C=1 works reliably            |
| 16384+ tokens  | C=1 (with chunking)   | Requires A100 or chunking          |

### 2.2 Concurrency Best Practices

1. **Start with C=1** for new deployments
2. **Gradually increase** to C=10 for 1K-4K tokens
3. **Monitor memory** - if OOM errors occur, reduce concurrency
4. **Use chunking** for sequences > 8K tokens
5. **Avoid C=50+** for 8K+ tokens (memory fragmentation issues)

---

## 3. Device Selection

### 3.1 GPU vs CPU Decision Matrix

| Context Length | Recommended Device | Reason                              |
|----------------|--------------------|-------------------------------------|
| 512 tokens     | GPU                | 45x faster than CPU                 |
| 1K tokens      | GPU                | Similar to CPU, but more scalable   |
| 4K tokens      | GPU                | CPU not tested, GPU recommended     |
| 8K tokens      | GPU                | CPU not tested, GPU recommended     |

### 3.2 Device Selection Heuristics

```rust
fn select_device(context_length: usize, available_gpu: bool) -> Device {
    if !available_gpu {
        return Device::Cpu;  // Fallback to CPU
    }
    
    // GPU recommended for all context lengths
    // Flash Attention 2 provides significant speedup
    Device::Cuda(0)
}
```

**Recommendations:**

- Always use GPU if available
- Flash Attention 2 provides 1.75x-11.9x speedup
- CPU only as fallback for 512 tokens

---

## 4. Chunking Strategy

### 4.1 When to Use Chunking

| Context Length | Chunking Required | Reason                                      |
|----------------|--------------------|---------------------------------------------|
| ≤ 8K tokens    | No                 | Can process in single pass                  |
| > 8K tokens    | Yes                | Memory limitations or latency optimization  |
| > 32K tokens   | Yes                | Model limit is 32K, must chunk              |

### 4.2 Chunking Threshold Recommendations

```rust
fn should_chunk(context_length: usize, concurrency: usize) -> bool {
    if context_length > 32768 {
        return true;  // Must chunk (model limit)
    }
    
    if context_length > 8192 && concurrency > 1 {
        return true;  // Chunk for 8K+ with concurrency
    }
    
    if context_length > 16384 {
        return true;  // Chunk for 16K+ (memory optimization)
    }
    
    false
}
```

### 4.3 Chunking Best Practices

1. **Overlap**: Use 10-20% overlap between chunks
2. **Size**: Keep chunks ≤ 8K tokens for optimal performance
3. **Aggregation**:
   - Domain classification: Average scores
   - PII detection: Union with deduplication
   - Jailbreak detection: Maximum score

---

## 5. Production Deployment Checklist

### 5.1 Pre-Deployment

- [ ] Verify GPU availability (NVIDIA GPU with CUDA support)
- [ ] Enable Flash Attention 2 (recommended)
- [ ] Set appropriate concurrency limits based on context length
- [ ] Configure chunking for sequences > 8K tokens
- [ ] Test with expected production load

### 5.2 Monitoring

- [ ] Monitor GPU memory usage
- [ ] Track latency (p50, p95, p99)
- [ ] Monitor error rates (OOM errors)
- [ ] Track concurrency levels
- [ ] Monitor accuracy (signal extraction)

### 5.3 Scaling

- [ ] Start with C=1 for new deployments
- [ ] Gradually increase to C=10 for 1K-4K tokens
- [ ] Monitor memory and error rates
- [ ] Adjust concurrency based on results
- [ ] Use chunking for sequences > 8K tokens

---

## 6. Troubleshooting

### 6.1 Common Issues

#### OOM (Out of Memory) Errors

**Symptoms:**

- `CUDA_ERROR_OUT_OF_MEMORY` errors
- Requests failing at high concurrency

**Solutions:**

1. Reduce concurrency (C=10 → C=1)
2. Use chunking for sequences > 8K tokens
3. Increase wait time between requests
4. Use GPU with more VRAM (A100 40GB+)

#### High Latency

**Symptoms:**

- Latency higher than expected
- p95/p99 latency spikes

**Solutions:**

1. Enable Flash Attention 2
2. Reduce concurrency
3. Use chunking for long sequences
4. Monitor GPU utilization

#### Memory Fragmentation

**Symptoms:**

- Tests pass initially, then fail
- Memory not released between requests

**Solutions:**

1. Add explicit memory cleanup
2. Increase wait time between requests
3. Restart service periodically
4. Use memory pool management

---

## 7. Best Practices

### 7.1 Performance

1. **Always use GPU** with Flash Attention 2
2. **Monitor concurrency** - don't exceed recommended limits
3. **Use chunking** for sequences > 8K tokens
4. **Cache results** when possible
5. **Batch requests** when appropriate

### 7.2 Reliability

1. **Start conservative** - C=1 for new deployments
2. **Gradually scale** - increase concurrency based on results
3. **Monitor errors** - track OOM and timeout errors
4. **Have fallbacks** - CPU mode for critical paths
5. **Test thoroughly** - validate with production-like load

### 7.3 Resource Management

1. **Monitor memory** - track GPU memory usage
2. **Set limits** - enforce concurrency limits
3. **Clean up** - explicit memory cleanup between requests
4. **Use timeouts** - prevent hanging requests
5. **Scale horizontally** - add more GPUs if needed

---

## 8. Resource Requirements

### 8.1 Minimum Requirements

- **GPU**: NVIDIA GPU with CUDA support (≥23GB VRAM for 8K tokens)
- **Memory**: ~5-11MB per request
- **CPU**: Multi-core recommended for preprocessing
- **Storage**: Model files (~500MB)

### 8.2 Recommended Requirements

- **GPU**: NVIDIA L4 or better (23GB+ VRAM)
- **Memory**: 32GB+ system RAM
- **CPU**: 8+ cores
- **Storage**: SSD for model files

### 8.3 For Long Context (16K-32K)

- **GPU**: NVIDIA A100 (40GB+ VRAM) - **Required**
- **Memory**: 64GB+ system RAM
- **See**: [Long Context Test Plan](./modernbert-32k-long-context-test-plan.md)

---

## 9. References

- **Performance Validation**: [Performance Validation](./modernbert-32k-performance-validation.md)
- **Long Context Test Plan**: [Long Context Test Plan](./modernbert-32k-long-context-test-plan.md)
- **Big Batch Test Plan**: [Big Batch Test Plan](./modernbert-32k-big-batch-test-plan.md)
- **Benchmark Results**: `BENCHMARK_RESULTS_ANALYSIS.md`

---
