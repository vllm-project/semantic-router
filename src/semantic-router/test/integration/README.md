# Integration Tests

This directory contains integration tests for the vLLM Semantic Router. These tests validate the full stack integration including Go, CGO, Rust, and ML model inference.

## Overview

Integration tests in this directory test **real components working together**, not mocked units:

- ✅ Real ML models loaded from disk (Qwen3-Embedding-0.6B, embeddinggemma-300m)
- ✅ Real Rust inference engine via Candle framework
- ✅ Real CGO/FFI boundary between Go ↔ Rust
- ✅ Real GPU/CPU computation
- ✅ Real memory allocation across language boundaries

## Test Categories

### Embedding Infrastructure Tests

#### `embedding_concurrency_test.go`

Tests thread safety and concurrent embedding generation:

- **Basic Concurrency**: 10 goroutines × 10 operations
- **High Stress**: 50 goroutines × 20 operations
- **Concurrent Same Text**: Deterministic output validation
- **Concurrent Different Texts**: Uniqueness validation
- **Deadlock Detection**: Timeout-based detection
- **Goroutine Leak Detection**: Runtime goroutine counting
- **Mutex Contention**: Sequential vs concurrent performance
- **Mixed Operations**: Different text lengths and model types

#### `embedding_memory_test.go`

Tests memory behavior and leak detection:

- **Memory Leak Detection**: 1000 embeddings with GC validation
- **Memory Growth Rate**: Progressive allocation tracking
- **Baseline Memory Tracking**: Initial state capture
- **Memory Efficiency**: Per-embedding memory cost

#### `embedding_performance_test.go`

Benchmarks embedding model performance:

- **Single Embedding**: Individual operation latency
- **Batch Embedding**: Multiple embeddings
- **Text Length Variations**: Short, medium, long texts
- **Concurrent Performance**: Parallel operation throughput
- **Model Comparison**: Qwen3 vs Gemma performance

## Prerequisites

### Required Models

Download models before running tests:

```bash
# From repository root
make download-models-lora
```

This downloads:

- `models/Qwen3-Embedding-0.6B/` (~600MB)
- `models/embeddinggemma-300m/` (~300MB, gated - requires HF token)

### Environment Setup

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# For gated models like Gemma (optional)
export HF_TOKEN=your_huggingface_token
```

## Running Tests

### All Integration Tests

```bash
# From repository root
make test-embedding
```

### Specific Test Patterns

```bash
# Concurrency tests only
go test -tags=integration -v ./test/integration -run TestEmbedding_.*Concurrency

# Memory tests only
go test -tags=integration -v ./test/integration -run TestEmbedding_.*Memory

# Performance tests only
go test -tags=integration -v ./test/integration -run TestEmbedding_.*Performance
```

### Benchmarks

```bash
# All embedding benchmarks
make bench-embedding

# Specific benchmark
go test -tags=integration -v ./test/integration -run=^$ -bench=BenchmarkEmbedding_Single
```

### Short Mode (CI)

```bash
# Reduced iterations for faster CI runs
go test -tags=integration -v -short ./test/integration
```

## Build Tags

All integration tests use the `integration` build tag:

```go
//go:build integration
// +build integration
```

This allows:

- **Fast unit tests**: `go test ./...` (skips integration tests)
- **Separate integration runs**: `go test -tags=integration ./test/integration`

## Test Helpers

### `embedding_test_helpers.go`

Provides shared utilities:

- **`initEmbeddingModelsWithFallback()`**: Graceful model initialization
  - Primary: Loads both Qwen3 and Gemma
  - Fallback: Loads only Qwen3 if Gemma unavailable (gated model)
  - Singleton: Only initializes once per test run
  - Note: Tests using "gemma" model type will fail gracefully in fallback mode

- **`testModelsDir`**: Constant pointing to `../../../../models`

## CI/CD Integration

These tests run automatically in GitHub Actions when:

### Automatic Triggers

- **Pull Requests**: Automatically runs if PR modifies:
  - `candle-binding/**` (Rust inference engine)
  - `src/semantic-router/test/integration/**` (test code)
  - `src/semantic-router/pkg/classification/**` (embedding infrastructure)
  - `.github/workflows/integration-tests-embedding.yml` (workflow itself)

- **Push to main**: After PR merge
- **Nightly schedule**: Every day at 3:00 AM UTC
- **Manual trigger**: Via GitHub Actions "Run workflow" button

### For Contributors

**You don't need to do anything special!** If your PR touches embedding-related code, the tests will run automatically. No labels or maintainer approval required.

### CI Configuration

- Runs with `-short` flag (reduced iterations for speed)
- 20-minute timeout
- Downloads models automatically (~1GB)
- Uses Ubuntu latest with Go 1.24 and Rust 1.90

## Success Criteria (from Issue #715)

These tests address the integration test requirements:

- ✅ **Embedding model loading**: Validated via `initEmbeddingModelsWithFallback`
- ✅ **Embedding model performance**: 14+ benchmarks across text lengths, models
- ✅ **Embedding model memory usage**: Leak detection, growth tracking
- ✅ **Concurrent embedding requests**: 15+ concurrency tests
- ✅ **Test execution time**: < 5 minutes (with `-short` flag)
- ✅ **CI/CD integration**: GitHub Actions workflow with automatic path-based triggers

## Performance Expectations

### Concurrency Tests

- **100 operations** should complete in < 60s
- **No goroutine leaks** (growth < 10 goroutines)
- **No deadlocks** (timeout: 30s)

### Memory Tests

- **1000 embeddings** should show < 10MB growth after GC
- **Memory per embedding** should be consistent

### Benchmarks

- **Single embedding** (Qwen3): ~400-600ms
- **Batch throughput**: Scales with goroutines
- **Speedup ratio**: Concurrent vs sequential > 1.5x

## Common Issues

### Gated Model Access

If Gemma model download fails:

```
⚠️  Gemma unavailable, falling back to Qwen3-only mode
✓ Fallback successful: Qwen3-only mode
```

This is expected without HuggingFace token. Tests will run with only Qwen3 available. Tests that explicitly require Gemma will skip or handle errors gracefully.

### GPU Memory

Tests run on GPU if available, CPU otherwise. For consistent results:

```bash
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU
# or
export CUDA_VISIBLE_DEVICES=""  # Force CPU
```

### LD_LIBRARY_PATH

CGO requires Rust library to be in library path:

```bash
export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:$LD_LIBRARY_PATH
```

The Makefile handles this automatically.

## Contributing

When adding new integration tests:

1. **Add build tag**: `//go:build integration`
2. **Use package**: `package integration`
3. **Reuse helpers**: Use `initEmbeddingModelsWithFallback()`
4. **Support `-short`**: Reduce iterations when `testing.Short()` is true
5. **Document**: Add to this README

## Architecture

```
Go Tests (integration package)
     ↓
initEmbeddingModelsWithFallback()
     ↓
candle_binding.GetEmbeddingWithModelType()
     ↓
CGO → Rust FFI → Candle ML Framework
     ↓
Real Models (Qwen3, Gemma)
     ↓
GPU/CPU Computation
```

## Related Documentation

- [Issue #715](https://github.com/vllm-project/semantic-router/issues/715): Original integration test requirements
