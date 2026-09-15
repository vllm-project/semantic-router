//go:build !windows && cgo

package benchmarks

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
)

var (
	cacheEmbeddingOnce     sync.Once
	cacheEmbeddingErr      error
	cacheEmbeddingOwner    *embedding.Set
	cacheEmbeddingProvider embedding.Provider
)

// resolveCacheEmbeddingModelDir points at the repo-root models/ directory unless
// QWEN3_MODEL_PATH overrides it; `go test` runs with perf/benchmarks as the
// working directory (same resolution as initIntentClassifier).
func resolveCacheEmbeddingModelDir() string {
	if path := os.Getenv(embeddingModelPathEnv); path != "" {
		return path
	}
	wd, err := os.Getwd()
	if err != nil {
		return defaultEmbeddingModelDir
	}
	return filepath.Join(wd, "..", "..", defaultEmbeddingModelDir)
}

// initCacheEmbeddingModels prepares the owned Qwen3 embedding provider once per
// process and returns it for cache.BenchmarkConfig.EmbeddingProvider.
func initCacheEmbeddingModels(b *testing.B) embedding.Provider {
	b.Helper()
	cacheEmbeddingOnce.Do(func() {
		modelDir := resolveCacheEmbeddingModelDir()
		if _, statErr := os.Stat(modelDir); statErr != nil {
			cacheEmbeddingErr = fmt.Errorf("embedding model dir not found at %s: %w", modelDir, statErr)
			return
		}
		cfg := cacheEmbeddingConfig(modelDir)
		// Measure the same owned provider used by router generations. The legacy
		// process-global continuous-batching FFI has different lifecycle semantics.
		set, err := modelruntime.PrepareOwnedEmbeddings(context.Background(), cfg, native.New(nil))
		if err != nil {
			cacheEmbeddingErr = fmt.Errorf("failed to prepare embedding model from %s: %w", modelDir, err)
			return
		}
		provider, err := set.Default()
		if err != nil {
			cacheEmbeddingErr = errors.Join(err, set.Close())
			return
		}
		cacheEmbeddingOwner = set
		cacheEmbeddingProvider = provider
	})
	if cacheEmbeddingErr != nil {
		b.Fatalf("Failed to initialize embedding models: %v", cacheEmbeddingErr)
	}
	return cacheEmbeddingProvider
}

func TestMain(m *testing.M) {
	code := m.Run()
	code, err := closeCacheEmbeddingOwner(code, cacheEmbeddingOwner)
	if err != nil {
		fmt.Fprintf(os.Stderr, "close cache embedding owner: %v\n", err)
	}
	os.Exit(code)
}

// BenchmarkCacheSearch_1000Entries benchmarks cache search with 1000 entries
func BenchmarkCacheSearch_1000Entries(b *testing.B) {
	// Initialize embedding models once
	provider := initCacheEmbeddingModels(b)

	config := cache.BenchmarkConfig{
		CacheSize:         1000,
		ConcurrencyLevels: []int{1},
		RequestsPerLevel:  b.N,
		SimilarityThresh:  0.85,
		UseHNSW:           true,
		EmbeddingModel:    cacheEmbeddingModelType,
		EmbeddingProvider: provider,
		HitRatio:          0.7,
	}

	b.ResetTimer()
	b.ReportAllocs()

	results := cache.RunStandaloneBenchmark(context.Background(), config)

	if len(results) > 0 {
		result := results[0]
		b.ReportMetric(result.OverallP95, "p95_ms")
		b.ReportMetric(result.OverallP99, "p99_ms")
		b.ReportMetric(result.Throughput, "qps")
		b.ReportMetric(result.CacheHitRate*100, "hit_rate_%")
	}
}

// BenchmarkCacheSearch_10000Entries benchmarks cache search with 10,000 entries
func BenchmarkCacheSearch_10000Entries(b *testing.B) {
	provider := initCacheEmbeddingModels(b)

	config := cache.BenchmarkConfig{
		CacheSize:         10000,
		ConcurrencyLevels: []int{1},
		RequestsPerLevel:  b.N,
		SimilarityThresh:  0.85,
		UseHNSW:           true,
		EmbeddingModel:    cacheEmbeddingModelType,
		EmbeddingProvider: provider,
		HitRatio:          0.7,
	}

	b.ResetTimer()
	b.ReportAllocs()

	results := cache.RunStandaloneBenchmark(context.Background(), config)

	if len(results) > 0 {
		result := results[0]
		b.ReportMetric(result.OverallP95, "p95_ms")
		b.ReportMetric(result.OverallP99, "p99_ms")
		b.ReportMetric(result.Throughput, "qps")
		b.ReportMetric(result.CacheHitRate*100, "hit_rate_%")
	}
}

// BenchmarkCacheSearch_HNSW benchmarks HNSW index search
func BenchmarkCacheSearch_HNSW(b *testing.B) {
	provider := initCacheEmbeddingModels(b)

	config := cache.BenchmarkConfig{
		CacheSize:         5000,
		ConcurrencyLevels: []int{1},
		RequestsPerLevel:  b.N,
		SimilarityThresh:  0.85,
		UseHNSW:           true,
		EmbeddingModel:    cacheEmbeddingModelType,
		EmbeddingProvider: provider,
		HitRatio:          0.7,
	}

	b.ResetTimer()
	b.ReportAllocs()

	results := cache.RunStandaloneBenchmark(context.Background(), config)

	if len(results) > 0 {
		result := results[0]
		b.ReportMetric(result.SearchP95, "search_p95_ms")
		b.ReportMetric(result.EmbeddingP95, "embedding_p95_ms")
	}
}

// BenchmarkCacheSearch_Linear benchmarks linear search (no HNSW)
func BenchmarkCacheSearch_Linear(b *testing.B) {
	provider := initCacheEmbeddingModels(b)

	config := cache.BenchmarkConfig{
		CacheSize:         1000, // Smaller for linear search
		ConcurrencyLevels: []int{1},
		RequestsPerLevel:  b.N,
		SimilarityThresh:  0.85,
		UseHNSW:           false,
		EmbeddingModel:    cacheEmbeddingModelType,
		EmbeddingProvider: provider,
		HitRatio:          0.7,
	}

	b.ResetTimer()
	b.ReportAllocs()

	results := cache.RunStandaloneBenchmark(context.Background(), config)

	if len(results) > 0 {
		result := results[0]
		b.ReportMetric(result.SearchP95, "search_p95_ms")
		b.ReportMetric(result.EmbeddingP95, "embedding_p95_ms")
	}
}

// BenchmarkCacheConcurrency_1 benchmarks cache with concurrency level 1
func BenchmarkCacheConcurrency_1(b *testing.B) {
	provider := initCacheEmbeddingModels(b)

	config := cache.BenchmarkConfig{
		CacheSize:         5000,
		ConcurrencyLevels: []int{1},
		RequestsPerLevel:  b.N,
		SimilarityThresh:  0.85,
		UseHNSW:           true,
		EmbeddingModel:    cacheEmbeddingModelType,
		EmbeddingProvider: provider,
		HitRatio:          0.7,
	}

	b.ResetTimer()
	b.ReportAllocs()

	results := cache.RunStandaloneBenchmark(context.Background(), config)

	if len(results) > 0 {
		result := results[0]
		b.ReportMetric(result.Throughput, "qps")
	}
}

// BenchmarkCacheConcurrency_10 benchmarks cache with concurrency level 10
func BenchmarkCacheConcurrency_10(b *testing.B) {
	provider := initCacheEmbeddingModels(b)

	config := cache.BenchmarkConfig{
		CacheSize:         5000,
		ConcurrencyLevels: []int{10},
		RequestsPerLevel:  b.N,
		SimilarityThresh:  0.85,
		UseHNSW:           true,
		EmbeddingModel:    cacheEmbeddingModelType,
		EmbeddingProvider: provider,
		HitRatio:          0.7,
	}

	b.ResetTimer()
	b.ReportAllocs()

	results := cache.RunStandaloneBenchmark(context.Background(), config)

	if len(results) > 0 {
		result := results[0]
		b.ReportMetric(result.Throughput, "qps")
	}
}

// BenchmarkCacheConcurrency_50 benchmarks cache with concurrency level 50
func BenchmarkCacheConcurrency_50(b *testing.B) {
	provider := initCacheEmbeddingModels(b)

	config := cache.BenchmarkConfig{
		CacheSize:         5000,
		ConcurrencyLevels: []int{50},
		RequestsPerLevel:  b.N,
		SimilarityThresh:  0.85,
		UseHNSW:           true,
		EmbeddingModel:    cacheEmbeddingModelType,
		EmbeddingProvider: provider,
		HitRatio:          0.7,
	}

	b.ResetTimer()
	b.ReportAllocs()

	results := cache.RunStandaloneBenchmark(context.Background(), config)

	if len(results) > 0 {
		result := results[0]
		b.ReportMetric(result.Throughput, "qps")
		b.ReportMetric(result.CacheHitRate*100, "hit_rate_%")
	}
}

// BenchmarkCacheHitRate benchmarks cache hit rate effectiveness
func BenchmarkCacheHitRate(b *testing.B) {
	provider := initCacheEmbeddingModels(b)

	// High hit ratio scenario
	config := cache.BenchmarkConfig{
		CacheSize:         5000,
		ConcurrencyLevels: []int{10},
		RequestsPerLevel:  b.N,
		SimilarityThresh:  0.85,
		UseHNSW:           true,
		EmbeddingModel:    cacheEmbeddingModelType,
		EmbeddingProvider: provider,
		HitRatio:          0.9, // 90% expected hit rate
	}

	b.ResetTimer()
	b.ReportAllocs()

	results := cache.RunStandaloneBenchmark(context.Background(), config)

	if len(results) > 0 {
		result := results[0]
		b.ReportMetric(result.CacheHitRate*100, "hit_rate_%")
		b.ReportMetric(result.OverallP95, "p95_ms")
	}
}
