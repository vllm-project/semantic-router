//go:build !windows && cgo

package benchmarks

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
)

// BenchmarkCacheSearch_1000Entries benchmarks cache search with 1000 entries
func BenchmarkCacheSearch_1000Entries(b *testing.B) {
	// Initialize embedding models once
	if err := cache.InitEmbeddingModels(); err != nil {
		b.Fatalf("Failed to initialize embedding models: %v", err)
	}

	config := cache.BenchmarkConfig{
		CacheSize:         1000,
		ConcurrencyLevels: []int{1},
		RequestsPerLevel:  b.N,
		SimilarityThresh:  0.85,
		UseHNSW:           true,
		EmbeddingModel:    "qwen3",
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
	if err := cache.InitEmbeddingModels(); err != nil {
		b.Fatalf("Failed to initialize embedding models: %v", err)
	}

	config := cache.BenchmarkConfig{
		CacheSize:         10000,
		ConcurrencyLevels: []int{1},
		RequestsPerLevel:  b.N,
		SimilarityThresh:  0.85,
		UseHNSW:           true,
		EmbeddingModel:    "qwen3",
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
	if err := cache.InitEmbeddingModels(); err != nil {
		b.Fatalf("Failed to initialize embedding models: %v", err)
	}

	config := cache.BenchmarkConfig{
		CacheSize:         5000,
		ConcurrencyLevels: []int{1},
		RequestsPerLevel:  b.N,
		SimilarityThresh:  0.85,
		UseHNSW:           true,
		EmbeddingModel:    "qwen3",
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
	if err := cache.InitEmbeddingModels(); err != nil {
		b.Fatalf("Failed to initialize embedding models: %v", err)
	}

	config := cache.BenchmarkConfig{
		CacheSize:         1000, // Smaller for linear search
		ConcurrencyLevels: []int{1},
		RequestsPerLevel:  b.N,
		SimilarityThresh:  0.85,
		UseHNSW:           false,
		EmbeddingModel:    "qwen3",
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
	if err := cache.InitEmbeddingModels(); err != nil {
		b.Fatalf("Failed to initialize embedding models: %v", err)
	}

	config := cache.BenchmarkConfig{
		CacheSize:         5000,
		ConcurrencyLevels: []int{1},
		RequestsPerLevel:  b.N,
		SimilarityThresh:  0.85,
		UseHNSW:           true,
		EmbeddingModel:    "qwen3",
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
	if err := cache.InitEmbeddingModels(); err != nil {
		b.Fatalf("Failed to initialize embedding models: %v", err)
	}

	config := cache.BenchmarkConfig{
		CacheSize:         5000,
		ConcurrencyLevels: []int{10},
		RequestsPerLevel:  b.N,
		SimilarityThresh:  0.85,
		UseHNSW:           true,
		EmbeddingModel:    "qwen3",
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
	if err := cache.InitEmbeddingModels(); err != nil {
		b.Fatalf("Failed to initialize embedding models: %v", err)
	}

	config := cache.BenchmarkConfig{
		CacheSize:         5000,
		ConcurrencyLevels: []int{50},
		RequestsPerLevel:  b.N,
		SimilarityThresh:  0.85,
		UseHNSW:           true,
		EmbeddingModel:    "qwen3",
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
	if err := cache.InitEmbeddingModels(); err != nil {
		b.Fatalf("Failed to initialize embedding models: %v", err)
	}

	// High hit ratio scenario
	config := cache.BenchmarkConfig{
		CacheSize:         5000,
		ConcurrencyLevels: []int{10},
		RequestsPerLevel:  b.N,
		SimilarityThresh:  0.85,
		UseHNSW:           true,
		EmbeddingModel:    "qwen3",
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
