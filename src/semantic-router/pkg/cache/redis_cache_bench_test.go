//go:build !windows && cgo

package cache

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// redisBenchmarkAddr resolves the Redis host/port for benchmarks, honoring
// REDIS_HOST / REDIS_PORT and falling back to localhost:6379 (the port
// `make start-redis` maps). Mirrors valkeyIntegrationAddr.
func redisBenchmarkAddr() (string, int) {
	host, port := "localhost", 6379
	if v := os.Getenv("REDIS_HOST"); v != "" {
		host = v
	}
	if v := os.Getenv("REDIS_PORT"); v != "" {
		if p, err := strconv.Atoi(v); err == nil {
			port = p
		}
	}
	return host, port
}

// benchCacheSizes returns the cache pre-population sizes to sweep. It defaults
// to a single small size to keep CI wall-clock low, and honors a comma-separated
// SR_BENCH_CACHE_SIZES override (e.g. "1000,10000,50000") for large-scale runs.
func benchCacheSizes() []int {
	raw := os.Getenv("SR_BENCH_CACHE_SIZES")
	if raw == "" {
		return []int{1000}
	}
	var sizes []int
	for _, part := range strings.Split(raw, ",") {
		if n, err := strconv.Atoi(strings.TrimSpace(part)); err == nil && n > 0 {
			sizes = append(sizes, n)
		}
	}
	if len(sizes) == 0 {
		return []int{1000}
	}
	return sizes
}

// setupRedisCacheBench builds a live RedisCache for benchmarking, mirroring
// setupValkeyCacheIntegration. It fails hard under SR_BENCHMARK_MODE=true (CI,
// where `make start-redis` guarantees a server) so a dead backend can no longer
// pass silently; it skips locally when no server is present.
func setupRedisCacheBench(b *testing.B) *RedisCache {
	b.Helper()

	failClosed := os.Getenv("SR_BENCHMARK_MODE") == "true"
	unavailable := func(format string, args ...any) {
		if failClosed {
			b.Fatalf(format, args...)
		}
		b.Skipf(format, args...)
	}

	if err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true); err != nil {
		unavailable("failed to initialize BERT model: %v", err)
	}

	host, port := redisBenchmarkAddr()

	redisConfig := &config.RedisConfig{}
	redisConfig.Connection.Host = host
	redisConfig.Connection.Port = port
	redisConfig.Connection.Database = 0
	redisConfig.Connection.Timeout = 5
	redisConfig.Index.Name = "bench_redis_idx"
	redisConfig.Index.Prefix = "bench:"
	redisConfig.Index.VectorField.Name = "embedding"
	redisConfig.Index.VectorField.Dimension = 384 // all-MiniLM-L6-v2
	redisConfig.Index.VectorField.MetricType = "COSINE"
	redisConfig.Index.IndexType = "HNSW"
	redisConfig.Index.Params.M = 16
	redisConfig.Index.Params.EfConstruction = 64
	redisConfig.Search.TopK = 1
	redisConfig.Development.DropIndexOnStartup = true
	redisConfig.Development.AutoCreateIndex = true

	cache, err := NewRedisCache(RedisCacheOptions{
		// Gate at raw cosine >= 0.8 to match the in-memory backend, so
		// BenchmarkCacheComparison compares equivalent hit paths. Redis reports
		// COSINE distance and maps it to similarity = (1 + cos) / 2
		// (distanceToSimilarity), so a raw cos >= 0.8 cutoff is similarity >= 0.9.
		SimilarityThreshold: 0.9,
		TTLSeconds:          300,
		Enabled:             true,
		Config:              redisConfig,
		EmbeddingModel:      "bert",
	})
	if err != nil {
		unavailable("redis server not available: %v", err)
	}
	if err := cache.CheckConnection(context.Background()); err != nil {
		unavailable("redis connection check failed: %v", err)
	}
	return cache
}

// BenchmarkRedisCache measures the FindSimilar hot path against a Redis-backed
// semantic cache pre-populated with CacheSize entries. It is referenced by
// `make benchmark-redis`, which had no matching benchmark until now (the target
// silently passed because `go test -bench=<nonexistent>` exits 0).
func BenchmarkRedisCache(b *testing.B) {
	const model = "bench-model"
	// Populate cost is O(size): each entry runs one BERT embedding plus a Redis
	// vector-index insert (~100-200ms combined on a loaded CPU host), so it
	// dominates wall-clock setup. The reported ns/op measures only FindSimilar
	// (the timer is reset after populate), and HNSW search latency grows ~log(N),
	// so 1000 entries is already representative. Default to a single small size
	// to keep `make benchmark-redis` fast; opt into larger scales for profiling
	// via SR_BENCH_CACHE_SIZES (comma-separated, e.g. "1000,10000,50000").
	cacheSizes := benchCacheSizes()

	for _, size := range cacheSizes {
		b.Run(fmt.Sprintf("CacheSize_%d", size), func(b *testing.B) {
			cache := setupRedisCacheBench(b)
			defer func() { _ = cache.Close() }()
			runCacheFindSimilarBench(b, cache, model, size)
		})
	}
}
