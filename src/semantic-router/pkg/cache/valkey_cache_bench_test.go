//go:build !windows && cgo

package cache

import (
	"fmt"
	"os"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// setupValkeyCacheBench builds a live ValkeyCache for benchmarking, mirroring
// setupRedisCacheBench. It fails hard under SR_BENCHMARK_MODE=true (CI, where
// `make start-valkey` guarantees a server) so a dead backend can no longer pass
// silently; it skips locally when no server is present. Host/port come from
// valkeyIntegrationAddr (VALKEY_HOST / VALKEY_PORT, default localhost:6379).
func setupValkeyCacheBench(b *testing.B) *ValkeyCache {
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

	host, port := valkeyIntegrationAddr()

	valkeyConfig := &config.ValkeyConfig{}
	valkeyConfig.Connection.Host = host
	valkeyConfig.Connection.Port = port
	valkeyConfig.Connection.Database = 0
	valkeyConfig.Connection.Timeout = 5
	valkeyConfig.Index.Name = "bench_valkey_idx"
	valkeyConfig.Index.Prefix = "bench:"
	valkeyConfig.Index.VectorField.Name = "embedding"
	valkeyConfig.Index.VectorField.Dimension = 384 // all-MiniLM-L6-v2
	valkeyConfig.Index.VectorField.MetricType = "COSINE"
	valkeyConfig.Index.IndexType = "HNSW"
	valkeyConfig.Index.Params.M = 16
	valkeyConfig.Index.Params.EfConstruction = 64
	valkeyConfig.Search.TopK = 1
	valkeyConfig.Development.DropIndexOnStartup = true
	valkeyConfig.Development.AutoCreateIndex = true

	cache, err := NewValkeyCache(ValkeyCacheOptions{
		SimilarityThreshold: 0.8,
		TTLSeconds:          300,
		Enabled:             true,
		Config:              valkeyConfig,
		EmbeddingModel:      "bert",
	})
	if err != nil {
		unavailable("valkey server not available: %v", err)
	}
	if err := cache.CheckConnection(); err != nil {
		unavailable("valkey connection check failed: %v", err)
	}
	return cache
}

// BenchmarkValkeyCache measures the FindSimilar hot path against a Valkey-backed
// semantic cache pre-populated with CacheSize entries. It is referenced by
// `make benchmark-valkey`, which had no matching benchmark until now (the target
// silently passed because `go test -bench=<nonexistent>` exits 0). Sizes are
// controlled by benchCacheSizes (SR_BENCH_CACHE_SIZES).
func BenchmarkValkeyCache(b *testing.B) {
	const model = "bench-model"

	for _, size := range benchCacheSizes() {
		b.Run(fmt.Sprintf("CacheSize_%d", size), func(b *testing.B) {
			cache := setupValkeyCacheBench(b)
			defer func() { _ = cache.Close() }()
			runCacheFindSimilarBench(b, cache, model, size)
		})
	}
}
