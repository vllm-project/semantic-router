//go:build !windows && cgo

package benchmarks

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
)

const (
	cacheEmbeddingModelType  = "mmbert"
	cacheEmbeddingDeployment = "perf-cache-embedding"
)

var (
	cacheEmbeddingOnce     sync.Once
	cacheEmbeddingErr      error
	cacheEmbeddingOwner    *embedding.Set
	cacheEmbeddingProvider embedding.Provider
)

func cacheEmbeddingDevice() string {
	if useGPU := os.Getenv("USE_GPU"); useGPU == "true" || useGPU == "1" {
		return "cuda:0"
	}
	return "cpu"
}

func cacheEmbeddingConfig(spec config.ResolvedModelBinding) *config.RouterConfig {
	cfg := &config.RouterConfig{}
	cfg.EmbeddingConfig.ModelType = cacheEmbeddingModelType
	cfg.SemanticCache.Enabled = true
	cfg.SemanticCache.EmbeddingModel = cacheEmbeddingModelType
	cfg.GlobalModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: cacheEmbeddingDeployment, Contract: "embedding.v1", Adapter: cacheEmbeddingModelType}}
	cfg.ModelDeployments = map[string]config.ModelDeployment{cacheEmbeddingDeployment: spec.Deployment}
	return cfg
}

// Prepare the shared response-cache owner used by the router and retain it
// until TestMain cleanup, including failures after preparation.
func initCacheEmbeddingModels(b *testing.B) embedding.Provider {
	b.Helper()
	cacheEmbeddingOnce.Do(func() {
		spec := benchmarkModel(b, "embedding", "embedding.v1")
		cfg := cacheEmbeddingConfig(spec)
		set, err := modelruntime.PrepareOwnedResponseCacheEmbeddings(context.Background(), cfg, benchmarkRuntime)
		if err != nil {
			cacheEmbeddingErr = fmt.Errorf("prepare Vela embedding: %w", err)
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
		b.Fatal(cacheEmbeddingErr)
	}
	recordModelIdentity(b, "embedding")
	return cacheEmbeddingProvider
}

func BenchmarkCacheSearch_1000Entries(b *testing.B) {
	runCacheLookupBenchmark(b, cacheScenario{entries: 1000, workers: 1, hnsw: true, repeatPercent: 70})
}

func BenchmarkCacheSearch_10000Entries(b *testing.B) {
	runCacheLookupBenchmark(b, cacheScenario{entries: 10000, workers: 1, hnsw: true, repeatPercent: 70})
}

func BenchmarkCacheSearch_HNSW(b *testing.B) {
	runCacheLookupBenchmark(b, cacheScenario{entries: 5000, workers: 1, hnsw: true, repeatPercent: 70})
}

func BenchmarkCacheSearch_Linear(b *testing.B) {
	runCacheLookupBenchmark(b, cacheScenario{entries: 1000, workers: 1, hnsw: false, repeatPercent: 70})
}

func BenchmarkCacheConcurrency_1(b *testing.B) {
	runCacheLookupBenchmark(b, cacheScenario{entries: 5000, workers: 1, hnsw: true, repeatPercent: 70})
}

func BenchmarkCacheConcurrency_10(b *testing.B) {
	runCacheLookupBenchmark(b, cacheScenario{entries: 5000, workers: 10, hnsw: true, repeatPercent: 70})
}

func BenchmarkCacheConcurrency_50(b *testing.B) {
	runCacheLookupBenchmark(b, cacheScenario{entries: 5000, workers: 50, hnsw: true, repeatPercent: 70})
}

func BenchmarkCacheHitRate(b *testing.B) {
	runCacheLookupBenchmark(b, cacheScenario{entries: 5000, workers: 10, hnsw: true, repeatPercent: 90})
}
