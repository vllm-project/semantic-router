//go:build !windows && cgo && !riscv64

package cache

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// StorageIntegration: valkey
func TestValkeySemanticPolarityIntegration(t *testing.T) {
	storagetest.Require(t, "valkey")
	host, port := valkeyIntegrationAddr()
	cfg := &config.ValkeyConfig{}
	cfg.Connection.Host, cfg.Connection.Port, cfg.Connection.Timeout = host, port, 5
	cfg.Index.Name = fmt.Sprintf("polarity_valkey_%d", time.Now().UnixNano())
	cfg.Index.Prefix = cfg.Index.Name + ":"
	cfg.Index.VectorField.Name, cfg.Index.VectorField.Dimension = "embedding", 384
	cfg.Index.VectorField.MetricType, cfg.Index.IndexType = "COSINE", "HNSW"
	cfg.Index.Params.M, cfg.Index.Params.EfConstruction = 16, 64
	cfg.Search.TopK = 4
	cfg.Development.AutoCreateIndex = true
	backend, err := NewValkeyCache(ValkeyCacheOptions{
		Enabled: true, Config: cfg, TTLSeconds: 60, SimilarityThreshold: .8,
		EmbeddingModel: "bert", EmbeddingProvider: redisValkeyPolarityVectors(),
	})
	if err != nil {
		storagetest.Unavailable(t, "valkey", fmt.Sprintf("Valkey vector search unavailable: %v", err))
	}
	t.Cleanup(func() {
		_, _ = backend.client.CustomCommand(context.Background(), []string{"FT.DROPINDEX", cfg.Index.Name})
		_ = backend.Close()
	})
	// Only this test's unique index is dropped. Its document keys retain their
	// 60-second TTL; no shared database or unowned key is flushed.
	runRedisValkeyStoredPolarityCases(t, backend, ValkeyCacheType)
}
