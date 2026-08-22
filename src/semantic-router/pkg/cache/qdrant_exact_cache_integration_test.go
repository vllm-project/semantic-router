//go:build !windows && cgo

package cache

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestQdrantExactCacheIntegrationRoundTripAndPartitionIsolation(t *testing.T) {
	if os.Getenv("SKIP_QDRANT_TESTS") == "true" {
		t.Skip("Qdrant integration tests disabled")
	}
	host := os.Getenv("QDRANT_HOST")
	if host == "" {
		host = "localhost"
	}
	port := 6334
	if configured := os.Getenv("QDRANT_PORT"); configured != "" {
		parsed, err := strconv.Atoi(configured)
		require.NoError(t, err)
		port = parsed
	}
	cache, err := NewQdrantCache(QdrantCacheOptions{
		Enabled:        true,
		TTLSeconds:     60,
		EmbeddingModel: "bert",
		Config: &config.QdrantConfig{
			Host:           host,
			Port:           port,
			ConnectTimeout: 5,
			CollectionName: fmt.Sprintf(
				"exact_cache_%d",
				time.Now().UnixNano(),
			),
		},
	})
	if err != nil {
		t.Skipf("Qdrant unavailable: %v", err)
	}
	t.Cleanup(func() {
		_ = cache.client.DeleteCollection(context.Background(), cache.collectionName)
		_ = cache.Close()
	})

	fingerprint := fmt.Sprintf("exact-%d", time.Now().UnixNano())
	require.NoError(
		t,
		cache.AddExact(context.Background(), "tenant-a", fingerprint, []byte(`{"answer":"cached"}`), 60),
	)
	hit, err := cache.FindExact(context.Background(), "tenant-a", fingerprint)
	require.NoError(t, err)
	require.True(t, hit.Found)
	assert.JSONEq(t, `{"answer":"cached"}`, string(hit.ResponseBody))

	miss, err := cache.FindExact(context.Background(), "tenant-b", fingerprint)
	require.NoError(t, err)
	assert.False(t, miss.Found)
}
