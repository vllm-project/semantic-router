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

func TestMilvusExactCacheIntegrationRoundTripAndPartitionIsolation(t *testing.T) {
	if os.Getenv("SKIP_MILVUS_TESTS") == "true" {
		t.Skip("Milvus integration tests disabled")
	}
	host := os.Getenv("MILVUS_HOST")
	if host == "" {
		host = "localhost"
	}
	port := 19530
	if configured := os.Getenv("MILVUS_PORT"); configured != "" {
		parsed, err := strconv.Atoi(configured)
		require.NoError(t, err)
		port = parsed
	}
	milvusConfig := milvusExactTestConfig(host, port)

	cache, err := NewMilvusCache(MilvusCacheOptions{
		Enabled:        true,
		TTLSeconds:     60,
		EmbeddingModel: "bert",
		Config:         milvusConfig,
	})
	if err != nil {
		t.Skipf("Milvus unavailable: %v", err)
	}
	t.Cleanup(func() {
		_ = cache.client.DropCollection(context.Background(), cache.collectionName)
		_ = cache.Close()
	})

	fingerprint := fmt.Sprintf("exact-%d", time.Now().UnixNano())
	require.NoError(
		t,
		cache.AddExact("tenant-a", fingerprint, []byte(`{"answer":"cached"}`), 60),
	)
	hit, err := cache.FindExact("tenant-a", fingerprint)
	require.NoError(t, err)
	require.True(t, hit.Found)
	assert.JSONEq(t, `{"answer":"cached"}`, string(hit.ResponseBody))

	miss, err := cache.FindExact("tenant-b", fingerprint)
	require.NoError(t, err)
	assert.False(t, miss.Found)
}

func milvusExactTestConfig(host string, port int) *config.MilvusConfig {
	milvusConfig := &config.MilvusConfig{}
	milvusConfig.Connection.Host = host
	milvusConfig.Connection.Port = port
	milvusConfig.Connection.Timeout = 5
	milvusConfig.Collection.Name = fmt.Sprintf(
		"exact_cache_%d",
		time.Now().UnixNano(),
	)
	milvusConfig.Collection.VectorField.Name = "embedding"
	milvusConfig.Collection.VectorField.Dimension = 384
	milvusConfig.Collection.VectorField.MetricType = "COSINE"
	milvusConfig.Collection.Index.Type = "HNSW"
	milvusConfig.Collection.Index.Params.M = 16
	milvusConfig.Collection.Index.Params.EfConstruction = 64
	milvusConfig.Search.Params.Ef = 64
	milvusConfig.Search.TopK = 1
	milvusConfig.Development.AutoCreateCollection = true
	return milvusConfig
}

func TestHybridExactCacheIntegrationDelegatesToMilvus(t *testing.T) {
	if os.Getenv("SKIP_MILVUS_TESTS") == "true" {
		t.Skip("Milvus integration tests disabled")
	}
	host := os.Getenv("MILVUS_HOST")
	if host == "" {
		host = "localhost"
	}
	port := 19530
	if configured := os.Getenv("MILVUS_PORT"); configured != "" {
		parsed, err := strconv.Atoi(configured)
		require.NoError(t, err)
		port = parsed
	}
	milvusConfig := milvusExactTestConfig(host, port)
	cache, err := NewHybridCache(HybridCacheOptions{
		Enabled:        true,
		TTLSeconds:     60,
		EmbeddingModel: "bert",
		Milvus:         milvusConfig,
		DisableRebuildOnStartup: true,
	})
	if err != nil {
		t.Skipf("Milvus unavailable: %v", err)
	}
	t.Cleanup(func() {
		_ = cache.milvusCache.client.DropCollection(
			context.Background(),
			cache.milvusCache.collectionName,
		)
		_ = cache.Close()
	})

	fingerprint := fmt.Sprintf("exact-%d", time.Now().UnixNano())
	require.NoError(
		t,
		cache.AddExact("tenant-a", fingerprint, []byte(`{"answer":"cached"}`), 60),
	)
	hit, err := cache.FindExact("tenant-a", fingerprint)
	require.NoError(t, err)
	require.True(t, hit.Found)
	assert.JSONEq(t, `{"answer":"cached"}`, string(hit.ResponseBody))

	miss, err := cache.FindExact("tenant-b", fingerprint)
	require.NoError(t, err)
	assert.False(t, miss.Found)
}
