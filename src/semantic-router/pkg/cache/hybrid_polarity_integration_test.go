//go:build !windows && cgo

package cache

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
)

// StorageIntegration: milvus
func TestHybridSemanticPolarityStorageIntegration(t *testing.T) {
	storagetest.Require(t, "milvus")
	host := os.Getenv("MILVUS_HOST")
	if host == "" {
		host = "localhost"
	}
	port := 19530
	if value := os.Getenv("MILVUS_PORT"); value != "" {
		parsed, err := strconv.Atoi(value)
		require.NoError(t, err)
		port = parsed
	}
	cfg := milvusExactTestConfig(host, port)
	cfg.Search.TopK = 5
	cfg.Search.ConsistencyLevel = "Strong"
	const query = "enable logging for this service"
	const opposite = "disable logging for this service"
	const paraphrase = "turn on service logging"
	cache, err := NewHybridCache(HybridCacheOptions{
		Enabled: true, TTLSeconds: 60, EmbeddingModel: "bert", Milvus: cfg,
		SimilarityThreshold: 0.8, MaxMemoryEntries: 16, DisableRebuildOnStartup: true,
		EmbeddingProvider: storagetest.Vectors{Size: 384, Aliases: map[string]string{opposite: query, paraphrase: query}},
	})
	if err != nil {
		storagetest.Unavailable(t, "milvus", fmt.Sprintf("Hybrid Milvus unavailable: %v", err))
	}
	t.Cleanup(func() {
		_ = cache.milvusCache.client.DropCollection(context.Background(), cache.milvusCache.collectionName)
		_ = cache.Close()
	})
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	service := NewResponseCacheService(NewLegacyBackendAdapter(cache, HybridCacheType), ResponseCacheServiceOptions{L1MaxEntries: -1})
	identity := CacheIdentity{Partition: CachePartition{RequestModel: "tenant-a", Protocol: "openai:body"}, SemanticQuery: opposite}
	require.NoError(t, service.StoreSemantic(ctx, CacheWrite{
		Identity: identity, RequestID: "opposite", RequestBody: []byte(`{}`), ResponseBody: []byte("DISABLE_ANSWER"), TTL: TTL(time.Minute),
	}))
	control, err := service.LookupSemantic(ctx, SemanticLookup{Identity: identity, Threshold: 0.8})
	require.NoError(t, err)
	require.True(t, control.Found)
	require.Equal(t, "DISABLE_ANSWER", string(control.ResponseBody))

	identity.SemanticQuery = query
	miss, err := service.LookupSemantic(ctx, SemanticLookup{Identity: identity, Threshold: 0.8})
	require.NoError(t, err)
	require.False(t, miss.Found, "opposite-polarity response escaped HNSW or Milvus filtering")

	// A valid candidate can exist only in persistent storage (for example after
	// another router wrote it). Rejecting the resident nearest neighbor must
	// still reach Milvus's model-scoped, polarity-checked fallback.
	partition := service.ResolveIdentity(identity).SemanticPartitionKey()
	require.NoError(t, cache.milvusCache.AddEntry(ctx, "valid", partition, paraphrase, []byte(`{}`), []byte("ENABLE_ANSWER"), 60))
	hit, err := service.LookupSemantic(ctx, SemanticLookup{Identity: identity, Threshold: 0.8})
	require.NoError(t, err)
	require.True(t, hit.Found)
	require.Equal(t, "ENABLE_ANSWER", string(hit.ResponseBody))
	require.GreaterOrEqual(t, hit.Similarity, float32(0.8))
	require.True(t, hit.AgeKnown)
	require.False(t, hit.ExpiresAt.IsZero())

	identity.Partition.RequestModel = "tenant-b"
	other, err := service.LookupSemantic(ctx, SemanticLookup{Identity: identity, Threshold: 0.8})
	require.NoError(t, err)
	require.False(t, other.Found, "fallback must retain exact model partition isolation")
}
