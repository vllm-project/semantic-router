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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

const (
	remoteVectorIncoming   = "enable logging for this service"
	remoteVectorOpposite   = "disable logging for this service"
	remoteVectorParaphrase = "turn on service logging"
)

func remoteVectorPolarityProvider() embedding.Provider {
	// Deliberately identical fixture vectors put all three queries above threshold.
	// These tests validate storage/guard wiring, not language-model quality.
	return storagetest.Vectors{Size: 384, Aliases: map[string]string{
		remoteVectorOpposite: remoteVectorIncoming, remoteVectorParaphrase: remoteVectorIncoming,
		"": remoteVectorIncoming,
	}}
}

func remoteVectorStorageAddress(t *testing.T, prefix string, defaultPort int) (string, int) {
	t.Helper()
	host := os.Getenv(prefix + "_HOST")
	if host == "" {
		host = "localhost"
	}
	port := defaultPort
	if configured := os.Getenv(prefix + "_PORT"); configured != "" {
		parsed, err := strconv.Atoi(configured)
		require.NoError(t, err)
		port = parsed
	}
	return host, port
}

// StorageIntegration: milvus
func TestMilvusSemanticPolarityStorageIntegration(t *testing.T) {
	storagetest.Require(t, "milvus")
	host, port := remoteVectorStorageAddress(t, "MILVUS", 19530)
	cfg := milvusExactTestConfig(host, port)
	cfg.Search.TopK = 5
	cfg.Search.ConsistencyLevel = "Strong"
	backend, err := NewMilvusCache(MilvusCacheOptions{EmbeddingProvider: remoteVectorPolarityProvider(), Enabled: true, TTLSeconds: 60, EmbeddingModel: "bert", Config: cfg})
	if err != nil {
		storagetest.Unavailable(t, "milvus", err)
	}
	t.Cleanup(func() {
		_ = backend.client.DropCollection(context.Background(), backend.collectionName)
		_ = backend.Close()
	})
	exerciseRemoteVectorPolarityStorage(t, backend, MilvusCacheType)
	t.Run("expired by-ID candidate is absent", func(t *testing.T) {
		ctx := context.Background()
		require.NoError(t, backend.AddEntry(ctx, "expiring", "ttl-partition", remoteVectorIncoming, nil, []byte("expired"), 10))
		live, readErr := backend.GetByID(ctx, "expiring", "ttl-partition")
		require.NoError(t, readErr)
		require.Equal(t, "expired", string(live))
		require.Eventually(t, func() bool {
			_, probeErr := backend.GetByID(ctx, "expiring", "ttl-partition")
			return probeErr != nil
		}, 12*time.Second, 100*time.Millisecond)
		_, readErr = backend.GetByID(ctx, "expiring", "ttl-partition")
		require.ErrorIs(t, readErr, errMilvusCacheEntryNotFound)
	})
}

// StorageIntegration: qdrant
func TestQdrantSemanticPolarityStorageIntegration(t *testing.T) {
	storagetest.Require(t, "qdrant")
	host, port := remoteVectorStorageAddress(t, "QDRANT", 6334)
	backend, err := NewQdrantCache(QdrantCacheOptions{EmbeddingProvider: remoteVectorPolarityProvider(), Enabled: true, TTLSeconds: 60, EmbeddingModel: "bert", Config: &config.QdrantConfig{Host: host, Port: port, ConnectTimeout: 5, CollectionName: fmt.Sprintf("polarity_cache_%d", time.Now().UnixNano())}})
	if err != nil {
		storagetest.Unavailable(t, "qdrant", err)
	}
	t.Cleanup(func() {
		_ = backend.client.DeleteCollection(context.Background(), backend.collectionName)
		_ = backend.Close()
	})
	exerciseRemoteVectorPolarityStorage(t, backend, QdrantCacheType)
}

func exerciseRemoteVectorPolarityStorage(t *testing.T, backend CacheBackend, kind CacheBackendType) {
	t.Helper()
	ctx := context.Background()
	identity := CacheIdentity{Partition: CachePartition{RequestModel: "tenant-a", Protocol: "openai:body"}, SemanticQuery: remoteVectorIncoming}
	adapter := NewLegacyBackendAdapter(backend, kind).WithEmbeddingProvider(remoteVectorPolarityProvider())
	service := NewResponseCacheService(adapter, ResponseCacheServiceOptions{L1MaxEntries: -1})
	opposite := identity
	opposite.SemanticQuery = remoteVectorOpposite
	require.NoError(t, service.StoreSemantic(ctx, CacheWrite{Identity: opposite, RequestID: "opposite", ResponseBody: []byte("opposite"), TTL: TTL(time.Minute)}))
	// Verify the stored entry is visible before asserting the guarded miss.
	control, err := service.LookupSemantic(ctx, SemanticLookup{Identity: opposite, Threshold: .8})
	require.NoError(t, err)
	require.True(t, control.Found)
	miss, err := service.LookupSemantic(ctx, SemanticLookup{Identity: identity, Threshold: .8})
	require.NoError(t, err)
	require.False(t, miss.Found)

	paraphrase := identity
	paraphrase.SemanticQuery = remoteVectorParaphrase
	require.NoError(t, service.StoreSemantic(ctx, CacheWrite{Identity: paraphrase, RequestID: "paraphrase", ResponseBody: []byte("correct"), TTL: TTL(time.Minute)}))
	hit, err := service.LookupSemantic(ctx, SemanticLookup{Identity: identity, Threshold: .8})
	require.NoError(t, err)
	require.True(t, hit.Found)
	require.Equal(t, "correct", string(hit.ResponseBody))
	require.True(t, hit.AgeKnown)
	require.WithinDuration(t, time.Now().Add(time.Minute), hit.ExpiresAt, 5*time.Second)

	other := identity
	other.Partition.RequestModel = "tenant-b"
	isolated, err := service.LookupSemantic(ctx, SemanticLookup{Identity: other, Threshold: .8})
	require.NoError(t, err)
	require.False(t, isolated.Found)
	// Older records that lack their original query must not be replayed.
	require.NoError(t, backend.AddEntry(ctx, "legacy-unknown", service.ResolveIdentity(other).SemanticPartitionKey(), "", nil, []byte("unknown"), 60))
	unknown, err := service.LookupSemantic(ctx, SemanticLookup{Identity: other, Threshold: .8})
	require.NoError(t, err)
	require.False(t, unknown.Found)
}
