//go:build !windows

package cache

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/qdrant/go-client/qdrant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const expiredRowSweepPartition = "tenant-sweep"

// StorageIntegration: milvus
func TestMilvusDeleteExpiredRowsKeepsUnexpiredRows(t *testing.T) {
	storagetest.Require(t, "milvus")
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
	cache, err := NewMilvusCache(MilvusCacheOptions{
		EmbeddingProvider:   cacheTestEmbeddingProvider(),
		Enabled:             true,
		TTLSeconds:          0,
		EmbeddingDeployment: "qwen3-embedding",
		Config:              milvusExactTestConfig(host, port),
	})
	if err != nil {
		storagetest.Unavailable(t, "milvus", fmt.Sprintf("Milvus unavailable: %v", err))
	}
	t.Cleanup(func() {
		_ = cache.client.DropCollection(context.Background(), cache.collectionName)
		_ = cache.Close()
	})

	ctx := context.Background()
	expiringID, keptID := writeExpiringAndKeptExactRows(t, cache)
	require.Equal(t, 2, countMilvusRows(t, cache, expiringID, keptID))

	require.NoError(t, cache.deleteExpiredRows(ctx))

	assert.Equal(t, 0, countMilvusRows(t, cache, expiringID))
	assert.Equal(t, 1, countMilvusRows(t, cache, keptID))
	assert.NotNil(t, cache.lastCleanupTime)
}

// StorageIntegration: qdrant
func TestQdrantDeleteExpiredRowsKeepsUnexpiredRows(t *testing.T) {
	storagetest.Require(t, "qdrant")
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
		EmbeddingProvider:   cacheTestEmbeddingProvider(),
		Enabled:             true,
		TTLSeconds:          0,
		EmbeddingDeployment: "qwen3-embedding",
		Config: &config.QdrantConfig{
			Host:           host,
			Port:           port,
			ConnectTimeout: 5,
			CollectionName: fmt.Sprintf("expired_sweep_%d", time.Now().UnixNano()),
		},
	})
	if err != nil {
		storagetest.Unavailable(t, "qdrant", fmt.Sprintf("Qdrant unavailable: %v", err))
	}
	t.Cleanup(func() {
		_ = cache.client.DeleteCollection(context.Background(), cache.collectionName)
		_ = cache.Close()
	})

	ctx := context.Background()
	expiringID, keptID := writeExpiringAndKeptExactRows(t, cache)
	require.Equal(t, 2, countQdrantPoints(t, cache, expiringID, keptID))

	require.NoError(t, cache.deleteExpiredRows(ctx))

	assert.Equal(t, 0, countQdrantPoints(t, cache, expiringID))
	assert.Equal(t, 1, countQdrantPoints(t, cache, keptID))
}

// writeExpiringAndKeptExactRows stores one row that expires after a second and
// one that never expires (ttl -1 inherits the cache's zero default), then waits
// until the first is past its expires_at.
func writeExpiringAndKeptExactRows(t *testing.T, cache ExactCacheBackend) (string, string) {
	t.Helper()
	ctx := context.Background()
	suffix := time.Now().UnixNano()
	expiring := fmt.Sprintf("expiring-%d", suffix)
	kept := fmt.Sprintf("kept-%d", suffix)
	require.NoError(t, cache.AddExact(ctx, expiredRowSweepPartition, expiring, []byte(`{"answer":"old"}`), 1))
	require.NoError(t, cache.AddExact(ctx, expiredRowSweepPartition, kept, []byte(`{"answer":"kept"}`), -1))
	time.Sleep(2100 * time.Millisecond)
	return exactCacheRecordID(expiredRowSweepPartition, expiring), exactCacheRecordID(expiredRowSweepPartition, kept)
}

func countMilvusRows(t *testing.T, cache *MilvusCache, ids ...string) int {
	t.Helper()
	quoted := make([]string, 0, len(ids))
	for _, id := range ids {
		quoted = append(quoted, milvusStringLiteral(id))
	}
	expr := fmt.Sprintf("id in [%s]", strings.Join(quoted, ", "))
	rows, err := cache.client.Query(
		context.Background(),
		cache.collectionName,
		nil,
		expr,
		[]string{"id"},
		client.WithSearchQueryConsistencyLevel(entity.ClStrong),
	)
	require.NoError(t, err)
	if len(rows) == 0 {
		return 0
	}
	return rows[0].Len()
}

func countQdrantPoints(t *testing.T, cache *QdrantCache, ids ...string) int {
	t.Helper()
	pointIDs := make([]*qdrant.PointId, 0, len(ids))
	for _, id := range ids {
		pointIDs = append(pointIDs, arbitraryIDToUUID(id))
	}
	points, err := cache.client.Get(context.Background(), &qdrant.GetPoints{
		CollectionName: cache.collectionName,
		Ids:            pointIDs,
	})
	require.NoError(t, err)
	return len(points)
}
