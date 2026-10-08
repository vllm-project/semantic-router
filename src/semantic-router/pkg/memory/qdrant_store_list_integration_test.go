//go:build !windows

package memory

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/qdrant/go-client/qdrant"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// StorageIntegration: qdrant
func TestQdrantListIntegrationCrossesOrderedScrollBatches(t *testing.T) {
	storagetest.Require(t, "qdrant")
	ctx := context.Background()
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

	client, err := qdrant.NewClient(&qdrant.Config{Host: host, Port: port})
	if err != nil {
		storagetest.Unavailable(t, "qdrant", err)
		return
	}
	collection := fmt.Sprintf("memory_list_pagination_%d", time.Now().UnixNano())
	store, err := NewQdrantStore(QdrantStoreOptions{
		Client: client,
		Config: config.MemoryConfig{},
		QdrantConfig: &config.MemoryQdrantConfig{
			Host: host, Port: port, Collection: collection, Dimension: 2,
		},
		Enabled:         true,
		EmbeddingConfig: &EmbeddingConfig{Model: EmbeddingModelQwen3, Dimension: 2},
	})
	if err != nil {
		_ = client.Close()
		storagetest.Unavailable(t, "qdrant", err)
		return
	}
	t.Cleanup(func() {
		_ = client.DeleteCollection(context.Background(), collection)
		_ = store.Close()
	})

	createdAt := time.Unix(1_700_000_000, 0)
	for i := 0; i < 105; i++ {
		require.NoError(t, store.Store(ctx, &Memory{
			ID:        fmt.Sprintf("memory-%03d", i),
			Type:      MemoryTypeSemantic,
			Content:   "pagination test memory",
			Embedding: []float32{1, 0},
			UserID:    "pagination-user",
			CreatedAt: createdAt,
		}))
	}

	first, err := store.List(ctx, ListOptions{UserID: "pagination-user", Limit: 100})
	require.NoError(t, err)
	second, err := store.List(ctx, ListOptions{UserID: "pagination-user", Limit: 20, Offset: 100})
	require.NoError(t, err)
	require.Equal(t, 105, first.Total)
	require.Equal(t, 105, second.Total)
	require.Len(t, first.Memories, 100)
	require.Len(t, second.Memories, 5)
	require.Equal(t, 100, second.Offset)

	got := append(idsOf(first.Memories), idsOf(second.Memories)...)
	want := make([]string, 0, 105)
	for i := 104; i >= 0; i-- {
		want = append(want, fmt.Sprintf("memory-%03d", i))
	}
	require.Equal(t, want, got)
}

// TestQdrantListIntegrationDistinctTimestampsCrossBatch verifies that startFrom
// advances between scroll calls when records span two timestamp groups and the
// newer group fills the first batch exactly.
//
// Layout: 105 records at T+60s (newer), 60 records at T (older). Total = 165.
//
//   - List(Limit:100): first scroll → 100 newer; second scroll → 5 newer + older.
//     Page returns the 100 newest records (all newer timestamps).
//   - List(Limit:20, Offset:100): second page returns the remaining 5 newer and
//     then 15 older records, proving startFrom moved from T+60s to T between calls.
//
// StorageIntegration: qdrant
func TestQdrantListIntegrationDistinctTimestampsCrossBatch(t *testing.T) {
	storagetest.Require(t, "qdrant")
	ctx := context.Background()
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

	client, err := qdrant.NewClient(&qdrant.Config{Host: host, Port: port})
	if err != nil {
		storagetest.Unavailable(t, "qdrant", err)
		return
	}
	collection := fmt.Sprintf("memory_list_distinct_ts_%d", time.Now().UnixNano())
	store, err := NewQdrantStore(QdrantStoreOptions{
		Client: client,
		Config: config.MemoryConfig{},
		QdrantConfig: &config.MemoryQdrantConfig{
			Host: host, Port: port, Collection: collection, Dimension: 2,
		},
		Enabled:         true,
		EmbeddingConfig: &EmbeddingConfig{Model: EmbeddingModelQwen3, Dimension: 2},
	})
	if err != nil {
		_ = client.Close()
		storagetest.Unavailable(t, "qdrant", err)
		return
	}
	t.Cleanup(func() {
		_ = client.DeleteCollection(context.Background(), collection)
		_ = store.Close()
	})

	base := time.Unix(1_700_000_000, 0)
	newer := base.Add(60 * time.Second)
	for i := 0; i < 105; i++ {
		require.NoError(t, store.Store(ctx, &Memory{
			ID:        fmt.Sprintf("newer-%03d", i),
			Type:      MemoryTypeSemantic,
			Content:   "newer timestamp memory",
			Embedding: []float32{1, 0},
			UserID:    "distinct-ts-user",
			CreatedAt: newer,
		}))
	}
	for i := 0; i < 60; i++ {
		require.NoError(t, store.Store(ctx, &Memory{
			ID:        fmt.Sprintf("older-%02d", i),
			Type:      MemoryTypeSemantic,
			Content:   "older timestamp memory",
			Embedding: []float32{1, 0},
			UserID:    "distinct-ts-user",
			CreatedAt: base,
		}))
	}

	first, err := store.List(ctx, ListOptions{UserID: "distinct-ts-user", Limit: 100})
	require.NoError(t, err)
	second, err := store.List(ctx, ListOptions{UserID: "distinct-ts-user", Limit: 20, Offset: 100})
	require.NoError(t, err)

	require.Equal(t, 165, first.Total)
	require.Equal(t, 165, second.Total)
	require.Len(t, first.Memories, 100)
	require.Len(t, second.Memories, 20)

	// Page 1: all 100 records are from the newer group.
	for _, mem := range first.Memories {
		require.True(t, mem.CreatedAt.Equal(newer),
			"expected newer timestamp on page 1, got %v (id=%s)", mem.CreatedAt, mem.ID)
	}

	// Page 2: 5 remaining newer then 15 older.
	for _, mem := range second.Memories[:5] {
		require.True(t, mem.CreatedAt.Equal(newer),
			"expected newer timestamp in first 5 of page 2, got %v (id=%s)", mem.CreatedAt, mem.ID)
	}
	for _, mem := range second.Memories[5:] {
		require.True(t, mem.CreatedAt.Equal(base),
			"expected older timestamp in last 15 of page 2, got %v (id=%s)", mem.CreatedAt, mem.ID)
	}

	// No duplicates across pages (100 + 20 = 120 unique records).
	seen := make(map[string]struct{}, 120)
	for _, mem := range append(first.Memories, second.Memories...) {
		_, dup := seen[mem.ID]
		require.False(t, dup, "duplicate memory ID %s across pages", mem.ID)
		seen[mem.ID] = struct{}{}
	}
	require.Len(t, seen, 120) // 100 page-1 + 20 page-2
}
