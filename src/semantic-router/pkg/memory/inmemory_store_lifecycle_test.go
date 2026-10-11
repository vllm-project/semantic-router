package memory

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

func TestInMemoryStoreCloseReleasesMemories(t *testing.T) {
	ctx := context.Background()
	store := NewInMemoryStore()
	require.True(t, store.IsEnabled())
	require.NoError(t, store.CheckConnection(ctx))
	for _, id := range []string{"first", "second"} {
		require.NoError(t, store.Store(ctx, &Memory{
			ID: id, UserID: "user-1", Embedding: []float32{1},
		}))
	}
	require.Len(t, store.memories, 2, "seed data must exist before checking release")

	for attempt := 0; attempt < 2; attempt++ {
		require.NoError(t, store.Close(), "Close must be idempotent")
		require.Empty(t, store.memories, "Close must release the store's retained memories")
		require.False(t, store.IsEnabled())
		require.ErrorContains(t, store.CheckConnection(ctx), "not enabled")
	}
}

func TestInMemoryStoreClosedOperations(t *testing.T) {
	ctx := context.Background()
	provider, err := embedding.NewFuncProvider("test", 1, func(context.Context, string) ([]float32, error) {
		t.Fatal("closed operations must not request embeddings")
		return nil, nil
	})
	require.NoError(t, err)
	store := NewInMemoryStoreWithConfig(EmbeddingConfig{Provider: provider})
	require.NoError(t, store.Store(ctx, &Memory{
		ID: "remembered", UserID: "user-1", Embedding: []float32{1},
	}))
	require.NoError(t, store.Close())

	t.Run("Get", func(t *testing.T) {
		memory, err := store.Get(ctx, "remembered")
		require.ErrorContains(t, err, "not enabled")
		require.Nil(t, memory)
	})
	t.Run("List", func(t *testing.T) {
		page, err := store.List(ctx, ListOptions{UserID: "user-1"})
		require.ErrorContains(t, err, "not enabled")
		require.Nil(t, page)
	})
	t.Run("Retrieve", func(t *testing.T) {
		// As with Store, do not establish nil as the disabled error contract.
		results, _ := store.Retrieve(ctx, RetrieveOptions{Query: "turn", UserID: "user-1"})
		require.Empty(t, results)
	})
	for name, operation := range map[string]func() error{
		"Update": func() error {
			return store.Update(ctx, "remembered", &Memory{Content: "changed"})
		},
		"Forget": func() error { return store.Forget(ctx, "remembered") },
		"ForgetByScope": func() error {
			return store.ForgetByScope(ctx, MemoryScope{UserID: "user-1"})
		},
	} {
		t.Run(name, func(t *testing.T) { require.ErrorContains(t, operation(), "not enabled") })
	}
}

func TestInMemoryStoreClosedWritesDoNotPersist(t *testing.T) {
	for _, precomputed := range []bool{false, true} {
		name := "needs embedding"
		if precomputed {
			name = "precomputed embedding"
		}
		t.Run(name, func(t *testing.T) {
			provider, err := embedding.NewFuncProvider("test", 1, func(context.Context, string) ([]float32, error) {
				t.Fatal("closed store must not request embeddings")
				return nil, nil
			})
			require.NoError(t, err)
			store := NewInMemoryStoreWithConfig(EmbeddingConfig{Provider: provider})
			require.NoError(t, store.Close())
			memory := &Memory{ID: "after-close", Content: "turn", UserID: "user-1"}
			if precomputed {
				memory.Embedding = []float32{1}
			}

			// #4393 leaves the closed-write error contract for maintainer agreement.
			// Verify no persistence without treating the current nil error as a guarantee.
			_ = store.Store(context.Background(), memory)
			require.Empty(t, store.memories)
			require.False(t, store.IsEnabled(), "a write must not reopen the store")
			require.Zero(t, memory.CreatedAt, "a closed write must not prepare a persisted record")
		})
	}
}

func TestInMemoryStoreLifetimeAndEmbeddingOwnership(t *testing.T) {
	ctx := context.Background()
	embeddingCalls := 0
	provider, err := embedding.NewFuncProvider("test", 1, func(context.Context, string) ([]float32, error) {
		embeddingCalls++
		return []float32{1}, nil
	})
	require.NoError(t, err)
	cfg := EmbeddingConfig{Provider: provider}
	first := NewInMemoryStoreWithConfig(cfg)
	require.NoError(t, first.Store(ctx, &Memory{ID: "old", UserID: "user-1", Content: "turn"}))
	require.Equal(t, 1, embeddingCalls)
	require.NoError(t, first.Close())

	// A new instance starts empty and can still use the caller-owned provider.
	second := NewInMemoryStoreWithConfig(cfg)
	t.Cleanup(func() { require.NoError(t, second.Close()) })
	require.True(t, second.IsEnabled())
	require.NoError(t, second.CheckConnection(ctx))
	page, err := second.List(ctx, ListOptions{UserID: "user-1"})
	require.NoError(t, err)
	require.Zero(t, page.Total)
	require.Empty(t, page.Memories)
	require.NoError(t, second.Store(ctx, &Memory{ID: "new", UserID: "user-1", Content: "turn"}))
	require.Equal(t, 2, embeddingCalls)
	results, err := second.Retrieve(ctx, RetrieveOptions{Query: "turn", UserID: "user-1"})
	require.NoError(t, err)
	require.Len(t, results, 1)
	require.Equal(t, "new", results[0].Memory.ID)
	require.Equal(t, 3, embeddingCalls)
	require.False(t, first.IsEnabled(), "the new instance must not reopen the old instance")
}
