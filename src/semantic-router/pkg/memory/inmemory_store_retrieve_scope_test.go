package memory

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

// Record the in-memory capability difference from Milvus and Valkey, which
// reject an empty retrieval UserID. This is not a shared Store contract.
func TestInMemoryStoreRetrieveUserScope(t *testing.T) {
	ctx := context.Background()
	provider, err := embedding.NewFuncProvider("test", 1, func(context.Context, string) ([]float32, error) {
		return []float32{1}, nil
	})
	require.NoError(t, err)
	store := NewInMemoryStoreWithConfig(EmbeddingConfig{Provider: provider})
	t.Cleanup(func() { require.NoError(t, store.Close()) })
	for _, memory := range []*Memory{
		{ID: "alice-fact", UserID: "alice", ProjectID: "shared", Type: MemoryTypeSemantic, Embedding: []float32{1}},
		{ID: "bob-fact", UserID: "bob", ProjectID: "shared", Type: MemoryTypeSemantic, Embedding: []float32{1}},
		{ID: "bob-other-project", UserID: "bob", ProjectID: "other", Type: MemoryTypeSemantic, Embedding: []float32{1}},
		{ID: "bob-procedure", UserID: "bob", ProjectID: "shared", Type: MemoryTypeProcedural, Embedding: []float32{1}},
	} {
		require.NoError(t, store.Store(ctx, memory))
	}

	for _, tc := range []struct {
		name string
		opts RetrieveOptions
		ids  []string
	}{
		{
			name: "explicit user",
			opts: RetrieveOptions{UserID: "alice"},
			ids:  []string{"alice-fact"},
		},
		{
			name: "unknown user",
			opts: RetrieveOptions{UserID: "unknown"},
		},
		{
			name: "empty user searches all users",
			ids:  []string{"alice-fact", "bob-fact", "bob-other-project", "bob-procedure"},
		},
		{
			name: "empty user with project filter",
			opts: RetrieveOptions{ProjectID: "shared"},
			ids:  []string{"alice-fact", "bob-fact", "bob-procedure"},
		},
		{
			name: "empty user with type filter",
			opts: RetrieveOptions{Types: []MemoryType{MemoryTypeSemantic}},
			ids:  []string{"alice-fact", "bob-fact", "bob-other-project"},
		},
		{
			name: "empty user with project and type filters",
			opts: RetrieveOptions{ProjectID: "shared", Types: []MemoryType{MemoryTypeSemantic}},
			ids:  []string{"alice-fact", "bob-fact"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			opts := tc.opts
			opts.Query = "turn"
			opts.Threshold = 0.5
			results, err := store.Retrieve(ctx, opts)
			require.NoError(t, err)
			var ids []string
			for _, result := range results {
				ids = append(ids, result.Memory.ID)
			}
			// Equal vectors have tied scores; map iteration does not define order.
			require.ElementsMatch(t, tc.ids, ids)
		})
	}
}
