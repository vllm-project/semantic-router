package memory

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

func TestStoreEmbeddingPreservesCancellationContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	called := false
	provider, err := embedding.NewFuncProvider("test", 1, func(got context.Context, _ string) ([]float32, error) {
		called = true
		require.Same(t, ctx, got, "persistence cancellation must reach the generation provider")
		cancel()
		return []float32{1}, nil
	})
	require.NoError(t, err)
	backend := NewInMemoryStoreWithConfig(EmbeddingConfig{Provider: provider})
	err = backend.Store(ctx, &Memory{ID: "cancelled-embedding", Content: "turn"})
	require.True(t, called)
	require.ErrorIs(t, err, context.Canceled)
	require.Empty(t, backend.memories, "a result returned after cancellation must not be persisted")
}

func TestEmbeddingCancellationDiscardsNativeResult(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	embedding, err := generateEmbeddingContext(ctx, "turn", EmbeddingConfig{}, func(string, EmbeddingConfig) ([]float32, error) {
		cancel() // Native work finished after the persistence attempt was canceled.
		return []float32{1}, nil
	})
	require.ErrorIs(t, err, context.Canceled)
	require.Nil(t, embedding, "a late result must not reach a backend write")
}

func TestEmbeddingCancellationSkipsNativeWork(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := generateEmbeddingContext(ctx, "turn", EmbeddingConfig{}, func(string, EmbeddingConfig) ([]float32, error) {
		t.Fatal("canceled attempt entered native work")
		return nil, nil
	})
	require.ErrorIs(t, err, context.Canceled)
}

func TestCanceledStoreDoesNotPersistPrecomputedEmbedding(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	backend := NewInMemoryStore()
	err := backend.Store(ctx, &Memory{ID: "canceled", Embedding: []float32{1}})
	require.ErrorIs(t, err, context.Canceled)
	require.Empty(t, backend.memories)
}
