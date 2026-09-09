package memory

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
)

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
