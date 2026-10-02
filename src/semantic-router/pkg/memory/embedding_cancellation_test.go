package memory

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

func TestEmbeddingCancellationPreservesProviderError(t *testing.T) {
	for _, operation := range []string{"embed", "store", "retrieve"} {
		t.Run(operation, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			providerErr := errors.New("provider inference failed")
			provider, err := embedding.NewFuncProvider("test", 1, func(context.Context, string) ([]float32, error) {
				cancel()
				return nil, providerErr
			})
			require.NoError(t, err)
			cfg := EmbeddingConfig{Model: EmbeddingModelBERT, Provider: provider}
			backend := NewInMemoryStoreWithConfig(cfg)
			switch operation {
			case "embed":
				_, err = GenerateEmbeddingWithContext(ctx, "turn", cfg)
			case "store":
				err = backend.Store(ctx, &Memory{ID: "failed", Content: "turn"})
			case "retrieve":
				_, err = backend.Retrieve(ctx, RetrieveOptions{Query: "turn"})
			}
			require.ErrorIs(t, ctx.Err(), context.Canceled)
			require.ErrorIs(t, err, providerErr)
			require.ErrorContains(t, err, "bert embedding failed:")
			require.Empty(t, backend.memories)
		})
	}
}

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
	provider, err := embedding.NewFuncProvider("test", 1, func(context.Context, string) ([]float32, error) {
		cancel() // Native work finished after the persistence attempt was canceled.
		return []float32{1}, nil
	})
	require.NoError(t, err)
	vector, err := embedForWrite(ctx, "turn", EmbeddingConfig{Provider: provider})
	require.ErrorIs(t, err, context.Canceled)
	require.Nil(t, vector, "a late result must not reach a backend write")
}

func TestEmbeddingCancellationSkipsNativeWork(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	provider, err := embedding.NewFuncProvider("test", 1, func(context.Context, string) ([]float32, error) {
		t.Fatal("canceled attempt entered native work")
		return nil, nil
	})
	require.NoError(t, err)
	_, err = GenerateEmbeddingWithContext(ctx, "turn", EmbeddingConfig{Provider: provider})
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

func TestRetrievalKeepsEmbeddingCompletedAfterCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	provider, err := embedding.NewFuncProvider("test", 1, func(got context.Context, _ string) ([]float32, error) {
		require.Same(t, ctx, got, "retrieval cancellation must reach the generation provider")
		cancel() // The client disconnected while native work was already running.
		return []float32{1}, nil
	})
	require.NoError(t, err)

	backend := NewInMemoryStoreWithConfig(EmbeddingConfig{Provider: provider})
	memory := &Memory{ID: "remembered", UserID: "user-1", Content: "turn", Embedding: []float32{1}}
	require.NoError(t, backend.Store(context.Background(), memory))

	results, err := backend.Retrieve(ctx, RetrieveOptions{Query: "turn", UserID: "user-1", Limit: 1})
	require.NoError(t, err, "retrieval must keep a vector the native call already produced")
	require.ErrorIs(t, ctx.Err(), context.Canceled)
	require.Len(t, results, 1)
	require.Same(t, memory, results[0].Memory)
	require.Equal(t, float32(1), results[0].Score)
}
