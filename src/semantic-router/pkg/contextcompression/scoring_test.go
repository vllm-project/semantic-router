package contextcompression

import (
	"context"
	"sync/atomic"
	"testing"
)

func TestMemoizedEmbeddingScorerBatchesAndCaches(t *testing.T) {
	var calls atomic.Int64
	scorer := NewMemoizedEmbeddingScorer(
		func(_ context.Context, texts []string) ([][]float32, error) {
			calls.Add(1)
			vectors := make([][]float32, len(texts))
			for index, text := range texts {
				vectors[index] = []float32{float32(len(text)), 1}
			}
			return vectors, nil
		},
		8,
	)
	for range 2 {
		scores, err := scorer.Score(
			context.Background(),
			"embedding-model",
			"query",
			[]string{"first", "second"},
		)
		if err != nil || len(scores) != 2 {
			t.Fatalf("Score() = %#v, %v", scores, err)
		}
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("embedding batches = %d, want 1", got)
	}
}
