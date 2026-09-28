package contextcompression

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

type EmbeddingBatchFunc func(context.Context, []string) ([][]float32, error)

type EmbeddingScorer struct {
	EmbedBatch EmbeddingBatchFunc
}

func (scorer EmbeddingScorer) Score(
	ctx context.Context,
	_ string,
	query string,
	texts []string,
) ([]float64, error) {
	if scorer.EmbedBatch == nil {
		return nil, fmt.Errorf("embedding provider is unavailable")
	}
	inputs := make([]string, 0, len(texts)+1)
	inputs = append(inputs, query)
	inputs = append(inputs, texts...)
	vectors, err := scorer.EmbedBatch(ctx, inputs)
	if err != nil {
		return nil, err
	}
	if len(vectors) != len(inputs) {
		return nil, fmt.Errorf(
			"embedding provider returned %d vectors for %d inputs",
			len(vectors),
			len(inputs),
		)
	}
	result := make([]float64, len(texts))
	for index := range texts {
		result[index] = cosineSimilarity(vectors[0], vectors[index+1])
	}
	return result, nil
}

// MemoizedEmbeddingScorer scores against a bounded memo of previously computed
// embeddings. The memo itself is the shared embedding.Memo primitive; this type
// owns the semantics on top of it: sha256 keys namespaced by model, storing
// copies so the memo never aliases provider buffers, and memoizing every input
// including the query. Returned vectors are read-only views (the sole consumer
// computes cosine similarity and retains nothing).
type MemoizedEmbeddingScorer struct {
	embedBatch EmbeddingBatchFunc
	memo       *embedding.Memo
}

func NewMemoizedEmbeddingScorer(
	embedBatch EmbeddingBatchFunc,
	maxEntries int,
) *MemoizedEmbeddingScorer {
	if maxEntries <= 0 {
		maxEntries = 1024
	}
	return &MemoizedEmbeddingScorer{
		embedBatch: embedBatch,
		memo:       embedding.NewMemo(maxEntries),
	}
}

func (scorer *MemoizedEmbeddingScorer) Score(
	ctx context.Context,
	modelRef string,
	query string,
	texts []string,
) ([]float64, error) {
	if scorer == nil || scorer.embedBatch == nil {
		return nil, fmt.Errorf("embedding provider is unavailable")
	}
	inputs := make([]string, 0, len(texts)+1)
	inputs = append(inputs, query)
	inputs = append(inputs, texts...)
	vectors, err := scorer.vectors(ctx, modelRef, inputs)
	if err != nil {
		return nil, err
	}
	result := make([]float64, len(texts))
	for index := range texts {
		result[index] = cosineSimilarity(vectors[0], vectors[index+1])
	}
	return result, nil
}

func (scorer *MemoizedEmbeddingScorer) vectors(
	ctx context.Context,
	modelRef string,
	inputs []string,
) ([][]float32, error) {
	keys := make([]string, len(inputs))
	vectors := make([][]float32, len(inputs))
	missingInputs := make([]string, 0)
	missingIndexes := make([]int, 0)
	for index, input := range inputs {
		keys[index] = embeddingMemoKey(modelRef, input)
		if cached, ok := scorer.memo.Get(keys[index]); ok {
			vectors[index] = cached
			continue
		}
		missingInputs = append(missingInputs, input)
		missingIndexes = append(missingIndexes, index)
	}
	if len(missingInputs) == 0 {
		return vectors, nil
	}
	embedded, err := scorer.embedBatch(ctx, missingInputs)
	if err != nil {
		return nil, err
	}
	if len(embedded) != len(missingInputs) {
		return nil, fmt.Errorf(
			"embedding provider returned %d vectors for %d inputs",
			len(embedded),
			len(missingInputs),
		)
	}
	for missingIndex, vector := range embedded {
		index := missingIndexes[missingIndex]
		// Store a copy so the memo never aliases the provider's slice. Racing
		// misses on the same key may each Put (the first write wins); the copies
		// are interchangeable because an embedding is deterministic per key.
		scorer.memo.Put(keys[index], append([]float32(nil), vector...))
		vectors[index] = vector
	}
	return vectors, nil
}

func embeddingMemoKey(modelRef string, text string) string {
	sum := sha256.Sum256([]byte(modelRef + "\x00" + text))
	return hex.EncodeToString(sum[:])
}

func cosineSimilarity(left []float32, right []float32) float64 {
	if len(left) == 0 || len(left) != len(right) {
		return 0
	}
	var dot float64
	var leftNorm float64
	var rightNorm float64
	for index := range left {
		l := float64(left[index])
		r := float64(right[index])
		dot += l * r
		leftNorm += l * l
		rightNorm += r * r
	}
	if leftNorm == 0 || rightNorm == 0 {
		return 0
	}
	return dot / (math.Sqrt(leftNorm) * math.Sqrt(rightNorm))
}
