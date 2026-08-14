package contextcompression

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"sync"
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

type MemoizedEmbeddingScorer struct {
	embedBatch EmbeddingBatchFunc
	maxEntries int

	mu    sync.Mutex
	cache map[string][]float32
	order []string
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
		maxEntries: maxEntries,
		cache:      make(map[string][]float32, maxEntries),
		order:      make([]string, 0, maxEntries),
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
	scorer.mu.Lock()
	for index, input := range inputs {
		keys[index] = embeddingMemoKey(modelRef, input)
		if cached, ok := scorer.cache[keys[index]]; ok {
			vectors[index] = append([]float32(nil), cached...)
			continue
		}
		missingInputs = append(missingInputs, input)
		missingIndexes = append(missingIndexes, index)
	}
	scorer.mu.Unlock()
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
	scorer.mu.Lock()
	defer scorer.mu.Unlock()
	for missingIndex, vector := range embedded {
		index := missingIndexes[missingIndex]
		if cached, ok := scorer.cache[keys[index]]; ok {
			vectors[index] = append([]float32(nil), cached...)
			continue
		}
		vectors[index] = append([]float32(nil), vector...)
		scorer.cache[keys[index]] = append([]float32(nil), vector...)
		scorer.order = append(scorer.order, keys[index])
		for len(scorer.order) > scorer.maxEntries {
			oldest := scorer.order[0]
			scorer.order = scorer.order[1:]
			delete(scorer.cache, oldest)
		}
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
