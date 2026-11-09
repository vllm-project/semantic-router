//go:build !windows

package apiserver

import (
	"context"
	"net/http"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

type dimensionDeclaringProvider struct {
	*embedding.FuncProvider
	dimensions []int
}

func (p dimensionDeclaringProvider) EmbeddingInfo() embedding.ModelInfo {
	return embedding.ModelInfo{Dimension: p.Dimension(), Dimensions: p.dimensions}
}

func autoDimensionSet(t *testing.T) *embedding.Set {
	t.Helper()
	base, err := embedding.NewFuncProvider("synthetic", 1024, func(context.Context, string) ([]float32, error) {
		vector := make([]float32, 1024)
		vector[0] = 1
		return vector, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return embedding.NewSet(map[string]embedding.Provider{
		"qwen3":  dimensionDeclaringProvider{base, []int{1024, 768}},
		"mmbert": dimensionDeclaringProvider{base, []int{768, 256}},
	}, "mmbert")
}

// Long inputs prefer mmbert, which does not declare 1024; auto must serve the
// request from the prepared model that does instead of rejecting it.
func TestOwnedAutoEmbeddingSkipsModelsWithoutRequestedDimension(t *testing.T) {
	prepared := autoDimensionSet(t)
	long := strings.Repeat("word ", 600)

	results, _, err := buildOwnedEmbeddingResults(context.Background(), prepared, EmbeddingRequest{Model: "auto", Texts: []string{long}, Dimension: 1024, QualityPriority: 0.5, LatencyPriority: 0.5})
	if err != nil {
		t.Fatalf("auto embedding with 1024 dimensions failed: %v", err)
	}
	if len(results) != 1 || results[0].ModelUsed != "qwen3" {
		t.Fatalf("auto embedding results = %+v, want qwen3", results)
	}

	batch, err := ownedBatchSimilarity(context.Background(), prepared, BatchSimilarityRequest{Model: "auto", Query: long, Candidates: []string{"short text"}, Dimension: 1024, TopK: 1, QualityPriority: 0.5, LatencyPriority: 0.5})
	if err != nil || batch.ModelUsed != "qwen3" {
		t.Fatalf("auto batch similarity selected %q (%v), want qwen3", batch.ModelUsed, err)
	}
}

func TestOwnedAutoEmbeddingNamesUnsupportedDimension(t *testing.T) {
	_, _, err := buildOwnedEmbeddingResults(context.Background(), autoDimensionSet(t), EmbeddingRequest{Model: "auto", Texts: []string{"short text"}, Dimension: 384, QualityPriority: 0.5, LatencyPriority: 0.5})
	if err == nil {
		t.Fatal("auto embedding accepted a dimension no prepared model declares")
	}
	status, code, message := classifyEmbeddingError(err)
	if status != http.StatusBadRequest || code != "INVALID_INPUT" || !strings.Contains(message, "dimension 384") {
		t.Fatalf("unsupported dimension mapped to %d %q %q, want 400 INVALID_INPUT naming dimension 384", status, code, message)
	}
}
