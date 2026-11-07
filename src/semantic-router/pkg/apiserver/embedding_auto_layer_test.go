//go:build !windows

package apiserver

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

type layerDeclaringProvider struct {
	*embedding.FuncProvider
	dimensions, layers []int
}

func (p layerDeclaringProvider) EmbeddingInfo() embedding.ModelInfo {
	return embedding.ModelInfo{Dimension: p.Dimension(), Dimensions: p.dimensions, Layers: p.layers}
}

func autoLayerSet(t *testing.T) *embedding.Set {
	t.Helper()
	base, err := embedding.NewFuncProvider("synthetic", 768, func(context.Context, string) ([]float32, error) {
		vector := make([]float32, 768)
		vector[0] = 1
		return vector, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return embedding.NewSet(map[string]embedding.Provider{
		"qwen3":  layerDeclaringProvider{base, []int{1024, 768}, nil},
		"mmbert": layerDeclaringProvider{base, []int{768, 256}, []int{3, 6, 11, 22}},
	}, "mmbert")
}

// A quality-first request prefers qwen3, which has no layer exits; auto must
// serve a target_layer request from the prepared model that declares it.
func TestOwnedAutoEmbeddingSkipsModelsWithoutRequestedLayer(t *testing.T) {
	prepared := autoLayerSet(t)
	api := &ClassificationAPIServer{config: &config.RouterConfig{}}
	recorder := httptest.NewRecorder()
	request, ok := api.prepareEmbeddingRequest(recorder, EmbeddingRequest{Model: "auto", Texts: []string{"short text"}, TargetLayer: 6, QualityPriority: 0.9, LatencyPriority: 0.1}, prepared)
	if !ok {
		t.Fatalf("auto request with target_layer 6 rejected: %s", recorder.Body.String())
	}
	results, _, err := buildOwnedEmbeddingResults(context.Background(), prepared, request)
	if err != nil {
		t.Fatalf("auto embedding with target_layer 6 failed: %v", err)
	}
	if len(results) != 1 || results[0].ModelUsed != "mmbert" {
		t.Fatalf("auto embedding results = %+v, want mmbert", results)
	}

	batch, err := ownedBatchSimilarity(context.Background(), prepared, BatchSimilarityRequest{Model: "auto", Query: "short text", Candidates: []string{"other text"}, TargetLayer: 6, TopK: 1, QualityPriority: 0.9, LatencyPriority: 0.1})
	if err != nil || batch.ModelUsed != "mmbert" {
		t.Fatalf("auto batch similarity selected %q (%v), want mmbert", batch.ModelUsed, err)
	}
}

func TestOwnedAutoEmbeddingNamesUnsupportedLayer(t *testing.T) {
	_, _, err := buildOwnedEmbeddingResults(context.Background(), autoLayerSet(t), EmbeddingRequest{Model: "auto", Texts: []string{"short text"}, TargetLayer: 5, QualityPriority: 0.9, LatencyPriority: 0.1})
	if err == nil {
		t.Fatal("auto embedding accepted a layer no prepared candidate declares")
	}
	status, code, message := classifyEmbeddingError(err)
	if status != http.StatusBadRequest || code != "INVALID_INPUT" || !strings.Contains(message, "layer 5") {
		t.Fatalf("unsupported layer mapped to %d %q %q, want 400 INVALID_INPUT naming layer 5", status, code, message)
	}
}
