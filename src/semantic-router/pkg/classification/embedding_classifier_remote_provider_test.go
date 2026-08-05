package classification

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestEmbeddingClassifierUsesRemoteProvider(t *testing.T) {
	provider := &stubEmbeddingProvider{
		embeddings: map[string][]float32{
			"billing support": {1, 0, 0},
			"billing invoice": {1, 0, 0},
		},
	}
	classifier, err := NewEmbeddingClassifierWithProvider([]config.EmbeddingRule{{
		Name:                      "billing",
		Candidates:                []string{"billing invoice"},
		SimilarityThreshold:       0.9,
		AggregationMethodConfiged: config.AggregationMethodMax,
	}}, config.HNSWConfig{
		Backend:           config.EmbeddingBackendOpenAICompatible,
		ModelType:         config.EmbeddingModelTypeRemote,
		TargetDimension:   3,
		PreloadEmbeddings: true,
	}, provider)
	if err != nil {
		t.Fatalf("NewEmbeddingClassifierWithProvider failed: %v", err)
	}

	if warmupErr := classifier.WarmupCandidateEmbeddings(); warmupErr != nil {
		t.Fatalf("WarmupCandidateEmbeddings failed: %v", warmupErr)
	}
	detailed, err := classifier.ClassifyDetailed("billing support")
	if err != nil {
		t.Fatalf("ClassifyDetailed failed: %v", err)
	}
	if len(detailed.Matches) != 1 || detailed.Matches[0].RuleName != "billing" {
		t.Fatalf("matches = %+v, want billing match", detailed.Matches)
	}
}

type stubEmbeddingProvider struct {
	embeddings map[string][]float32
}

func (p *stubEmbeddingProvider) Embed(_ context.Context, text string) ([]float32, error) {
	if embedding, ok := p.embeddings[text]; ok {
		return embedding, nil
	}
	return []float32{0, 1, 0}, nil
}

func (p *stubEmbeddingProvider) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	embeddings := make([][]float32, len(texts))
	for i, text := range texts {
		embedding, err := p.Embed(ctx, text)
		if err != nil {
			return nil, err
		}
		embeddings[i] = embedding
	}
	return embeddings, nil
}

func (p *stubEmbeddingProvider) Dimension() int { return 3 }

func (p *stubEmbeddingProvider) Backend() string { return config.EmbeddingBackendOpenAICompatible }
