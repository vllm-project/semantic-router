package extproc

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type stubToolSelectionEmbeddingProvider struct {
	embeddings map[string][]float32
}

func (p *stubToolSelectionEmbeddingProvider) Embed(_ context.Context, text string) ([]float32, error) {
	if embedding, ok := p.embeddings[text]; ok {
		return embedding, nil
	}
	return []float32{0, 0, 1}, nil
}

func (p *stubToolSelectionEmbeddingProvider) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
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

func (p *stubToolSelectionEmbeddingProvider) Dimension() int { return 3 }

func (p *stubToolSelectionEmbeddingProvider) Backend() string {
	return config.EmbeddingBackendOpenAICompatible
}

func TestFilterRequestToolsAgainstQuerySemanticUsesRemoteProvider(t *testing.T) {
	provider := &stubToolSelectionEmbeddingProvider{embeddings: map[string][]float32{
		"weather today":               {1, 0, 0},
		"get_weather weather reports": {1, 0, 0},
		"calculator math":             {0, 1, 0},
	}}
	requestTools := []llmprotocol.Tool{
		{Name: "get_weather", Description: "weather reports", InputSchema: []byte(`{"type":"object"}`)},
		{Name: "calculator", Description: "math", InputSchema: []byte(`{"type":"object"}`)},
	}

	filtered, confidence, err := filterRequestToolsAgainstQuerySemantic(
		"weather today",
		requestTools,
		config.EmbeddingModelTypeRemote,
		3,
		provider,
		0.5,
		0,
	)
	if err != nil {
		t.Fatalf("filterRequestToolsAgainstQuerySemantic failed: %v", err)
	}
	if confidence <= 0 || len(filtered) != 1 || filtered[0].Name != "get_weather" {
		t.Fatalf("filtered=%+v confidence=%v, want get_weather", filtered, confidence)
	}
}

func TestToolEmbeddingText_IncludesDescription(t *testing.T) {
	tp := llmprotocol.Tool{Name: "alpha", Description: "desc here"}
	if got := toolEmbeddingText(tp); got != "alpha desc here" {
		t.Fatalf("unexpected text: %q", got)
	}
}
