package extproc

import (
	"context"
	"testing"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
	requestTools := []openai.ChatCompletionToolParam{
		{Function: openai.FunctionDefinitionParam{Name: "get_weather", Description: param.NewOpt("weather reports")}},
		{Function: openai.FunctionDefinitionParam{Name: "calculator", Description: param.NewOpt("math")}},
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
	if confidence <= 0 || len(filtered) != 1 || filtered[0].Function.Name != "get_weather" {
		t.Fatalf("filtered=%+v confidence=%v, want get_weather", filtered, confidence)
	}
}

func TestToolEmbeddingText_IncludesDescription(t *testing.T) {
	tp := openai.ChatCompletionToolParam{
		Type: "function",
		Function: openai.FunctionDefinitionParam{
			Name:        "alpha",
			Description: param.NewOpt("desc here"),
		},
	}
	if got := toolEmbeddingText(tp); got != "alpha desc here" {
		t.Fatalf("unexpected text: %q", got)
	}
}
