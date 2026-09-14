package tools_test

import (
	"context"
	"testing"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

type stubToolEmbeddingProvider struct {
	embeddings map[string][]float32
}

func (p *stubToolEmbeddingProvider) Embed(_ context.Context, text string) ([]float32, error) {
	if embedding, ok := p.embeddings[text]; ok {
		return embedding, nil
	}
	return []float32{0, 0, 1}, nil
}

func (p *stubToolEmbeddingProvider) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
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

func (p *stubToolEmbeddingProvider) Dimension() int { return 3 }

func (p *stubToolEmbeddingProvider) Backend() string { return config.EmbeddingBackendOpenAICompatible }

func TestToolsDatabaseUsesRemoteProvider(t *testing.T) {
	database := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
		Enabled:             true,
		SimilarityThreshold: 0.5,
		ModelType:           config.EmbeddingModelTypeRemote,
		TargetDimension:     3,
		Provider: &stubToolEmbeddingProvider{embeddings: map[string][]float32{
			"weather forecast": {1, 0, 0},
			"math calculator":  {0, 1, 0},
			"weather today":    {1, 0, 0},
		}},
	})

	if err := database.AddTool(openai.ChatCompletionToolParam{}, "weather forecast", "weather", nil); err != nil {
		t.Fatalf("AddTool weather failed: %v", err)
	}
	if err := database.AddTool(openai.ChatCompletionToolParam{}, "math calculator", "math", nil); err != nil {
		t.Fatalf("AddTool math failed: %v", err)
	}

	results, err := database.FindSimilarToolsWithScores("weather today", 1)
	if err != nil {
		t.Fatalf("FindSimilarToolsWithScores failed: %v", err)
	}
	if len(results) != 1 || results[0].Entry.Category != "weather" {
		t.Fatalf("results = %+v, want weather tool", results)
	}
}

func TestToolsDatabaseTieBreakIsDeterministic(t *testing.T) {
	database := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
		Enabled:             true,
		SimilarityThreshold: 0,
		ModelType:           config.EmbeddingModelTypeRemote,
		TargetDimension:     3,
		Provider: &stubToolEmbeddingProvider{embeddings: map[string][]float32{
			"alpha description": {1, 0, 0},
			"zeta description":  {1, 0, 0},
			"same query":        {1, 0, 0},
		}},
	})

	// Add in the reverse lexical order to model nondeterministic worker
	// completion order. Equal similarity must still produce one stable order.
	for _, entry := range []struct {
		name        string
		description string
	}{
		{name: "zeta", description: "zeta description"},
		{name: "alpha", description: "alpha description"},
	} {
		if err := database.AddTool(openai.ChatCompletionToolParam{
			Type:     "function",
			Function: openai.FunctionDefinitionParam{Name: entry.name},
		}, entry.description, "", nil); err != nil {
			t.Fatalf("AddTool(%s) failed: %v", entry.name, err)
		}
	}

	results, err := database.FindSimilarToolsWithScores("same query", 2)
	if err != nil {
		t.Fatalf("FindSimilarToolsWithScores failed: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("result count = %d, want 2", len(results))
	}
	if got := results[0].Entry.Tool.Function.Name; got != "alpha" {
		t.Fatalf("first tied result = %q, want alpha", got)
	}
	if got := results[1].Entry.Tool.Function.Name; got != "zeta" {
		t.Fatalf("second tied result = %q, want zeta", got)
	}
}
