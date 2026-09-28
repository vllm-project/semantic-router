package tools_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

type retrievalFingerprintProvider struct {
	embedding []float32
	dimension int
	backend   string
}

func (p retrievalFingerprintProvider) Embed(context.Context, string) ([]float32, error) {
	return append([]float32(nil), p.embedding...), nil
}

func (p retrievalFingerprintProvider) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	result := make([][]float32, len(texts))
	for index := range result {
		result[index] = append([]float32(nil), p.embedding...)
	}
	return result, nil
}

func (p retrievalFingerprintProvider) Dimension() int { return p.dimension }

func (p retrievalFingerprintProvider) Backend() string { return p.backend }

func retrievalFingerprintTool(name string) openai.ChatCompletionToolParam {
	return openai.ChatCompletionToolParam{
		Type: "function",
		Function: openai.FunctionDefinitionParam{
			Name:        name,
			Description: param.NewOpt("provider description"),
			Parameters:  openai.FunctionParameters{"type": "object"},
		},
	}
}

func newRetrievalFingerprintDatabase(provider retrievalFingerprintProvider, description, category string, tags []string) *tools.ToolsDatabase {
	database := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
		Enabled:          true,
		Backend:          config.EmbeddingBackendOpenAICompatible,
		ModelType:        config.EmbeddingModelTypeRemote,
		TargetDimension:  3,
		Provider:         provider,
		ProviderIdentity: "https://embedding.example/v1|model-a|3",
	})
	if err := database.AddTool(retrievalFingerprintTool("search"), description, category, tags); err != nil {
		panic(err)
	}
	return database
}

func TestToolsDatabaseRetrievalFingerprintIsStableForTagReordering(t *testing.T) {
	provider := retrievalFingerprintProvider{
		embedding: []float32{1, 2, 3},
		dimension: 3,
		backend:   config.EmbeddingBackendOpenAICompatible,
	}
	first := newRetrievalFingerprintDatabase(provider, "find weather", "utility", []string{"weather", "search"})
	second := newRetrievalFingerprintDatabase(provider, "find weather", "utility", []string{"search", "weather"})
	if first.RetrievalFingerprint() != second.RetrievalFingerprint() {
		t.Fatal("tag order alone must not change retrieval fingerprint")
	}
}

func TestToolsDatabaseRetrievalFingerprintTracksMetadataAndEmbedding(t *testing.T) {
	provider := retrievalFingerprintProvider{
		embedding: []float32{1, 2, 3},
		dimension: 3,
		backend:   config.EmbeddingBackendOpenAICompatible,
	}
	base := newRetrievalFingerprintDatabase(provider, "find weather", "utility", []string{"weather"})
	for name, database := range map[string]*tools.ToolsDatabase{
		"description": newRetrievalFingerprintDatabase(provider, "find forecasts", "utility", []string{"weather"}),
		"category":    newRetrievalFingerprintDatabase(provider, "find weather", "analytics", []string{"weather"}),
		"tags":        newRetrievalFingerprintDatabase(provider, "find weather", "utility", []string{"news"}),
		"embedding": newRetrievalFingerprintDatabase(retrievalFingerprintProvider{
			embedding: []float32{1, 2, 4},
			dimension: 3,
			backend:   config.EmbeddingBackendOpenAICompatible,
		}, "find weather", "utility", []string{"weather"}),
	} {
		if base.RetrievalFingerprint() == database.RetrievalFingerprint() {
			t.Errorf("%s change did not invalidate retrieval fingerprint", name)
		}
	}
}

func TestToolsDatabaseRetrievalFingerprintTracksProviderIdentityAndDimension(t *testing.T) {
	baseProvider := retrievalFingerprintProvider{
		embedding: []float32{1, 2, 3},
		dimension: 3,
		backend:   config.EmbeddingBackendOpenAICompatible,
	}
	base := newRetrievalFingerprintDatabase(baseProvider, "find weather", "utility", nil)
	changedIdentity := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
		Enabled:          true,
		Backend:          config.EmbeddingBackendOpenAICompatible,
		ModelType:        config.EmbeddingModelTypeRemote,
		TargetDimension:  3,
		Provider:         baseProvider,
		ProviderIdentity: "https://embedding.example/v1|model-b|3",
	})
	if err := changedIdentity.AddTool(retrievalFingerprintTool("search"), "find weather", "utility", nil); err != nil {
		t.Fatalf("AddTool changed identity: %v", err)
	}
	if base.RetrievalFingerprint() == changedIdentity.RetrievalFingerprint() {
		t.Fatal("provider identity change did not invalidate retrieval fingerprint")
	}

	changedDimension := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
		Enabled:         true,
		Backend:         config.EmbeddingBackendOpenAICompatible,
		ModelType:       config.EmbeddingModelTypeRemote,
		TargetDimension: 4,
		Provider: retrievalFingerprintProvider{
			embedding: []float32{1, 2, 3},
			dimension: 3,
			backend:   config.EmbeddingBackendOpenAICompatible,
		},
		ProviderIdentity: "https://embedding.example/v1|model-a|3",
	})
	if err := changedDimension.AddTool(retrievalFingerprintTool("search"), "find weather", "utility", nil); err != nil {
		t.Fatalf("AddTool changed dimension: %v", err)
	}
	if base.RetrievalFingerprint() == changedDimension.RetrievalFingerprint() {
		t.Fatal("configured target dimension change did not invalidate retrieval fingerprint")
	}
}

func TestToolsDatabaseSnapshotPairsToolsAndFingerprint(t *testing.T) {
	provider := retrievalFingerprintProvider{
		embedding: []float32{1, 2, 3},
		dimension: 3,
		backend:   config.EmbeddingBackendOpenAICompatible,
	}
	database := newRetrievalFingerprintDatabase(provider, "find weather", "utility", nil)
	toolSnapshot, fingerprint := database.Snapshot()
	if len(toolSnapshot) != 1 || toolSnapshot[0].Function.Name != "search" {
		t.Fatalf("snapshot tools = %#v", toolSnapshot)
	}
	if fingerprint == "" || fingerprint != database.RetrievalFingerprint() {
		t.Fatalf("snapshot fingerprint = %q, current = %q", fingerprint, database.RetrievalFingerprint())
	}

	encoded, err := json.Marshal(toolSnapshot)
	if err != nil || len(encoded) == 0 {
		t.Fatalf("snapshot should remain serializable: %v", err)
	}
}

func TestToolsDatabaseSnapshotDeepCopiesToolParameters(t *testing.T) {
	provider := retrievalFingerprintProvider{
		embedding: []float32{1, 2, 3},
		dimension: 3,
		backend:   config.EmbeddingBackendOpenAICompatible,
	}
	tool := retrievalFingerprintTool("search")
	tool.Function.Parameters = openai.FunctionParameters{
		"type": "object",
		"properties": map[string]any{
			"query": map[string]any{
				"type": "string",
				"enum": []any{"today", "tomorrow"},
			},
		},
		"required": []any{"query"},
	}
	database := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
		Enabled:          true,
		Backend:          config.EmbeddingBackendOpenAICompatible,
		ModelType:        config.EmbeddingModelTypeRemote,
		TargetDimension:  3,
		Provider:         provider,
		ProviderIdentity: "embedding-provider",
	})
	if err := database.AddTool(tool, "find weather", "utility", nil); err != nil {
		t.Fatalf("AddTool: %v", err)
	}

	first, beforeFingerprint := database.Snapshot()
	properties := first[0].Function.Parameters["properties"].(map[string]any)
	query := properties["query"].(map[string]any)
	query["type"] = "number"
	query["enum"].([]any)[0] = "changed"
	first[0].Function.Parameters["required"].([]any)[0] = "changed"
	first[0].Function.Parameters["type"] = "array"

	second, afterFingerprint := database.Snapshot()
	if beforeFingerprint != afterFingerprint {
		t.Fatalf("mutating a snapshot changed the database fingerprint: before=%q after=%q", beforeFingerprint, afterFingerprint)
	}
	secondProperties := second[0].Function.Parameters["properties"].(map[string]any)
	secondQuery := secondProperties["query"].(map[string]any)
	if got := secondQuery["type"]; got != "string" {
		t.Fatalf("snapshot mutation leaked into nested map: got %v", got)
	}
	if got := secondQuery["enum"].([]any)[0]; got != "today" {
		t.Fatalf("snapshot mutation leaked into nested slice: got %v", got)
	}
	if got := second[0].Function.Parameters["required"].([]any)[0]; got != "query" {
		t.Fatalf("snapshot mutation leaked into top-level slice: got %v", got)
	}
}

func TestToolsDatabaseDisabledSnapshotHasNoRetrievalFingerprint(t *testing.T) {
	database := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{Enabled: false})
	toolSnapshot, fingerprint := database.Snapshot()
	if len(toolSnapshot) != 0 || fingerprint != "" {
		t.Fatalf("disabled snapshot = tools=%v fingerprint=%q", toolSnapshot, fingerprint)
	}
}
