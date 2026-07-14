package extproc

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestResolveSelectionEmbeddingFuncUsesRemoteProvider(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			t.Fatalf("request path = %q, want /v1/embeddings", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"data": []map[string]interface{}{
				{"index": 0, "embedding": []float64{0.1, 0.2}},
			},
		}); err != nil {
			t.Fatalf("encode response: %v", err)
		}
	}))
	defer server.Close()

	embed := resolveSelectionEmbeddingFunc(&config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				EmbeddingConfig: config.HNSWConfig{
					Backend:         config.EmbeddingBackendOpenAICompatible,
					ModelType:       config.EmbeddingModelTypeRemote,
					TargetDimension: 2,
				},
				Endpoint: config.EmbeddingEndpointConfig{
					BaseURL: server.URL + "/v1",
					Model:   "BAAI/bge-m3",
				},
			},
		},
	})

	embedding, err := embed("hello")
	if err != nil {
		t.Fatalf("selection embedding function error = %v", err)
	}
	if len(embedding) != 2 || embedding[0] != float32(0.1) {
		t.Fatalf("embedding = %#v, want two remote values", embedding)
	}
}

func TestResolveSelectionEmbeddingFuncLocalConfigWithRemoteOverride(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", config.EmbeddingBackendOpenAICompatible)
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"data": []map[string]interface{}{{"index": 0, "embedding": []float64{0.3, 0.4}}},
		}); err != nil {
			t.Errorf("encode response: %v", err)
		}
	}))
	defer server.Close()

	cfg := &config.RouterConfig{InlineModels: config.InlineModels{EmbeddingModels: config.EmbeddingModels{
		Qwen3ModelPath: "models/local-qwen3",
		EmbeddingConfig: config.HNSWConfig{
			Backend:         config.EmbeddingBackendCandle,
			ModelType:       config.EmbeddingModelTypeQwen3,
			TargetDimension: 2,
		},
		Endpoint: config.EmbeddingEndpointConfig{BaseURL: server.URL + "/v1", Model: "remote-embedding"},
	}}}

	embedding, err := resolveSelectionEmbeddingFunc(cfg)("hello")
	if err != nil {
		t.Fatalf("selection embedding function error = %v", err)
	}
	if len(embedding) != 2 || embedding[0] != float32(0.3) {
		t.Fatalf("embedding = %#v, want remote values", embedding)
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("remote embedding HTTP calls = %d, want 1", got)
	}
}

func TestToolsProviderRemoteConfigWithLocalOverrideDoesNotCreateRemoteProvider(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", config.EmbeddingBackendOpenVINO)
	t.Setenv("EMBEDDING_MODEL_TYPE_OVERRIDE", "mmbert")
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		calls.Add(1)
	}))
	defer server.Close()

	cfg := &config.RouterConfig{InlineModels: config.InlineModels{EmbeddingModels: config.EmbeddingModels{
		MmBertModelPath: "models/local-mmbert",
		EmbeddingConfig: config.HNSWConfig{
			Backend:   config.EmbeddingBackendOpenAICompatible,
			ModelType: config.EmbeddingModelTypeRemote,
		},
		Endpoint: config.EmbeddingEndpointConfig{BaseURL: server.URL + "/v1", Model: "remote-embedding"},
	}}}
	plan, err := resolveRouterEmbeddingRuntimePlan(cfg)
	if err != nil {
		t.Fatalf("resolve plan: %v", err)
	}
	provider, err := toolsEmbeddingProvider(cfg, plan)
	if err != nil {
		t.Fatalf("tools provider: %v", err)
	}
	if provider != nil {
		t.Fatalf("local override created remote tools provider %T", provider)
	}
	if got := calls.Load(); got != 0 {
		t.Fatalf("remote embedding HTTP calls = %d, want 0", got)
	}
}
