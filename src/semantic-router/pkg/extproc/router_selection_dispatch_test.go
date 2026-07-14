package extproc

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
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
