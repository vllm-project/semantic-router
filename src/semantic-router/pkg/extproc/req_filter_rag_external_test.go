package extproc

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

func TestRetrieveFromExternalAPIEmbedsTheQueryOnce(t *testing.T) {
	tests := []struct {
		format   string
		response map[string]interface{}
		vector   string
	}{
		{
			format: "pinecone",
			response: map[string]interface{}{"matches": []interface{}{
				map[string]interface{}{"metadata": map[string]interface{}{"content": "first"}},
				map[string]interface{}{"metadata": map[string]interface{}{"text": "second"}},
				map[string]interface{}{"metadata": map[string]interface{}{"content": "third"}},
			}},
			vector: `"vector":[1,0,0]`,
		},
		{
			format: "weaviate",
			response: map[string]interface{}{"data": map[string]interface{}{"Get": map[string]interface{}{"Document": []interface{}{
				map[string]interface{}{"content": "first"},
				map[string]interface{}{"content": "second"},
				map[string]interface{}{"content": "third"},
			}}}},
			vector: "vector: [1,0,0]",
		},
	}

	for _, tt := range tests {
		t.Run(tt.format, func(t *testing.T) {
			var requests atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				body, err := io.ReadAll(r.Body)
				if err != nil || !strings.Contains(string(body), tt.vector) {
					t.Errorf("request does not carry the query vector %s: %s", tt.vector, body)
				}
				if err := json.NewEncoder(w).Encode(tt.response); err != nil {
					t.Errorf("encode response: %v", err)
				}
			}))
			defer server.Close()

			embedder := &plainRAGEmbedder{}
			router := &OpenAIRouter{Embeddings: embedding.NewSet(
				map[string]embedding.Provider{config.RAGQueryEmbeddingModel: embedder},
				config.RAGQueryEmbeddingModel,
			)}
			topK := 2
			ragConfig := &config.RAGPluginConfig{
				Enabled:       true,
				Backend:       "external_api",
				TopK:          &topK,
				BackendConfig: config.MustStructuredPayload(&config.ExternalAPIRAGConfig{Endpoint: server.URL, RequestFormat: tt.format}),
			}
			query := strings.Repeat("a long preamble before the question. ", 64) + "how long does a refund take"

			retrieved, err := router.retrieveFromExternalAPI(context.Background(), &RequestContext{UserContent: query}, ragConfig)
			if err != nil {
				t.Fatalf("retrieveFromExternalAPI() error = %v", err)
			}
			if got := requests.Load(); got != 1 {
				t.Fatalf("sent %d requests, want 1", got)
			}
			if len(embedder.seen) != 1 || embedder.seen[0] != query {
				t.Fatalf("embedded %q, want the whole query once", embedder.seen)
			}
			if want := "first\n\n---\n\nsecond"; retrieved != want {
				t.Fatalf("retrieved %q, want the first top_k documents %q", retrieved, want)
			}
		})
	}
}

func TestRetrieveFromExternalAPIResponseLimit(t *testing.T) {
	responseBody, err := json.Marshal(map[string]interface{}{"content": "retrieved context"})
	if err != nil {
		t.Fatalf("marshal response: %v", err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(responseBody)
	}))
	defer server.Close()

	tests := []struct {
		name             string
		maxResponseBytes int64
		wantErr          bool
	}{
		{name: "default"},
		{name: "at limit", maxResponseBytes: int64(len(responseBody))},
		{name: "one byte over", maxResponseBytes: int64(len(responseBody)) - 1, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ragConfig := &config.RAGPluginConfig{
				Enabled: true,
				Backend: "external_api",
				BackendConfig: config.MustStructuredPayload(&config.ExternalAPIRAGConfig{
					Endpoint:         server.URL,
					RequestFormat:    "custom",
					RequestTemplate:  `{"query":"{{.Query}}"}`,
					MaxResponseBytes: tt.maxResponseBytes,
				}),
			}

			contextText, err := (&OpenAIRouter{}).retrieveFromExternalAPI(
				context.Background(),
				&RequestContext{UserContent: "hello"},
				ragConfig,
			)
			if tt.wantErr {
				if err == nil || !strings.Contains(err.Error(), "response body exceeds limit") {
					t.Fatalf("retrieveFromExternalAPI() error = %v, want response limit error", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("retrieveFromExternalAPI() error = %v", err)
			}
			if contextText != "retrieved context" {
				t.Fatalf("context = %q, want retrieved context", contextText)
			}
		})
	}
}
