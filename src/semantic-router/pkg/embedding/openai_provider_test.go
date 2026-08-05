package embedding

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestOpenAICompatibleProviderEmbedBatch(t *testing.T) {
	server := newEmbeddingBatchServer(t)
	defer server.Close()

	t.Setenv("EMBEDDING_API_KEY", "test-key")
	provider := newTestOpenAIProvider(t, OpenAICompatibleConfig{
		BaseURL:           server.URL + "/v1",
		Model:             "BAAI/bge-m3",
		APIKeyEnv:         "EMBEDDING_API_KEY",
		Dimensions:        2,
		ExpectedDimension: 2,
	})

	embeddings, err := provider.EmbedBatch(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("EmbedBatch() error = %v", err)
	}
	if len(embeddings) != 2 || len(embeddings[0]) != 2 || len(embeddings[1]) != 2 {
		t.Fatalf("unexpected embeddings shape: %#v", embeddings)
	}
	if embeddings[1][0] != float32(0.3) {
		t.Fatalf("embeddings[1][0] = %v, want 0.3", embeddings[1][0])
	}
}

func newEmbeddingBatchServer(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assertEmbeddingBatchRequest(t, r)
		writeEmbeddingResponse(t, w, [][]float64{{0.1, 0.2}, {0.3, 0.4}})
	}))
}

func assertEmbeddingBatchRequest(t *testing.T, r *http.Request) {
	t.Helper()
	if r.URL.Path != "/v1/embeddings" {
		t.Fatalf("request path = %q, want /v1/embeddings", r.URL.Path)
	}
	if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
		t.Fatalf("Authorization header = %q, want bearer token", got)
	}
	var req embeddingsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	assertEmbeddingRequestBody(t, req)
}

func assertEmbeddingRequestBody(t *testing.T, req embeddingsRequest) {
	t.Helper()
	if req.Model != "BAAI/bge-m3" {
		t.Fatalf("model = %q, want BAAI/bge-m3", req.Model)
	}
	if req.Dimensions != 2 {
		t.Fatalf("dimensions = %d, want 2", req.Dimensions)
	}
	if len(req.Input) != 2 || req.Input[0] != "hello" || req.Input[1] != "world" {
		t.Fatalf("input = %#v, want hello/world", req.Input)
	}
}

func TestOpenAICompatibleProviderRetriesRetryableStatus(t *testing.T) {
	var calls int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		call := atomic.AddInt32(&calls, 1)
		if call == 1 {
			http.Error(w, "temporary failure", http.StatusInternalServerError)
			return
		}
		writeEmbeddingResponse(t, w, [][]float64{{0.1, 0.2}})
	}))
	defer server.Close()

	provider := newTestOpenAIProvider(t, OpenAICompatibleConfig{
		BaseURL:           server.URL,
		Model:             "embedding-model",
		MaxRetries:        1,
		ExpectedDimension: 2,
	})

	if _, err := provider.Embed(context.Background(), "hello"); err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if calls != 2 {
		t.Fatalf("calls = %d, want 2", calls)
	}
}

func TestOpenAICompatibleProviderAuthFailureIsClear(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "bad key", http.StatusUnauthorized)
	}))
	defer server.Close()

	provider := newTestOpenAIProvider(t, OpenAICompatibleConfig{
		BaseURL: server.URL,
		Model:   "embedding-model",
	})

	_, err := provider.Embed(context.Background(), "hello")
	if err == nil || !strings.Contains(err.Error(), "authentication failed") {
		t.Fatalf("Embed() error = %v, want authentication failure", err)
	}
}

func TestOpenAICompatibleProviderRequiresConfiguredAPIKeyEnv(t *testing.T) {
	var calls int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		atomic.AddInt32(&calls, 1)
		writeEmbeddingResponse(t, w, [][]float64{{0.1, 0.2}})
	}))
	defer server.Close()

	provider := newTestOpenAIProvider(t, OpenAICompatibleConfig{
		BaseURL:   server.URL,
		Model:     "embedding-model",
		APIKeyEnv: "MISSING_EMBEDDING_API_KEY",
	})

	_, err := provider.Embed(context.Background(), "hello")
	if err == nil || !strings.Contains(err.Error(), "MISSING_EMBEDDING_API_KEY") {
		t.Fatalf("Embed() error = %v, want missing env error", err)
	}
	if calls != 0 {
		t.Fatalf("provider made %d request(s), want 0", calls)
	}
}

func TestOpenAICompatibleProviderTimeoutIsClear(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-time.After(100 * time.Millisecond):
			writeEmbeddingResponse(t, w, [][]float64{{0.1, 0.2}})
		case <-r.Context().Done():
		}
	}))
	defer server.Close()

	provider := newTestOpenAIProvider(t, OpenAICompatibleConfig{
		BaseURL: server.URL,
		Model:   "embedding-model",
	})
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()

	_, err := provider.Embed(ctx, "hello")
	if err == nil || !strings.Contains(err.Error(), "timed out") {
		t.Fatalf("Embed() error = %v, want timeout", err)
	}
}

func TestOpenAICompatibleProviderDimensionMismatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeEmbeddingResponse(t, w, [][]float64{{0.1, 0.2}})
	}))
	defer server.Close()

	provider := newTestOpenAIProvider(t, OpenAICompatibleConfig{
		BaseURL:           server.URL,
		Model:             "embedding-model",
		ExpectedDimension: 3,
	})

	_, err := provider.Embed(context.Background(), "hello")
	if err == nil || !strings.Contains(err.Error(), "dimension mismatch") {
		t.Fatalf("Embed() error = %v, want dimension mismatch", err)
	}
}

func TestNewOpenAICompatibleProviderValidatesConfig(t *testing.T) {
	cases := []OpenAICompatibleConfig{
		{Model: "embedding-model"},
		{BaseURL: "localhost:8000", Model: "embedding-model"},
		{BaseURL: "http://localhost:8000", Dimensions: 2, ExpectedDimension: 3},
		{BaseURL: "http://localhost:8000"},
	}
	for _, cfg := range cases {
		if _, err := NewOpenAICompatibleProvider(cfg); err == nil {
			t.Fatalf("NewOpenAICompatibleProvider(%+v) returned nil error", cfg)
		}
	}
}

func newTestOpenAIProvider(t *testing.T, cfg OpenAICompatibleConfig) *OpenAICompatibleProvider {
	t.Helper()
	provider, err := NewOpenAICompatibleProvider(cfg)
	if err != nil {
		t.Fatalf("NewOpenAICompatibleProvider() error = %v", err)
	}
	return provider
}

func writeEmbeddingResponse(t *testing.T, w http.ResponseWriter, embeddings [][]float64) {
	t.Helper()
	data := make([]embeddingDatum, len(embeddings))
	for i, embedding := range embeddings {
		data[i] = embeddingDatum{Index: i, Embedding: embedding}
	}
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(embeddingsResponse{Data: data}); err != nil {
		t.Fatalf("encode response: %v", err)
	}
}
