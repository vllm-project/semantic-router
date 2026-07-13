package embedding

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestRetryDelaySaturatesWithoutOverflow(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	tests := []struct {
		attempt int
		want    time.Duration
	}{
		{attempt: -maxInt, want: baseRetryDelay},
		{attempt: 0, want: baseRetryDelay},
		{attempt: 1, want: baseRetryDelay},
		{attempt: 2, want: 2 * baseRetryDelay},
		{attempt: 3, want: 4 * baseRetryDelay},
		{attempt: 4, want: 8 * baseRetryDelay},
		{attempt: 5, want: maxRetryDelay},
		{attempt: maxInt, want: maxRetryDelay},
	}
	for _, tt := range tests {
		if got := retryDelay(tt.attempt); got != tt.want {
			t.Errorf("retryDelay(%d) = %s, want %s", tt.attempt, got, tt.want)
		}
	}
}

func TestOpenAICompatibleProviderEmbedBatch(t *testing.T) {
	server := newEmbeddingBatchServer(t)
	defer server.Close()

	t.Setenv(config.EmbeddingAPIKeyEnvName, "test-key")
	provider := newTestOpenAIProvider(t, OpenAICompatibleConfig{
		BaseURL:           server.URL + "/v1",
		Model:             "BAAI/bge-m3",
		APIKeyEnv:         config.EmbeddingAPIKeyEnvName,
		Dimensions:        2,
		ExpectedDimension: 2,
		HTTPClient:        server.Client(),
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
	return httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
		data[i] = embeddingDatum{Index: intPointer(i), Embedding: embedding}
	}
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(embeddingsResponse{Data: data}); err != nil {
		t.Fatalf("encode response: %v", err)
	}
}

func paddedEmbeddingResponse(t *testing.T, count int, paddingLength int) []byte {
	t.Helper()
	data := make([]embeddingDatum, count)
	for i := range data {
		data[i] = embeddingDatum{Index: intPointer(i), Embedding: []float64{float64(i)}}
	}
	payload, err := json.Marshal(struct {
		Data    []embeddingDatum `json:"data"`
		Padding string           `json:"padding"`
	}{Data: data, Padding: strings.Repeat("x", paddingLength)})
	if err != nil {
		t.Fatalf("marshal padded response: %v", err)
	}
	return payload
}

func assertEmbeddingResponseError(t *testing.T, err error, kind embeddingResponseErrorKind) {
	t.Helper()
	var responseErr *embeddingResponseError
	if !errors.As(err, &responseErr) || responseErr.kind != kind {
		t.Fatalf("error = %v, want embedding response error kind %q", err, kind)
	}
}

type countingReadCloser struct {
	*strings.Reader
	bytesRead int
	sawEOF    bool
}

func (r *countingReadCloser) Read(p []byte) (int, error) {
	n, err := r.Reader.Read(p)
	r.bytesRead += n
	if errors.Is(err, io.EOF) {
		r.sawEOF = true
	}
	return n, err
}

func (r *countingReadCloser) Close() error {
	return nil
}

func intPointer(value int) *int {
	return &value
}
