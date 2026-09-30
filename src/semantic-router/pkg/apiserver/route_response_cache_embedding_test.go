//go:build !windows && cgo

package apiserver

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

type responseCacheDiagnosticProvider struct {
	calls int
}

func (*responseCacheDiagnosticProvider) Dimension() int  { return 2 }
func (*responseCacheDiagnosticProvider) Backend() string { return "test" }

func (p *responseCacheDiagnosticProvider) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	vectors := make([][]float32, len(texts))
	for i, text := range texts {
		vector, err := p.Embed(ctx, text)
		if err != nil {
			return nil, err
		}
		vectors[i] = vector
	}
	return vectors, nil
}

func (p *responseCacheDiagnosticProvider) Embed(context.Context, string) ([]float32, error) {
	p.calls++
	return []float32{1, 0}, nil
}

func (*responseCacheDiagnosticProvider) Windows(context.Context, string, int) ([]embedding.Window, error) {
	return []embedding.Window{{Start: 0, End: 1}}, nil
}

func TestResponseCacheDiagnosticsBorrowActualServiceWithoutDefaultEmbedding(t *testing.T) {
	provider := &responseCacheDiagnosticProvider{}
	backend := cache.NewInMemoryCache(cache.InMemoryCacheOptions{Enabled: true, EmbeddingModel: "bert", EmbeddingProvider: provider})
	t.Cleanup(func() { _ = backend.Close() })
	adapter := cache.NewLegacyBackendAdapter(backend, cache.InMemoryCacheType).WithEmbeddingModel("bert").WithEmbeddingProvider(provider)
	server := &ClassificationAPIServer{responseCache: cache.NewResponseCacheService(adapter, cache.DefaultResponseCacheServiceOptions())}
	request := httptest.NewRequest(http.MethodPost, "/api/v1/storage/response-cache/test", strings.NewReader(`{"configuration":{"backend_type":"memory","enabled":true,"embedding_model":"bert"}}`))
	recorder := httptest.NewRecorder()
	server.handleResponseCacheTest(recorder, request)
	if recorder.Code != http.StatusOK || provider.calls == 0 {
		t.Fatalf("diagnostic did not use the live cache consumer: %d / %s / calls=%d", recorder.Code, recorder.Body.String(), provider.calls)
	}
	if _, err := server.responseCache.PreparedEmbedding("mmbert"); err == nil {
		t.Fatal("a different candidate model borrowed the cache consumer")
	}
}
