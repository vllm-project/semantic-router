//go:build !windows && cgo

package cache

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

type preparedCacheProvider struct {
	ownedCacheTestProvider
	dimension int
}

func (p *preparedCacheProvider) EmbedWithOptions(ctx context.Context, text string, options embedding.Options) ([]float32, error) {
	p.options = options
	return make([]float32, p.dimension), nil
}

func (p *preparedCacheProvider) Windows(context.Context, string, int) ([]embedding.Window, error) {
	return []embedding.Window{{Start: 0, End: 1}}, nil
}

func TestOwnedCachePreparationChecksActualViewBeforePublication(t *testing.T) {
	provider := &preparedCacheProvider{dimension: 128}
	backend := NewInMemoryCache(InMemoryCacheOptions{Enabled: true, EmbeddingModel: "mmbert", EmbeddingProvider: provider})
	defer backend.Close()
	if err := ValidateBackendEmbedding(context.Background(), backend); err == nil {
		t.Fatal("mismatched vector accepted")
	}
	if provider.options != (embedding.Options{Dimension: 256, Layer: 6}) {
		t.Fatalf("actual cache view not checked: %+v", provider.options)
	}
	provider.dimension = 256
	if err := ValidateBackendEmbedding(context.Background(), backend); err != nil {
		t.Fatal(err)
	}
	if provider.closed {
		t.Fatal("validator closed borrowed provider")
	}
}

type failedWindowProvider struct{ ownedCacheTestProvider }

func (*failedWindowProvider) Windows(context.Context, string, int) ([]embedding.Window, error) {
	return nil, context.DeadlineExceeded
}

func TestOwnedCacheWindowFailureSkipsSemanticAndPreservesExact(t *testing.T) {
	provider := &failedWindowProvider{}
	backend := NewInMemoryCache(InMemoryCacheOptions{Enabled: true, EmbeddingModel: "bert", EmbeddingProvider: provider})
	defer backend.Close()
	adapter := NewLegacyBackendAdapter(backend, InMemoryCacheType).WithEmbeddingProvider(provider)
	identity := CacheIdentity{Partition: CachePartition{RequestModel: "model"}, ExactFingerprint: "exact", SemanticQuery: "query"}
	write := CacheWrite{Identity: identity, ResponseBody: []byte("cached"), TTL: DefaultTTL()}
	if err := adapter.StoreExact(context.Background(), write); err != nil {
		t.Fatal(err)
	}
	if err := adapter.StoreSemantic(context.Background(), write); err != nil {
		t.Fatal(err)
	}
	semantic, err := adapter.LookupSemantic(context.Background(), SemanticLookup{Identity: identity})
	if err != nil || semantic.Found || provider.calls != 0 {
		t.Fatalf("failed tokenizer reached semantic inference: %+v %v calls=%d", semantic, err, provider.calls)
	}
	exact, err := adapter.LookupExact(context.Background(), ExactLookup{Identity: identity})
	if err != nil || !exact.Found || string(exact.ResponseBody) != "cached" {
		t.Fatalf("exact cache lost: %+v %v", exact, err)
	}
}
