//go:build !windows

package cache

import (
	"context"
	"strings"
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

// truncatingProvider reports every query longer than two words as truncated,
// or fails the check when err is set.
type truncatingProvider struct {
	ownedCacheTestProvider
	checked []embedding.Options
	err     error
}

func (p *truncatingProvider) FitsInputWithOptions(_ context.Context, text string, options embedding.Options) (bool, error) {
	p.checked = append(p.checked, options)
	return len(strings.Fields(text)) <= 2, p.err
}

func TestOwnedCacheKeepsTruncatedQueriesOutOfSemanticEntries(t *testing.T) {
	provider := &truncatingProvider{}
	backend := NewInMemoryCache(InMemoryCacheOptions{Enabled: true, EmbeddingModel: "mmbert", EmbeddingProvider: provider, SimilarityThreshold: 0.5})
	defer backend.Close()
	adapter := NewLegacyBackendAdapter(backend, InMemoryCacheType).WithEmbeddingProvider(provider)
	for query, stored := range map[string]bool{"short query": true, "a query the model reads only in part": false} {
		identity := CacheIdentity{Partition: CachePartition{RequestModel: "model"}, SemanticQuery: query}
		calls := provider.calls
		if err := adapter.StoreSemantic(context.Background(), CacheWrite{Identity: identity, ResponseBody: []byte("cached"), TTL: DefaultTTL()}); err != nil {
			t.Fatal(err)
		}
		if embedded := provider.calls > calls; embedded != stored {
			t.Fatalf("%q: embedded for storage = %v, want %v", query, embedded, stored)
		}
		semantic, err := adapter.LookupSemantic(context.Background(), SemanticLookup{Identity: identity, Threshold: 0.5})
		if err != nil || semantic.Found != stored {
			t.Fatalf("%q: semantic lookup %+v %v", query, semantic, err)
		}
	}
	for _, options := range provider.checked {
		if options != (embedding.Options{Dimension: 256, Layer: 6}) {
			t.Fatalf("truncation checked at %+v, not the cache's view", options)
		}
	}
}

func TestOwnedCacheTruncationCheckFailureSkipsSemanticAndPreservesExact(t *testing.T) {
	provider := &truncatingProvider{err: context.DeadlineExceeded}
	backend := NewInMemoryCache(InMemoryCacheOptions{Enabled: true, EmbeddingModel: "qwen3", EmbeddingProvider: provider})
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
		t.Fatalf("failed truncation check reached semantic inference: %+v %v calls=%d", semantic, err, provider.calls)
	}
	exact, err := adapter.LookupExact(context.Background(), ExactLookup{Identity: identity})
	if err != nil || !exact.Found || string(exact.ResponseBody) != "cached" {
		t.Fatalf("exact cache lost: %+v %v", exact, err)
	}
}
