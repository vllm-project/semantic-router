//go:build !windows && cgo

package cache

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

type ownedCacheTestProvider struct {
	options     embedding.Options
	seenContext context.Context
	calls       int
	closed      bool
}

func (p *ownedCacheTestProvider) Embed(ctx context.Context, text string) ([]float32, error) {
	return p.EmbedWithOptions(ctx, text, embedding.Options{})
}

func (p *ownedCacheTestProvider) EmbedWithOptions(ctx context.Context, _ string, options embedding.Options) ([]float32, error) {
	p.seenContext, p.options = ctx, options
	p.calls++
	return []float32{3, 4}, nil
}

func (p *ownedCacheTestProvider) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	return nil, errors.New("batch not used by cache")
}
func (*ownedCacheTestProvider) Dimension() int  { return 2 }
func (*ownedCacheTestProvider) Backend() string { return "owned-test" }
func (p *ownedCacheTestProvider) Close() error  { p.closed = true; return nil }

func TestOwnedCacheProviderPreservesViewContextAndOwnership(t *testing.T) {
	provider := &ownedCacheTestProvider{}
	view := embedding.WithOptions(provider, embedding.Options{Layer: 6, Dimension: 256})
	cache, err := NewCacheBackend(CacheConfig{Enabled: true, EmbeddingModel: "mmbert", EmbeddingProvider: view})
	if err != nil {
		t.Fatal(err)
	}
	memory := cache.(*InMemoryCache)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	vector, err := memory.generateEmbedding(ctx, "Unicode 中文")
	if err != nil || !reflect.DeepEqual(vector, []float32{3, 4}) {
		t.Fatalf("provider vector changed: %v, %v", vector, err)
	}
	if provider.seenContext != ctx || provider.options != (embedding.Options{Layer: 6, Dimension: 256}) {
		t.Fatal("context or prepared view options lost")
	}
	if err := cache.Close(); err != nil {
		t.Fatal(err)
	}
	if provider.closed {
		t.Fatal("cache closed generation-owned provider")
	}
	cancel()
	if _, err := memory.generateEmbedding(ctx, "Unicode 中文"); !errors.Is(err, context.Canceled) {
		t.Fatalf("memo served cancelled query: %v", err)
	}
	if provider.calls != 1 {
		t.Fatalf("cancelled request reached inference: %d", provider.calls)
	}
}

func TestOwnedCacheProvidersReachEveryBackend(t *testing.T) {
	provider := &ownedCacheTestProvider{}
	view := embedding.WithOptions(provider, embedding.Options{Dimension: 384})
	backends := map[string]func(context.Context, string) ([]float32, error){
		"redis":  (&RedisCache{embeddingProvider: view}).getEmbedding,
		"valkey": (&ValkeyCache{embeddingProvider: view}).getEmbedding,
		"milvus": (&MilvusCache{embeddingProvider: view}).getEmbedding,
		"qdrant": (&QdrantCache{embeddingProvider: view}).getEmbedding,
		"hybrid": (&HybridCache{milvusCache: &MilvusCache{embeddingProvider: view}}).generateEmbedding,
	}
	for name, compute := range backends {
		t.Run(name, func(t *testing.T) {
			if _, err := compute(context.Background(), "text"); err != nil {
				t.Fatal(err)
			}
			if provider.options.Dimension != 384 {
				t.Fatal("prepared embedding dimension lost")
			}
		})
	}
	if _, err := computeCacheEmbedding(context.Background(), nil, "text"); err == nil {
		t.Fatal("unprepared provider accepted")
	}
	options := hybridCacheOptionsFromConfig(CacheConfig{EmbeddingProvider: view})
	if milvusCacheOptionsFromHybridOptions(options).EmbeddingProvider != view {
		t.Fatal("hybrid constructor dropped provider")
	}
}
