package cache

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

type persistedSemanticRecord struct {
	Partition string
	Body      []byte
}

// A disk-backed L2 surrogate deliberately treats every vector as a nearest
// neighbor. Only the real partition passed by ResponseCacheService isolates it.
type persistentIdentityStore struct {
	*serviceTestStore
	path string
}

func (s *persistentIdentityStore) records() []persistedSemanticRecord {
	raw, err := os.ReadFile(s.path)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		panic(err)
	}
	var records []persistedSemanticRecord
	if err = json.Unmarshal(raw, &records); err != nil {
		panic(err)
	}
	return records
}

func (s *persistentIdentityStore) LookupSemantic(_ context.Context, query SemanticLookup) (CacheResult, error) {
	for _, record := range s.records() {
		if record.Partition == query.Identity.Partition.Key() {
			return CacheResult{Found: true, ResponseBody: record.Body, Similarity: 1}, nil
		}
	}
	return CacheResult{}, nil
}

func (s *persistentIdentityStore) StoreSemantic(_ context.Context, write CacheWrite) error {
	records := append(s.records(), persistedSemanticRecord{write.Identity.Partition.Key(), write.ResponseBody})
	raw, err := json.Marshal(records)
	if err != nil {
		return err
	}
	return os.WriteFile(s.path, raw, 0o600)
}

func TestEmbeddingIdentityIsolatesPersistentSemanticCacheWithoutDeletingLegacy(t *testing.T) {
	path := filepath.Join(t.TempDir(), "persistent-l2.json")
	newService := func(identity string) *ResponseCacheService {
		return NewResponseCacheService(&persistentIdentityStore{newServiceTestStore(), path}, ResponseCacheServiceOptions{EmbeddingIdentity: identity})
	}
	ctx := context.Background()
	request := serviceTestIdentity("same-query")
	request.Partition.Namespace = "existing-tenant-namespace"
	request.Partition.Epoch = "explicit-plugin-revision"
	for _, space := range []string{"", "old-model"} {
		if err := newService(space).StoreSemantic(ctx, CacheWrite{Identity: request, ResponseBody: []byte(space + " response")}); err != nil {
			t.Fatal(err)
		}
	}
	current := newService("new-model")
	result, err := current.LookupSemantic(ctx, SemanticLookup{Identity: request})
	if err != nil || result.Found {
		t.Fatalf("old/untagged vector adopted: %#v, %v", result, err)
	}
	// A resolved lease from an old service must not carry the old epoch through.
	oldResolved := newService("old-model").ResolveIdentity(request)
	if result, err = current.LookupSemantic(ctx, SemanticLookup{Identity: oldResolved}); err != nil || result.Found {
		t.Fatalf("old resolved identity bypassed isolation: %#v %v", result, err)
	}
	if err = current.StoreSemantic(ctx, CacheWrite{Identity: request, ResponseBody: []byte("new response")}); err != nil {
		t.Fatal(err)
	}
	// A fresh service has an empty L1 and reopens the persistent data.
	result, err = newService("new-model").LookupSemantic(ctx, SemanticLookup{Identity: request})
	if err != nil || !result.Found || string(result.ResponseBody) != "new response" {
		t.Fatalf("current space did not survive reopen: %#v %v", result, err)
	}
	for _, space := range []string{"", "old-model"} {
		result, err = newService(space).LookupSemantic(ctx, SemanticLookup{Identity: request})
		if err != nil || !result.Found || string(result.ResponseBody) != space+" response" {
			t.Fatalf("old data was destroyed: %#v %v", result, err)
		}
	}
	resolved := current.ResolveIdentity(request)
	if resolved.Partition.Namespace != request.Partition.Namespace {
		t.Fatal("user namespace changed")
	}
	if current.ResolveIdentity(resolved).Partition.Key() != resolved.Partition.Key() {
		t.Fatal("identity was applied twice")
	}
	if len((&persistentIdentityStore{path: path}).records()) != 3 {
		t.Fatal("persistent records changed unexpectedly")
	}
}

func TestCacheEmbeddingSettingsReflectActualBackend(t *testing.T) {
	memory := NewInMemoryCache(InMemoryCacheOptions{EmbeddingModel: "mmbert"})
	settings, ok, err := LocalEmbeddingSettings(memory)
	if err != nil || !ok || settings.Layer != 6 || settings.Dimension != 256 {
		t.Fatalf("inmemory actual settings: %#v %v %v", settings, ok, err)
	}
	persistent := &QdrantCache{
		embeddingModel:    "mmbert",
		embeddingProvider: &cacheIdentityContractProvider{},
	}
	settings, ok, err = LocalEmbeddingSettings(persistent)
	if err != nil || !ok || settings.Layer != 0 || settings.Dimension != 768 {
		t.Fatalf("persistent actual settings: %#v %v %v", settings, ok, err)
	}
	if _, ok, err := LocalEmbeddingSettings(NewInMemoryCache(InMemoryCacheOptions{EmbeddingModel: "bert"})); err != nil || ok {
		t.Fatal("unsupported provider received guessed identity")
	}
}

type cacheIdentityContractProvider struct {
	err error
}

func (p *cacheIdentityContractProvider) Embed(context.Context, string) ([]float32, error) {
	return nil, errors.New("embedding not used")
}

func (p *cacheIdentityContractProvider) EmbedBatch(context.Context, []string) ([][]float32, error) {
	return nil, errors.New("embedding not used")
}

func (*cacheIdentityContractProvider) Dimension() int { return 768 }

func (*cacheIdentityContractProvider) Backend() string { return "cache-identity-test" }

func (p *cacheIdentityContractProvider) EmbeddingDimensionContract() (embedding.DimensionContract, error) {
	if p.err != nil {
		return embedding.DimensionContract{}, p.err
	}
	return embedding.DimensionContract{NativeDimension: 768, SupportedDimensions: []int{256, 512, 768}}, nil
}

func TestCacheIdentityAndNamespacePropagateDimensionErrors(t *testing.T) {
	provider := &cacheIdentityContractProvider{err: errors.New("model is not loaded")}
	backend := &RedisCache{
		embeddingModel:    "mmbert",
		embeddingProvider: provider,
		config:            &config.RedisConfig{},
	}
	if _, supported, err := LocalEmbeddingSettings(backend); err == nil || supported {
		t.Fatalf("dimension error was not returned: supported=%v err=%v", supported, err)
	}
	milvusBackend := &MilvusCache{
		embeddingModel:    "mmbert",
		embeddingProvider: provider,
		config:            &config.MilvusConfig{},
	}
	if _, supported, err := LocalEmbeddingSettings(milvusBackend); err == nil || supported {
		t.Fatalf("Milvus dimension error was not returned: supported=%v err=%v", supported, err)
	}

	for _, backendType := range []CacheBackendType{RedisCacheType, MilvusCacheType, HybridCacheType, QdrantCacheType} {
		t.Run(string(backendType), func(t *testing.T) {
			cfg := namespaceFixture(backendType, 0)
			cfg.EmbeddingProvider = provider
			if _, _, err := PrepareEmbeddingNamespace(cfg, func(embedding.ConsumerSettings) (embedding.ContentIdentity, error) {
				t.Fatal("identity resolver called after dimension failure")
				return embedding.ContentIdentity{}, nil
			}); err == nil {
				t.Fatal("namespace preparation swallowed dimension error")
			}
		})
	}
}
