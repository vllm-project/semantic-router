package cache

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
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
	settings, ok := LocalEmbeddingSettings(memory)
	if !ok || settings.Layer != 6 || settings.Dimension != 256 {
		t.Fatalf("inmemory actual settings: %#v %v", settings, ok)
	}
	persistent := &QdrantCache{embeddingModel: "mmbert", embeddingProvider: storagetest.Vectors{Size: 768}}
	settings, ok = LocalEmbeddingSettings(persistent)
	if !ok || settings.Layer != 0 || settings.Dimension != 768 {
		t.Fatalf("persistent actual settings: %#v %v", settings, ok)
	}
	if _, ok = LocalEmbeddingSettings(NewInMemoryCache(InMemoryCacheOptions{EmbeddingModel: "qwen3"})); ok {
		t.Fatal("unsupported provider received guessed identity")
	}
	bert := NewInMemoryCache(InMemoryCacheOptions{EmbeddingModel: "bert", EmbeddingProvider: storagetest.Vectors{Size: 384}})
	settings, ok = LocalEmbeddingSettings(bert)
	if !ok || settings.Layer != 0 || settings.Dimension != 384 {
		t.Fatalf("bert actual settings: %#v %v", settings, ok)
	}
}
