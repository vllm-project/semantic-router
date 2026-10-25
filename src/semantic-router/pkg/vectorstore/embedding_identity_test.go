package vectorstore

import (
	"context"
	"errors"
	"testing"
)

func TestEmbeddingIdentitySurvivesReopenAndProtectsHistoricalStores(t *testing.T) {
	testEmbeddingIdentityLifecycle(t, NewMemoryMetadataRegistry())
}

func testEmbeddingIdentityLifecycle(t *testing.T, registry StoreRegistry) {
	t.Helper()
	ctx := context.Background()
	backend := NewMemoryBackend(MemoryBackendConfig{})
	old := NewManager(backend, registry, 3, BackendTypeMemory, WithEmbeddingIdentity("model-a"))
	store, err := old.CreateStore(ctx, CreateStoreRequest{Name: "historical", Metadata: map[string]interface{}{"project": "alpha"}})
	if err != nil {
		t.Fatal(err)
	}
	chunk := EmbeddedChunk{ID: "chunk", FileID: "original-file", Content: "original text", Embedding: []float32{1, 0, 0}}
	if err = old.InsertChunks(ctx, store.ID, []EmbeddedChunk{chunk}); err != nil {
		t.Fatal(err)
	}
	for _, identity := range []string{"model-a", "model-b", ""} {
		reopened := NewManager(backend, registry, 3, BackendTypeMemory, WithEmbeddingIdentity(identity))
		if err = reopened.LoadFromRegistry(ctx); err != nil {
			t.Fatal(err)
		}
		if _, err = reopened.GetStore(store.ID); err != nil {
			t.Fatalf("historical store disappeared: %v", err)
		}
		results, searchErr := reopened.Search(ctx, store.ID, chunk.Embedding, 1, 0, nil)
		if identity == "model-a" {
			if searchErr != nil || len(results) != 1 || results[0].Content != chunk.Content {
				t.Fatalf("same identity failed after reopen: %v, %v", results, searchErr)
			}
			continue
		}
		if !errors.Is(searchErr, ErrEmbeddingIncompatible) {
			t.Fatalf("identity %q reused old vectors: %v", identity, searchErr)
		}
		if err = reopened.InsertChunks(ctx, store.ID, []EmbeddedChunk{chunk}); !errors.Is(err, ErrEmbeddingIncompatible) {
			t.Fatalf("identity %q appended to old store: %v", identity, err)
		}
		if _, err = reopened.HybridSearch(ctx, store.ID, "original", chunk.Embedding, 1, 0, nil, &HybridSearchConfig{}); !errors.Is(err, ErrEmbeddingIncompatible) {
			t.Fatalf("hybrid search bypassed identity: %v", err)
		}
	}
	updated, err := old.UpdateStore(ctx, store.ID, UpdateStoreRequest{Metadata: map[string]interface{}{"project": "beta"}})
	if err != nil || updated.Metadata[EmbeddingIdentityMetadataKey] != "model-a" || updated.Metadata["project"] != "beta" {
		t.Fatalf("client metadata replaced identity: %v, %v", updated, err)
	}
	for _, value := range []interface{}{"model-b", nil, 1} {
		metadata := map[string]interface{}{EmbeddingIdentityMetadataKey: value}
		if _, err = old.CreateStore(ctx, CreateStoreRequest{Metadata: metadata}); err == nil {
			t.Fatal("client supplied a reserved identity")
		}
		if _, err = old.UpdateStore(ctx, store.ID, UpdateStoreRequest{Metadata: metadata}); err == nil {
			t.Fatal("client changed a reserved identity")
		}
	}
	legacy := NewManager(backend, registry, 3, BackendTypeMemory)
	untagged, err := legacy.CreateStore(ctx, CreateStoreRequest{Name: "untagged"})
	if err != nil {
		t.Fatal(err)
	}
	current := NewManager(backend, registry, 3, BackendTypeMemory, WithEmbeddingIdentity("model-b"))
	if err = current.LoadFromRegistry(ctx); err != nil {
		t.Fatal(err)
	}
	if err = current.CheckEmbeddingCompatibility(untagged.ID); !errors.Is(err, ErrEmbeddingIncompatible) {
		t.Fatalf("untagged store was adopted: %v", err)
	}
	fresh, err := current.CreateStore(ctx, CreateStoreRequest{Name: "reembedded"})
	if err != nil {
		t.Fatal(err)
	}
	if err := current.InsertChunks(ctx, fresh.ID, []EmbeddedChunk{chunk}); err != nil {
		t.Fatal(err)
	}
	if results, err := current.Search(ctx, fresh.ID, chunk.Embedding, 1, 0, nil); err != nil || len(results) != 1 {
		t.Fatalf("new representation cannot ingest/search: %v, %v", results, err)
	}
	// Model migration must not delete old collections or their original text.
	if results, err := backend.Search(ctx, store.ID, chunk.Embedding, 1, 0, nil); err != nil || len(results) != 1 || results[0].Content != chunk.Content {
		t.Fatalf("historical data changed: %v, %v", results, err)
	}
	if err := current.DeleteStore(ctx, store.ID); err != nil {
		t.Fatalf("explicit deletion of incompatible store failed: %v", err)
	}
}

func TestEmbeddingIdentityRejectsRemoteQuerySpace(t *testing.T) {
	backend, err := NewLlamaStackBackend(LlamaStackBackendConfig{Endpoint: "http://127.0.0.1:1"})
	if err != nil {
		t.Fatal(err)
	}
	manager := NewManager(backend, NewMemoryMetadataRegistry(), 768, BackendTypeLlamaStack, WithEmbeddingIdentity("local-model"))
	if _, err := manager.CreateStore(context.Background(), CreateStoreRequest{}); !errors.Is(err, ErrEmbeddingIncompatible) {
		t.Fatalf("remote query embedding was accepted as local: %v", err)
	}
}
