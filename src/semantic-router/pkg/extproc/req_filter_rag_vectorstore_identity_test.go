package extproc

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

type forbiddenRAGIdentityEmbedder struct{ t *testing.T }

func (e forbiddenRAGIdentityEmbedder) Embed(context.Context, string) ([]float32, error) {
	e.t.Fatal("RAG embedded a query against incompatible stored vectors")
	return nil, errors.New("unexpected embedding")
}
func (forbiddenRAGIdentityEmbedder) Dimension() int { return 3 }

func TestRAGVectorStoreRejectsHistoricalEmbeddingIdentity(t *testing.T) {
	ctx := context.Background()
	backend := vectorstore.NewMemoryBackend(vectorstore.MemoryBackendConfig{})
	stores := vectorstore.NewMemoryMetadataRegistry()
	old := vectorstore.NewManager(backend, stores, 3, vectorstore.BackendTypeMemory, vectorstore.WithEmbeddingIdentity("old-model"))
	vs, err := old.CreateStore(ctx, vectorstore.CreateStoreRequest{Name: "historical"})
	if err != nil {
		t.Fatal(err)
	}
	current := vectorstore.NewManager(backend, stores, 3, vectorstore.BackendTypeMemory, vectorstore.WithEmbeddingIdentity("new-model"))
	if err = current.LoadFromRegistry(ctx); err != nil {
		t.Fatal(err)
	}
	registry := routerruntime.NewRegistry(nil)
	registry.SetVectorStoreRuntime(&routerruntime.VectorStoreRuntime{Manager: old, Embedder: forbiddenRAGIdentityEmbedder{t}})
	router := &OpenAIRouter{RuntimeRegistry: registry}
	rag := &config.RAGPluginConfig{Enabled: true, Backend: "vectorstore", CacheResults: true, BackendConfig: config.MustStructuredPayload(&config.VectorStoreRAGConfig{VectorStoreID: vs.ID})}
	router.setRAGCache(config.DefaultRecipeName, "source", "cached historical context", rag)
	if cached, hit := router.getRAGCache(config.DefaultRecipeName, "source", rag); !hit || cached != "cached historical context" {
		t.Fatal("compatible cache fixture missed")
	}
	other, err := old.CreateStore(ctx, vectorstore.CreateStoreRequest{Name: "another collection"})
	if err != nil {
		t.Fatal(err)
	}
	otherRAG := *rag
	otherRAG.BackendConfig = config.MustStructuredPayload(&config.VectorStoreRAGConfig{VectorStoreID: other.ID})
	if _, hit := router.getRAGCache(config.DefaultRecipeName, "source", &otherRAG); hit {
		t.Fatal("RAG cache mixed different vector stores")
	}
	registry.SetVectorStoreRuntime(&routerruntime.VectorStoreRuntime{Manager: current, Embedder: forbiddenRAGIdentityEmbedder{t}})
	if _, hit := router.getRAGCache(config.DefaultRecipeName, "source", rag); hit {
		t.Fatal("cached RAG context bypassed model compatibility")
	}
	hybrid := &config.RAGPluginConfig{Backend: "hybrid", CacheResults: true, BackendConfig: config.MustStructuredPayload(&config.HybridRAGConfig{Primary: "vectorstore", PrimaryConfig: rag.BackendConfig})}
	if key := router.buildRAGCacheKey(config.DefaultRecipeName, "source", hybrid); key != "" {
		t.Fatal("hybrid RAG can cache an incompatible child")
	}
	text, err := router.retrieveFromVectorStore(ctx, &RequestContext{UserContent: "source"}, rag)
	if text != "" || !errors.Is(err, vectorstore.ErrEmbeddingIncompatible) {
		t.Fatalf("RAG used incompatible representations: %q, %v", text, err)
	}
	text, err = router.retrieveContext(ctx, &RequestContext{UserContent: "source"}, rag)
	if text != "" || !errors.Is(err, vectorstore.ErrEmbeddingIncompatible) {
		t.Fatalf("RAG result cache bypassed compatibility: %q, %v", text, err)
	}
}
