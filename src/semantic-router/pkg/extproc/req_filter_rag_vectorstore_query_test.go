package extproc

import (
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

func ragPluginConfig(storeID string, topK int, threshold float32) *config.RAGPluginConfig {
	return &config.RAGPluginConfig{
		Enabled:             true,
		Backend:             "vectorstore",
		TopK:                &topK,
		SimilarityThreshold: &threshold,
		BackendConfig:       config.MustStructuredPayload(&config.VectorStoreRAGConfig{VectorStoreID: storeID}),
	}
}

func ragVectorStoreFixture(t *testing.T, embedder vectorstore.Embedder) (*OpenAIRouter, string) {
	t.Helper()
	ctx := context.Background()
	backend := vectorstore.NewMemoryBackend(vectorstore.MemoryBackendConfig{})
	stores := vectorstore.NewMemoryMetadataRegistry()
	manager := vectorstore.NewManager(backend, stores, 3, vectorstore.BackendTypeMemory)
	store, err := manager.CreateStore(ctx, vectorstore.CreateStoreRequest{Name: "rag"})
	if err != nil {
		t.Fatal(err)
	}
	// Each chunk needs its own ID: the memory backend keys its map on it, so
	// chunks sharing an empty ID would overwrite one another.
	chunks := []vectorstore.EmbeddedChunk{
		{ID: "chunk_refund", VectorStoreID: store.ID, FileID: "file_refund", Filename: "refund.txt", ChunkIndex: 0, Content: "refunds take 14 business days", Embedding: []float32{1, 0, 0}},
		{ID: "chunk_outage", VectorStoreID: store.ID, FileID: "file_outage", Filename: "outage.txt", ChunkIndex: 0, Content: "status updates every 30 minutes", Embedding: []float32{0, 1, 0}},
	}
	if err = manager.InsertChunks(ctx, store.ID, chunks); err != nil {
		t.Fatal(err)
	}
	registry := routerruntime.NewRegistry(nil)
	registry.SetVectorStoreRuntime(&routerruntime.VectorStoreRuntime{Manager: manager, Embedder: embedder})
	return &OpenAIRouter{RuntimeRegistry: registry}, store.ID
}

// plainRAGEmbedder answers the refund chunk's vector for every text and
// records the texts it was asked to embed.
type plainRAGEmbedder struct{ seen []string }

func (e *plainRAGEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	e.seen = append(e.seen, text)
	return []float32{1, 0, 0}, nil
}

func (e *plainRAGEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	vectors := make([][]float32, 0, len(texts))
	for _, text := range texts {
		vector, err := e.Embed(ctx, text)
		if err != nil {
			return nil, err
		}
		vectors = append(vectors, vector)
	}
	return vectors, nil
}

func (e *plainRAGEmbedder) Dimension() int { return 3 }

func (e *plainRAGEmbedder) Backend() string { return "remote" }

func TestRAGVectorStoreEmbedsTheWholeQueryOnce(t *testing.T) {
	for _, query := range []string{
		"how long does a refund take",
		strings.Repeat("operations handbook preamble text. ", 64) + "how long does a refund take",
	} {
		embedder := &plainRAGEmbedder{}
		router, storeID := ragVectorStoreFixture(t, embedder)

		retrieved, err := router.retrieveFromVectorStore(context.Background(), &RequestContext{UserContent: query}, ragPluginConfig(storeID, 2, 0.5))
		if err != nil {
			t.Fatal(err)
		}
		if len(embedder.seen) != 1 || embedder.seen[0] != query {
			t.Fatalf("embedded %q, want the whole query once", embedder.seen)
		}
		if retrieved != "refunds take 14 business days" {
			t.Fatalf("retrieved %q, want only the chunk above the threshold", retrieved)
		}
	}
}

// The runtime hands RAG the prepared provider set's view of a model, not the
// model's provider itself.
func TestRAGVectorStoreRetrievesWithAPreparedRemoteEmbedder(t *testing.T) {
	plain := &plainRAGEmbedder{}
	set := embedding.NewSet(map[string]embedding.Provider{"remote": plain}, "remote")
	prepared, err := set.Get("remote", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	wrapped, ok := prepared.(vectorstore.Embedder)
	if !ok {
		t.Fatalf("prepared provider %T does not satisfy vectorstore.Embedder", prepared)
	}

	router, storeID := ragVectorStoreFixture(t, wrapped)
	rag := ragPluginConfig(storeID, 2, 0.5)

	retrieved, err := router.retrieveFromVectorStore(context.Background(), &RequestContext{UserContent: "how long does a refund take"}, rag)
	if err != nil {
		t.Fatalf("prepared remote embedder failed RAG retrieval: %v", err)
	}
	if len(plain.seen) != 1 {
		t.Fatalf("prepared remote embedder was asked to embed %d times, want 1", len(plain.seen))
	}
	if !strings.Contains(retrieved, "refunds take 14 business days") {
		t.Fatalf("prepared remote embedder retrieved %q", retrieved)
	}
}
