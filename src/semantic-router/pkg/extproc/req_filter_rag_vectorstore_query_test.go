package extproc

import (
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/match"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

// vectorStoreSource is a rag source over a vector store that recalls
// topK documents at least threshold similar, without a judge.
func vectorStoreSource(storeID string, topK int, threshold float64) *config.ContextSource {
	return &config.ContextSource{
		Type:          config.ContextSourceRAG,
		Backend:       "vectorstore",
		BackendConfig: config.MustStructuredPayload(&config.VectorStoreRAGConfig{VectorStoreID: storeID}),
		Match:         &config.MatchConfig{Recall: &config.MatchRecall{TopK: topK, MinSimilarity: &threshold}},
	}
}

// recalledTexts are the texts of recalled documents, joined as a source
// injects them.
func recalledTexts(documents []match.Scored) string {
	texts := make([]string, len(documents))
	for index, document := range documents {
		texts[index] = document.Text
	}
	return strings.Join(texts, "\n\n---\n\n")
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

		source := vectorStoreSource(storeID, 2, 0.5)
		documents, err := router.recallFromVectorStore(context.Background(), source, query, *source.Match.Recall)
		if err != nil {
			t.Fatal(err)
		}
		retrieved := recalledTexts(documents)
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
	rag := vectorStoreSource(storeID, 2, 0.5)

	documents, err := router.recallFromVectorStore(context.Background(), rag, "how long does a refund take", *rag.Match.Recall)
	if err != nil {
		t.Fatalf("prepared remote embedder failed RAG retrieval: %v", err)
	}
	retrieved := recalledTexts(documents)
	if len(plain.seen) != 1 {
		t.Fatalf("prepared remote embedder was asked to embed %d times, want 1", len(plain.seen))
	}
	if !strings.Contains(retrieved, "refunds take 14 business days") {
		t.Fatalf("prepared remote embedder retrieved %q", retrieved)
	}
}

// file_ids narrows a vectorstore source to the chunks of every file it
// lists, not only the first.
func TestRAGVectorStoreSearchesEveryListedFile(t *testing.T) {
	router, storeID := ragVectorStoreFixture(t, &plainRAGEmbedder{})
	source := vectorStoreSource(storeID, 5, -1)
	source.BackendConfig = config.MustStructuredPayload(&config.VectorStoreRAGConfig{VectorStoreID: storeID, FileIDs: []string{"file_refund", "file_outage"}})

	documents, err := router.recallFromVectorStore(context.Background(), source, "how long does a refund take", *source.Match.Recall)
	if err != nil {
		t.Fatal(err)
	}
	if retrieved := recalledTexts(documents); retrieved != "refunds take 14 business days\n\n---\n\nstatus updates every 30 minutes" {
		t.Fatalf("retrieved %q, want the chunks of both listed files", retrieved)
	}
}
