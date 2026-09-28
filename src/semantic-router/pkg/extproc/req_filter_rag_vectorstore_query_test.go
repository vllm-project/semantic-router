package extproc

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

// windowedRAGEmbedder answers one vector per window and records the text each
// search was given, so a test can tell which part of a long query was read.
type windowedRAGEmbedder struct {
	windowSize int
	seen       []string
}

func (e *windowedRAGEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	e.seen = append(e.seen, text)
	// The marker is one short token so it cannot straddle a window boundary and
	// leave every window looking alike.
	if strings.Contains(text, refundMarker) {
		return []float32{1, 0, 0}, nil
	}
	// The preamble matches neither stored chunk, so a window that carries no
	// question cannot decide the ranking on its own.
	return []float32{0, 0, 1}, nil
}

// refundMarker is the question's distinctive token. ragLongQuery places it at
// the end, past the first window, which is what a single embedding misses.
const refundMarker = "REFUND"

func ragLongQuery() string {
	return strings.Repeat("operations handbook preamble text. ", 4) + "how long does a " + refundMarker + " take"
}

func (e *windowedRAGEmbedder) Dimension() int { return 3 }

func (e *windowedRAGEmbedder) Windows(_ context.Context, text string, _ int) ([]embedding.Window, error) {
	if e.windowSize <= 0 {
		return nil, fmt.Errorf("windowedRAGEmbedder needs a positive window size")
	}
	var windows []embedding.Window
	for start := 0; start < len(text); start += e.windowSize {
		end := start + e.windowSize
		if end > len(text) {
			end = len(text)
		}
		windows = append(windows, embedding.Window{Start: start, End: end})
	}
	return windows, nil
}

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
	store, err := manager.CreateStore(ctx, vectorstore.CreateStoreRequest{Name: "windows"})
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

func TestRAGVectorStoreSearchesEveryWindowOfALongQuery(t *testing.T) {
	embedder := &windowedRAGEmbedder{windowSize: 60}
	router, storeID := ragVectorStoreFixture(t, embedder)
	rag := ragPluginConfig(storeID, 2, 0.5)
	// The question sits past the first window, which is what a single embedding
	// of the whole query would miss.
	query := ragLongQuery()

	retrieved, err := router.retrieveFromVectorStore(context.Background(), &RequestContext{UserContent: query}, rag)
	if err != nil {
		t.Fatal(err)
	}
	if len(embedder.seen) < 2 {
		t.Fatalf("embedded the query %d times, want one per window", len(embedder.seen))
	}
	if !strings.Contains(retrieved, "refunds take 14 business days") {
		t.Fatalf("the question past the first window was not read: %q", retrieved)
	}
}

func TestRAGVectorStoreEmbedsAShortQueryOnce(t *testing.T) {
	embedder := &windowedRAGEmbedder{windowSize: 4096}
	router, storeID := ragVectorStoreFixture(t, embedder)
	rag := ragPluginConfig(storeID, 2, 0.5)

	if _, err := router.retrieveFromVectorStore(context.Background(), &RequestContext{UserContent: "how long does a refund take"}, rag); err != nil {
		t.Fatal(err)
	}
	if len(embedder.seen) != 1 {
		t.Fatalf("a query inside one window was embedded %d times", len(embedder.seen))
	}
}

// plainRAGEmbedder has no token windows of its own, the way a remote embedding
// service does not.
type plainRAGEmbedder struct{ calls int }

func (e *plainRAGEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	e.calls++
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

// The runtime hands RAG a prepared provider, which satisfies WindowProvider
// through the wrapper even when the model behind it cannot tokenize locally.
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
	if plain.calls != 1 {
		t.Fatalf("prepared remote embedder was asked to embed %d times, want 1", plain.calls)
	}
	if !strings.Contains(retrieved, "refunds take 14 business days") {
		t.Fatalf("prepared remote embedder retrieved %q", retrieved)
	}
}
