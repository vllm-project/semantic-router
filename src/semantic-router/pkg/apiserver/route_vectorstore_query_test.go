//go:build !windows && cgo

package apiserver

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

// windowedSearchEmbedder answers one vector per window and records what it was
// asked to embed, so a test can tell whether the tail of a long query was read.
type windowedSearchEmbedder struct {
	windowSize int
	seen       []string
}

func (e *windowedSearchEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	e.seen = append(e.seen, text)
	// The marker is one short token so it cannot straddle a window boundary and
	// leave every window looking alike.
	if strings.Contains(text, searchRefundMarker) {
		return []float32{1, 0, 0}, nil
	}
	// The preamble matches neither stored chunk, so a window that carries no
	// question cannot decide the ranking on its own.
	return []float32{0, 0, 1}, nil
}

// searchRefundMarker is the question's distinctive token. searchLongQuery puts
// it at the end, past the first window, which is what a single embedding misses.
const searchRefundMarker = "REFUND"

func searchLongQuery() string {
	return strings.Repeat("operations handbook preamble text. ", 4) + "how long does a " + searchRefundMarker + " take"
}

func (e *windowedSearchEmbedder) Dimension() int { return 3 }

func (e *windowedSearchEmbedder) Windows(_ context.Context, text string, _ int) ([]embedding.Window, error) {
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

func vectorStoreSearchFixture(t *testing.T, embedder vectorstore.Embedder) (*ClassificationAPIServer, string) {
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
	return &ClassificationAPIServer{runtimeRegistry: registry}, store.ID
}

func searchFilenames(t *testing.T, body []byte) []string {
	t.Helper()
	var response struct {
		Data []struct {
			Filename string `json:"filename"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		t.Fatalf("parse search response: %v (%s)", err, body)
	}
	names := make([]string, 0, len(response.Data))
	for _, hit := range response.Data {
		names = append(names, hit.Filename)
	}
	return names
}

func TestVectorStoreSearchReadsAQuestionPastTheFirstWindow(t *testing.T) {
	embedder := &windowedSearchEmbedder{windowSize: 60}
	server, storeID := vectorStoreSearchFixture(t, embedder)
	query := searchLongQuery()
	body, err := json.Marshal(SearchRequest{Query: query, MaxNumResults: 2})
	if err != nil {
		t.Fatal(err)
	}

	request := httptest.NewRequest(http.MethodPost, apiStorageVectorStoresPath+"/"+storeID+"/search", strings.NewReader(string(body)))
	response := httptest.NewRecorder()
	server.handleSearchVectorStore(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("search failed: %d %s", response.Code, response.Body.String())
	}
	if len(embedder.seen) < 2 {
		t.Fatalf("embedded the query %d times, want one per window", len(embedder.seen))
	}
	names := searchFilenames(t, response.Body.Bytes())
	if len(names) == 0 || names[0] != "refund.txt" {
		t.Fatalf("the question past the first window was not read, hits: %v", names)
	}
}

func TestVectorStoreSearchEmbedsAShortQueryOnce(t *testing.T) {
	embedder := &windowedSearchEmbedder{windowSize: 4096}
	server, storeID := vectorStoreSearchFixture(t, embedder)
	body, err := json.Marshal(SearchRequest{Query: "how long does a refund take", MaxNumResults: 2})
	if err != nil {
		t.Fatal(err)
	}

	request := httptest.NewRequest(http.MethodPost, apiStorageVectorStoresPath+"/"+storeID+"/search", strings.NewReader(string(body)))
	response := httptest.NewRecorder()
	server.handleSearchVectorStore(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("search failed: %d %s", response.Code, response.Body.String())
	}
	if len(embedder.seen) != 1 {
		t.Fatalf("a query inside one window was embedded %d times", len(embedder.seen))
	}
}

func TestVectorStoreHybridSearchKeepsOneEmbedding(t *testing.T) {
	embedder := &windowedSearchEmbedder{windowSize: 60}
	server, storeID := vectorStoreSearchFixture(t, embedder)
	query := searchLongQuery()
	body, err := json.Marshal(SearchRequest{Query: query, MaxNumResults: 2, Hybrid: &vectorstore.HybridSearchConfig{}})
	if err != nil {
		t.Fatal(err)
	}

	request := httptest.NewRequest(http.MethodPost, apiStorageVectorStoresPath+"/"+storeID+"/search", strings.NewReader(string(body)))
	response := httptest.NewRecorder()
	server.handleSearchVectorStore(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("hybrid search failed: %d %s", response.Code, response.Body.String())
	}
	if len(embedder.seen) != 1 {
		t.Fatalf("hybrid search embedded %d times, want the whole query once", len(embedder.seen))
	}
	if embedder.seen[0] != query {
		t.Fatalf("hybrid search embedded %q, want the whole query", embedder.seen[0])
	}
}

// preparedRemoteSearchEmbedder has no token windows of its own, the way a remote
// embedding service does not. It implements the whole Provider surface so a test
// can hand it to NewSet and receive the wrapper the runtime would build.
type preparedRemoteSearchEmbedder struct{ calls int }

func (e *preparedRemoteSearchEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	e.calls++
	return []float32{1, 0, 0}, nil
}

func (e *preparedRemoteSearchEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
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

func (e *preparedRemoteSearchEmbedder) Dimension() int { return 3 }

func (e *preparedRemoteSearchEmbedder) Backend() string { return "remote" }

// Set.Get wraps every prepared provider, and the wrapper satisfies WindowProvider
// whether or not the model behind it can tokenize, so a remote embedder only
// reveals the missing capability when Windows is called.
func TestVectorStoreSearchAnswersWithAPreparedRemoteEmbedder(t *testing.T) {
	plain := &preparedRemoteSearchEmbedder{}
	set := embedding.NewSet(map[string]embedding.Provider{"remote": plain}, "remote")
	prepared, err := set.Get("remote", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	wrapped, ok := prepared.(vectorstore.Embedder)
	if !ok {
		t.Fatalf("prepared provider %T does not satisfy vectorstore.Embedder", prepared)
	}

	server, storeID := vectorStoreSearchFixture(t, wrapped)
	body, err := json.Marshal(SearchRequest{Query: "how long does a refund take", MaxNumResults: 2})
	if err != nil {
		t.Fatal(err)
	}

	request := httptest.NewRequest(http.MethodPost, apiStorageVectorStoresPath+"/"+storeID+"/search", strings.NewReader(string(body)))
	response := httptest.NewRecorder()
	server.handleSearchVectorStore(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("prepared remote embedder answered %d, want 200: %s", response.Code, response.Body.String())
	}
	if plain.calls != 1 {
		t.Fatalf("prepared remote embedder was asked to embed %d times, want 1", plain.calls)
	}
}
