//go:build !windows

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

// recordingSearchEmbedder answers the refund chunk's vector for every text and
// records the texts it was asked to embed.
type recordingSearchEmbedder struct{ seen []string }

func (e *recordingSearchEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	e.seen = append(e.seen, text)
	return []float32{1, 0, 0}, nil
}

func (e *recordingSearchEmbedder) Dimension() int { return 3 }

func searchLongQuery() string {
	return strings.Repeat("operations handbook preamble text. ", 64) + "how long does a refund take"
}

func vectorStoreSearchFixture(t *testing.T, embedder vectorstore.Embedder) (*ClassificationAPIServer, string) {
	t.Helper()
	ctx := context.Background()
	backend := vectorstore.NewMemoryBackend(vectorstore.MemoryBackendConfig{})
	stores := vectorstore.NewMemoryMetadataRegistry()
	manager := vectorstore.NewManager(backend, stores, 3, vectorstore.BackendTypeMemory)
	store, err := manager.CreateStore(ctx, vectorstore.CreateStoreRequest{Name: "search"})
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

func TestVectorStoreSearchEmbedsTheWholeQueryOnce(t *testing.T) {
	for _, query := range []string{"how long does a refund take", searchLongQuery()} {
		embedder := &recordingSearchEmbedder{}
		server, storeID := vectorStoreSearchFixture(t, embedder)
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
		if len(embedder.seen) != 1 || embedder.seen[0] != query {
			t.Fatalf("embedded %q, want the whole query once", embedder.seen)
		}
		if names := searchFilenames(t, response.Body.Bytes()); len(names) == 0 || names[0] != "refund.txt" {
			t.Fatalf("search ranked %v, want refund.txt first", names)
		}
	}
}

func TestVectorStoreHybridSearchKeepsOneEmbedding(t *testing.T) {
	embedder := &recordingSearchEmbedder{}
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

// preparedRemoteSearchEmbedder implements the whole Provider surface so a test
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

// Set.Get wraps every prepared provider, and search embeds through that wrapper.
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

// countingSearchEmbedder records how often it was asked to embed, so a test can
// show that a rejected query never reaches the model.
type countingSearchEmbedder struct{ calls int }

func (e *countingSearchEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	e.calls++
	return []float32{1, 0, 0}, nil
}

func (e *countingSearchEmbedder) Dimension() int { return 3 }

func TestVectorStoreSearchRejectsABlankQueryAsAClientError(t *testing.T) {
	for name, query := range map[string]string{
		"empty":      "",
		"spaces":     "   ",
		"whitespace": "   \n\t  ",
	} {
		t.Run(name, func(t *testing.T) {
			embedder := &countingSearchEmbedder{}
			server, storeID := vectorStoreSearchFixture(t, embedder)
			body, err := json.Marshal(SearchRequest{Query: query, MaxNumResults: 2})
			if err != nil {
				t.Fatal(err)
			}

			request := httptest.NewRequest(http.MethodPost, apiStorageVectorStoresPath+"/"+storeID+"/search", strings.NewReader(string(body)))
			response := httptest.NewRecorder()
			server.handleSearchVectorStore(response, request)

			if response.Code != http.StatusBadRequest {
				t.Fatalf("blank query answered %d, want 400: %s", response.Code, response.Body.String())
			}
			if code := parseErrorResponse(t, response.Body.Bytes()); code != "INVALID_INPUT" {
				t.Fatalf("blank query reported %q, want INVALID_INPUT", code)
			}
			if embedder.calls != 0 {
				t.Fatalf("a rejected query still reached the embedder %d times", embedder.calls)
			}
		})
	}
}

func TestVectorStoreSearchStillAnswersAQueryWithSurroundingSpace(t *testing.T) {
	embedder := &countingSearchEmbedder{}
	server, storeID := vectorStoreSearchFixture(t, embedder)
	body, err := json.Marshal(SearchRequest{Query: "  how long does a refund take  ", MaxNumResults: 2})
	if err != nil {
		t.Fatal(err)
	}

	request := httptest.NewRequest(http.MethodPost, apiStorageVectorStoresPath+"/"+storeID+"/search", strings.NewReader(string(body)))
	response := httptest.NewRecorder()
	server.handleSearchVectorStore(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("padded query answered %d, want 200: %s", response.Code, response.Body.String())
	}
	if embedder.calls == 0 {
		t.Fatal("a query with real text never reached the embedder")
	}
}
