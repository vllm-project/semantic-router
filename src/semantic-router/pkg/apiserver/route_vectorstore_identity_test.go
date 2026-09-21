//go:build !windows && cgo

package apiserver

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

type forbiddenIdentityEmbedder struct{ t *testing.T }

func (e forbiddenIdentityEmbedder) Embed(context.Context, string) ([]float32, error) {
	e.t.Fatal("incompatible vector store triggered embedding work")
	return nil, errors.New("unexpected embedding")
}
func (forbiddenIdentityEmbedder) Dimension() int { return 3 }

func TestVectorStoreSearchRejectsOldIdentityBeforeEmbedding(t *testing.T) {
	ctx := context.Background()
	backend := vectorstore.NewMemoryBackend(vectorstore.MemoryBackendConfig{})
	stores := vectorstore.NewMemoryMetadataRegistry()
	old := vectorstore.NewManager(backend, stores, 3, vectorstore.BackendTypeMemory, vectorstore.WithEmbeddingIdentity("old-model"))
	vs, err := old.CreateStore(ctx, vectorstore.CreateStoreRequest{Name: "old"})
	if err != nil {
		t.Fatal(err)
	}
	current := vectorstore.NewManager(backend, stores, 3, vectorstore.BackendTypeMemory, vectorstore.WithEmbeddingIdentity("new-model"))
	if err := current.LoadFromRegistry(ctx); err != nil {
		t.Fatal(err)
	}
	registry := routerruntime.NewRegistry(nil)
	registry.SetVectorStoreRuntime(&routerruntime.VectorStoreRuntime{Manager: current, Embedder: forbiddenIdentityEmbedder{t}})
	server := &ClassificationAPIServer{runtimeRegistry: registry}
	for _, body := range []string{`{"query":"source"}`, `{"query":"source","hybrid":{"enabled":true}}`} {
		request := httptest.NewRequest(http.MethodPost, apiStorageVectorStoresPath+"/"+vs.ID+"/search", strings.NewReader(body))
		response := httptest.NewRecorder()
		server.handleSearchVectorStore(response, request)
		if response.Code != http.StatusConflict || parseErrorResponse(t, response.Body.Bytes()) != "EMBEDDING_REINDEX_REQUIRED" {
			t.Fatalf("incompatible search response: %d %s", response.Code, response.Body.String())
		}
	}
	// Test the shared execution entry as well, so callers cannot bypass the
	// request preflight by choosing the hybrid route.
	for _, hybrid := range []*vectorstore.HybridSearchConfig{nil, {}} {
		_, err := performVectorStoreSearch(ctx, current, vectorStoreSearchParams{storeID: vs.ID, request: SearchRequest{Query: "source", Hybrid: hybrid}}, []float32{1, 0, 0})
		if !errors.Is(err, vectorstore.ErrEmbeddingIncompatible) {
			t.Fatalf("search execution bypassed identity: %v", err)
		}
	}
}
