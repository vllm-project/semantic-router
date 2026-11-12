//go:build !windows

package apiserver

import (
	"context"
	"net/http"
	"net/http/httptest"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

// TestOpenAPIListsTheErrorStatusesHandlersReturn sends requests that end in
// a handler's own error answer and checks the exact status and code, and that
// the operation's OpenAPI responses list that status.
func TestOpenAPIListsTheErrorStatusesHandlersReturn(t *testing.T) {
	ctx := context.Background()
	backend := vectorstore.NewMemoryBackend(vectorstore.MemoryBackendConfig{})
	stores := vectorstore.NewMemoryMetadataRegistry()
	old := vectorstore.NewManager(backend, stores, 3, vectorstore.BackendTypeMemory, vectorstore.WithEmbeddingIdentity("old-model"))
	vs, err := old.CreateStore(ctx, vectorstore.CreateStoreRequest{Name: "old"})
	if err != nil {
		t.Fatal(err)
	}
	current := vectorstore.NewManager(backend, stores, 3, vectorstore.BackendTypeMemory, vectorstore.WithEmbeddingIdentity("new-model"))
	if err = current.LoadFromRegistry(ctx); err != nil {
		t.Fatal(err)
	}
	files, err := vectorstore.NewFileStore(t.TempDir(), vectorstore.NewMemoryMetadataRegistry())
	if err != nil {
		t.Fatal(err)
	}
	file, err := files.Save("notes.txt", []byte("hello"), "assistants")
	if err != nil {
		t.Fatal(err)
	}
	registry := routerruntime.NewRegistry(nil)
	registry.SetVectorStoreRuntime(&routerruntime.VectorStoreRuntime{
		Manager: current, FileStore: files, Embedder: forbiddenIdentityEmbedder{t},
		Pipeline: vectorstore.NewIngestionPipeline(backend, files, current, forbiddenIdentityEmbedder{t}, vectorstore.PipelineConfig{}),
	})
	server := &ClassificationAPIServer{runtimeRegistry: registry}
	spec := server.generateOpenAPISpec()
	attach := `{"file_id":"` + file.ID + `"}`
	for _, tc := range []struct {
		name, operation, path, body string
		handle                      http.HandlerFunc
		status                      int
		code                        string
	}{
		{
			"config that does not validate", apiConfigValidatePath, apiConfigValidatePath,
			`{"yaml":"version: v0.5\nproviders:\n  defaults:\n    model: m1\n  models:\n    - name: m1\n      descriptin: typo\nrecipes:\n  - name: default\n"}`,
			server.handleConfigValidate, http.StatusUnprocessableEntity, "CONFIG_VALIDATION_ERROR",
		},
		{
			"search a store of another embedding model", apiStorageVectorStoresPath + "/{id}/search", apiStorageVectorStoresPath + "/" + vs.ID + "/search",
			`{"query":"hello"}`, server.handleSearchVectorStore, http.StatusConflict, "EMBEDDING_REINDEX_REQUIRED",
		},
		{
			"attach to a store of another embedding model", apiStorageVectorStoresPath + "/{id}/files", apiStorageVectorStoresPath + "/" + vs.ID + "/files",
			attach, server.handleAttachFile, http.StatusConflict, "EMBEDDING_REINDEX_REQUIRED",
		},
		{
			"attach to a missing store", apiStorageVectorStoresPath + "/{id}/files", apiStorageVectorStoresPath + "/vs_missing/files",
			attach, server.handleAttachFile, http.StatusNotFound, "NOT_FOUND",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			response := httptest.NewRecorder()
			tc.handle(response, httptest.NewRequest(http.MethodPost, tc.path, strings.NewReader(tc.body)))
			if code := parseErrorResponse(t, response.Body.Bytes()); response.Code != tc.status || code != tc.code {
				t.Fatalf("answered %d %s, want %d %s: %s", response.Code, code, tc.status, tc.code, response.Body.String())
			}
			responses := spec.Paths[tc.operation].Post.Responses
			if _, listed := responses[strconv.Itoa(tc.status)]; !listed {
				statuses := make([]string, 0, len(responses))
				for status := range responses {
					statuses = append(statuses, status)
				}
				slices.Sort(statuses)
				t.Fatalf("POST %s returns %d, but its OpenAPI responses are %v", tc.operation, tc.status, statuses)
			}
		})
	}
}
