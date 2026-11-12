//go:build !windows

package apiserver

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

func TestParseVectorStoreListParamsValidation(t *testing.T) {
	manager := vectorstore.NewManager(
		vectorstore.NewMemoryBackend(vectorstore.MemoryBackendConfig{}),
		vectorstore.NewMemoryMetadataRegistry(),
		2,
		vectorstore.BackendTypeMemory,
	)
	SetVectorStoreManager(manager)
	t.Cleanup(func() { SetVectorStoreManager(nil) })

	server := &ClassificationAPIServer{}
	cases := []struct {
		name string
		url  string
		code string
	}{
		{name: "invalid limit", url: "/api/v1/storage/vector-stores?limit=abc", code: "INVALID_LIMIT"},
		{name: "zero limit", url: "/api/v1/storage/vector-stores?limit=0", code: "INVALID_LIMIT"},
		{name: "negative limit", url: "/api/v1/storage/vector-stores?limit=-1", code: "INVALID_LIMIT"},
		{name: "invalid order", url: "/api/v1/storage/vector-stores?order=sideways", code: "INVALID_ORDER"},
		{name: "ambiguous cursors", url: "/api/v1/storage/vector-stores?after=vs_a&before=vs_b", code: "INVALID_CURSOR"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, tc.url, nil)
			rr := httptest.NewRecorder()

			server.handleListVectorStores(rr, req)

			if rr.Code != http.StatusBadRequest {
				t.Fatalf("expected 400 Bad Request, got %d: %s", rr.Code, rr.Body.String())
			}
			if code := parseErrorResponse(t, rr.Body.Bytes()); code != tc.code {
				t.Fatalf("expected %s, got %s", tc.code, code)
			}
		})
	}
}

func TestParseVectorStoreListParamsCapsLimit(t *testing.T) {
	server := &ClassificationAPIServer{}
	req := httptest.NewRequest(http.MethodGet, "/api/v1/storage/vector-stores?limit=1000&order=asc&after=vs_a", nil)
	rr := httptest.NewRecorder()

	params, ok := server.parseVectorStoreListParams(rr, req)
	if !ok {
		t.Fatalf("expected params to parse, got status %d: %s", rr.Code, rr.Body.String())
	}
	if params.Limit != maxVectorStoreListLimit {
		t.Fatalf("expected capped limit %d, got %d", maxVectorStoreListLimit, params.Limit)
	}
	if params.Order != "asc" {
		t.Fatalf("expected asc order, got %q", params.Order)
	}
	if params.After != "vs_a" {
		t.Fatalf("expected after cursor vs_a, got %q", params.After)
	}
}

func TestHandleListVectorStoresStablePagination(t *testing.T) {
	ctx := context.Background()
	registry := vectorstore.NewMemoryMetadataRegistry()
	for _, id := range []string{"vs_a", "vs_b", "vs_c"} {
		if err := registry.SaveStore(ctx, &vectorstore.VectorStore{
			ID:        id,
			Object:    "vector_store",
			CreatedAt: 1,
			Status:    "active",
		}); err != nil {
			t.Fatalf("save store %s: %v", id, err)
		}
	}

	manager := vectorstore.NewManager(
		vectorstore.NewMemoryBackend(vectorstore.MemoryBackendConfig{}),
		registry,
		2,
		vectorstore.BackendTypeMemory,
	)
	if err := manager.LoadFromRegistry(ctx); err != nil {
		t.Fatalf("load stores: %v", err)
	}
	SetVectorStoreManager(manager)
	t.Cleanup(func() { SetVectorStoreManager(nil) })

	server := &ClassificationAPIServer{}
	firstRequest := httptest.NewRequest(http.MethodGet, "/api/v1/storage/vector-stores?limit=2", nil)
	firstResponse := httptest.NewRecorder()
	server.handleListVectorStores(firstResponse, firstRequest)
	if firstResponse.Code != http.StatusOK {
		t.Fatalf("first page returned %d: %s", firstResponse.Code, firstResponse.Body.String())
	}

	var first objectListResponse[*vectorstore.VectorStore]
	if err := json.Unmarshal(firstResponse.Body.Bytes(), &first); err != nil {
		t.Fatalf("decode first page: %v", err)
	}
	if got := vectorStoreResponseIDs(first.Data); !reflect.DeepEqual(got, []string{"vs_c", "vs_b"}) {
		t.Fatalf("first page IDs = %v, want [vs_c vs_b]", got)
	}

	cursor := first.Data[len(first.Data)-1].ID
	secondRequest := httptest.NewRequest(
		http.MethodGet,
		"/api/v1/storage/vector-stores?limit=2&after="+cursor,
		nil,
	)
	secondResponse := httptest.NewRecorder()
	server.handleListVectorStores(secondResponse, secondRequest)
	if secondResponse.Code != http.StatusOK {
		t.Fatalf("second page returned %d: %s", secondResponse.Code, secondResponse.Body.String())
	}

	var second objectListResponse[*vectorstore.VectorStore]
	if err := json.Unmarshal(secondResponse.Body.Bytes(), &second); err != nil {
		t.Fatalf("decode second page: %v", err)
	}
	if got := vectorStoreResponseIDs(second.Data); !reflect.DeepEqual(got, []string{"vs_a"}) {
		t.Fatalf("second page IDs = %v, want [vs_a]", got)
	}
}

func vectorStoreResponseIDs(stores []*vectorstore.VectorStore) []string {
	ids := make([]string, 0, len(stores))
	for _, store := range stores {
		ids = append(ids, store.ID)
	}
	return ids
}

// unavailableStoreRegistry stands for a metadata database that is down: reads
// work from the manager's in-memory index, every write fails.
type unavailableStoreRegistry struct {
	vectorstore.StoreRegistry
	down bool
}

func (r *unavailableStoreRegistry) SaveStore(ctx context.Context, vs *vectorstore.VectorStore) error {
	if r.down {
		return errors.New("database unavailable")
	}
	return r.StoreRegistry.SaveStore(ctx, vs)
}

func (r *unavailableStoreRegistry) DeleteStore(ctx context.Context, id string) error {
	if r.down {
		return errors.New("database unavailable")
	}
	return r.StoreRegistry.DeleteStore(ctx, id)
}

func vectorStoreRouteServer(manager *vectorstore.Manager) *ClassificationAPIServer {
	registry := routerruntime.NewRegistry(nil)
	registry.SetVectorStoreRuntime(&routerruntime.VectorStoreRuntime{Manager: manager})
	return &ClassificationAPIServer{runtimeRegistry: registry}
}

func serveVectorStoreRoute(handle http.HandlerFunc, method, path, body string) *httptest.ResponseRecorder {
	response := httptest.NewRecorder()
	handle(response, httptest.NewRequest(method, path, strings.NewReader(body)))
	return response
}

// TestVectorStoreMutationsReportTheFailureTheyHit checks that a store the
// router cannot persist answers 500, a reserved metadata key answers 400, and
// only a store that does not exist answers 404.
func TestVectorStoreMutationsReportTheFailureTheyHit(t *testing.T) {
	registry := &unavailableStoreRegistry{StoreRegistry: vectorstore.NewMemoryMetadataRegistry()}
	manager := vectorstore.NewManager(vectorstore.NewMemoryBackend(vectorstore.MemoryBackendConfig{}), registry, 3, vectorstore.BackendTypeMemory)
	vs, err := manager.CreateStore(context.Background(), vectorstore.CreateStoreRequest{Name: "kb"})
	if err != nil {
		t.Fatal(err)
	}
	server := vectorStoreRouteServer(manager)
	storePath := apiStorageVectorStoresPath + "/" + vs.ID
	reserved := `{"metadata":{"` + vectorstore.EmbeddingIdentityMetadataKey + `":"client"}}`

	for _, tc := range []struct {
		name, method, path, body string
		handle                   http.HandlerFunc
		down                     bool
		status                   int
		code                     string
	}{
		{"create with a reserved metadata key", http.MethodPost, apiStorageVectorStoresPath, reserved, server.handleCreateVectorStore, false, http.StatusBadRequest, "INVALID_INPUT"},
		{"update with a reserved metadata key", http.MethodPost, storePath, reserved, server.handleUpdateVectorStore, false, http.StatusBadRequest, "INVALID_INPUT"},
		{"update of an unknown store", http.MethodPost, apiStorageVectorStoresPath + "/vs_missing", `{"name":"x"}`, server.handleUpdateVectorStore, false, http.StatusNotFound, "NOT_FOUND"},
		{"delete of an unknown store", http.MethodDelete, apiStorageVectorStoresPath + "/vs_missing", "", server.handleDeleteVectorStore, false, http.StatusNotFound, "NOT_FOUND"},
		{"update while the registry is down", http.MethodPost, storePath, `{"name":"renamed"}`, server.handleUpdateVectorStore, true, http.StatusInternalServerError, "UPDATE_FAILED"},
		{"delete while the registry is down", http.MethodDelete, storePath, "", server.handleDeleteVectorStore, true, http.StatusInternalServerError, "DELETE_FAILED"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			registry.down = tc.down
			response := serveVectorStoreRoute(tc.handle, tc.method, tc.path, tc.body)
			if response.Code != tc.status {
				t.Fatalf("status = %d, want %d: %s", response.Code, tc.status, response.Body.String())
			}
			if code := parseErrorResponse(t, response.Body.Bytes()); code != tc.code {
				t.Fatalf("code = %s, want %s", code, tc.code)
			}
		})
	}
}
