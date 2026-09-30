//go:build !windows && cgo

package apiserver

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

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
