//go:build !windows && cgo

package apiserver

import (
	"net/http"
	"net/http/httptest"
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
		{name: "invalid limit", url: "/v1/vector_stores?limit=abc", code: "INVALID_LIMIT"},
		{name: "zero limit", url: "/v1/vector_stores?limit=0", code: "INVALID_LIMIT"},
		{name: "negative limit", url: "/v1/vector_stores?limit=-1", code: "INVALID_LIMIT"},
		{name: "invalid order", url: "/v1/vector_stores?order=sideways", code: "INVALID_ORDER"},
		{name: "ambiguous cursors", url: "/v1/vector_stores?after=vs_a&before=vs_b", code: "INVALID_CURSOR"},
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
	req := httptest.NewRequest(http.MethodGet, "/v1/vector_stores?limit=1000&order=asc&after=vs_a", nil)
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
