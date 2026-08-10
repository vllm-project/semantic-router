//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
)

type responseCacheAPITestStore struct{}

func (responseCacheAPITestStore) LookupExact(context.Context, cache.ExactLookup) (cache.CacheResult, error) {
	return cache.CacheResult{}, nil
}

func (responseCacheAPITestStore) StoreExact(context.Context, cache.CacheWrite) error {
	return nil
}

func (responseCacheAPITestStore) LookupSemantic(context.Context, cache.SemanticLookup) (cache.CacheResult, error) {
	return cache.CacheResult{}, nil
}

func (responseCacheAPITestStore) StoreSemantic(context.Context, cache.CacheWrite) error {
	return nil
}
func (responseCacheAPITestStore) Health(context.Context) error { return nil }
func (responseCacheAPITestStore) Close() error                 { return nil }
func (responseCacheAPITestStore) Stats(context.Context) (cache.CacheStats, error) {
	return cache.CacheStats{TotalEntries: 3}, nil
}

func (responseCacheAPITestStore) Capabilities() cache.BackendCapabilities {
	return cache.CapabilitiesForBackend(cache.InMemoryCacheType)
}

func newResponseCacheAPITestServer() *ClassificationAPIServer {
	return &ClassificationAPIServer{
		responseCache: cache.NewResponseCacheService(
			responseCacheAPITestStore{},
			cache.DefaultResponseCacheServiceOptions(),
		),
	}
}

func TestResponseCacheStatsAreRedacted(t *testing.T) {
	server := newResponseCacheAPITestServer()
	request := httptest.NewRequest(http.MethodGet, "/api/v1/response-cache/stats", nil)
	recorder := httptest.NewRecorder()
	server.handleResponseCacheStats(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
	for _, forbidden := range []string{"prompt", "response_body", "namespace"} {
		if strings.Contains(recorder.Body.String(), forbidden) {
			t.Fatalf("stats response exposed %q: %s", forbidden, recorder.Body.String())
		}
	}
}

func TestResponseCacheInvalidateDefaultsToDryRun(t *testing.T) {
	server := newResponseCacheAPITestServer()
	request := httptest.NewRequest(
		http.MethodPost,
		"/api/v1/response-cache/invalidate",
		bytes.NewBufferString(`{"selector":{"epoch":"release-a"}}`),
	)
	recorder := httptest.NewRecorder()
	server.handleResponseCacheInvalidate(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
	if !strings.Contains(recorder.Body.String(), `"dry_run":true`) {
		t.Fatalf("response = %s, want dry_run", recorder.Body.String())
	}
	if got := server.responseCache.CurrentEpoch(); got != "0" {
		t.Fatalf("epoch = %s, want 0", got)
	}
}

func TestResponseCacheFlushRequiresExplicitConfirmation(t *testing.T) {
	server := newResponseCacheAPITestServer()
	request := httptest.NewRequest(
		http.MethodPost,
		"/api/v1/response-cache/flush",
		bytes.NewBufferString(`{"confirm":"yes"}`),
	)
	recorder := httptest.NewRecorder()
	server.handleResponseCacheFlush(recorder, request)
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", recorder.Code)
	}
}

func TestResponseCacheRoutesCarryDedicatedPermissions(t *testing.T) {
	permissions := map[string]RoutePermission{}
	for _, route := range apiResponseCacheRoutes() {
		permissions[route.Path] = route.Permission
	}
	if permissions["/api/v1/response-cache/stats"] != PermCacheRead {
		t.Fatalf("stats permission = %q", permissions["/api/v1/response-cache/stats"])
	}
	if permissions["/api/v1/response-cache/invalidate"] != PermCacheInvalidate {
		t.Fatalf("invalidate permission = %q", permissions["/api/v1/response-cache/invalidate"])
	}
	if permissions["/api/v1/response-cache/flush"] != PermCacheManage {
		t.Fatalf("flush permission = %q", permissions["/api/v1/response-cache/flush"])
	}
}

func TestResponseCacheMutationAddsHashChainedAuditEntry(t *testing.T) {
	server := newResponseCacheAPITestServer()
	mux := server.setupRoutes()
	request := httptest.NewRequest(
		http.MethodPost,
		"/api/v1/response-cache/invalidate",
		bytes.NewBufferString(`{"selector":{"epoch":"release-a"},"dry_run":false}`),
	)
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	mux.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
	entries := server.responseCacheAuditEntries()
	if len(entries) != 1 || entries[0].Hash == "" {
		t.Fatalf("audit entries = %#v", entries)
	}
	if strings.Contains(entries[0].Hash, "release-a") {
		t.Fatal("audit hash exposed selector value")
	}
}
