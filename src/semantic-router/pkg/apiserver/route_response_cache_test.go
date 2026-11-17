//go:build !windows

package apiserver

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type responseCacheAPITestStore struct{}

func (responseCacheAPITestStore) LookupExact(context.Context, cache.ExactLookup) (cache.CacheResult, error) {
	return cache.CacheResult{}, nil
}

func (responseCacheAPITestStore) StoreExact(context.Context, cache.CacheWrite) error {
	return nil
}

func (responseCacheAPITestStore) LookupSemantic(context.Context, cache.SemanticLookup) (cache.SemanticRecall, error) {
	return cache.SemanticRecall{}, nil
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
	request := httptest.NewRequest(http.MethodGet, "/api/v1/storage/response-cache/stats", nil)
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
		"/api/v1/storage/response-cache/invalidate",
		bytes.NewBufferString(`{"selector":{"partition":{"namespace":"tenant-a"}}}`),
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
		"/api/v1/storage/response-cache/flush",
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
	if permissions["/api/v1/storage/response-cache/stats"] != PermCacheRead {
		t.Fatalf("stats permission = %q", permissions["/api/v1/storage/response-cache/stats"])
	}
	if permissions["/api/v1/storage/response-cache/invalidate"] != PermCacheInvalidate {
		t.Fatalf("invalidate permission = %q", permissions["/api/v1/storage/response-cache/invalidate"])
	}
	if permissions["/api/v1/storage/response-cache/flush"] != PermCacheManage {
		t.Fatalf("flush permission = %q", permissions["/api/v1/storage/response-cache/flush"])
	}
}

func TestResponseCacheMutationAddsHashChainedAuditEntry(t *testing.T) {
	server := newResponseCacheAPITestServer()
	mux := server.setupRoutes()
	request := httptest.NewRequest(
		http.MethodPost,
		"/api/v1/storage/response-cache/invalidate",
		bytes.NewBufferString(`{"selector":{"partition":{"namespace":"release-a"}},"dry_run":false}`),
	)
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	mux.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
	entries := server.managementAuditPage(0, 100, "").Entries
	if len(entries) != 1 || entries[0].Hash == "" {
		t.Fatalf("audit entries = %#v", entries)
	}
	if strings.Contains(entries[0].Hash, "release-a") {
		t.Fatal("audit hash exposed selector value")
	}
}

// An invalidate reaches one partition. One without a partition is refused,
// so a role that may invalidate cannot flush every partition; the dry run
// and the real call say the backend cannot count the entries they reach.
func TestResponseCacheInvalidateReachesOnlyItsPartition(t *testing.T) {
	t.Setenv("RESPONSE_CACHE_OPERATOR_TOKEN", "operator-key")
	cfg := &config.RouterConfig{}
	cfg.ManagementAPI = config.ManagementAPIConfig{Port: 8080, Auth: config.ManagementAPIAuthConfig{
		Mode:   config.ManagementAuthModeBearer,
		Tokens: []config.ManagementAPITokenRef{{Env: "RESPONSE_CACHE_OPERATOR_TOKEN", Role: "operator"}},
	}}
	service := cache.NewResponseCacheService(responseCacheAPITestStore{}, cache.DefaultResponseCacheServiceOptions())
	mux := (&ClassificationAPIServer{config: cfg, responseCache: service}).setupRoutes()
	ctx := context.Background()
	tenantA := cache.CacheIdentity{Partition: cache.CachePartition{Namespace: "tenant-a", Decision: "d1"}, ExactFingerprint: "fa"}
	tenantB := cache.CacheIdentity{Partition: cache.CachePartition{Namespace: "tenant-b", Decision: "d2"}, ExactFingerprint: "fb"}
	for _, identity := range []cache.CacheIdentity{tenantA, tenantB} {
		if err := service.StoreExact(ctx, cache.CacheWrite{Identity: identity, ResponseBody: []byte("cached"), TTL: cache.DefaultTTL()}); err != nil {
			t.Fatal(err)
		}
	}
	hits := func() [2]bool {
		a, _ := service.LookupExact(ctx, cache.ExactLookup{Identity: tenantA})
		b, _ := service.LookupExact(ctx, cache.ExactLookup{Identity: tenantB})
		return [2]bool{a.Found, b.Found}
	}
	invalidate := func(body string) (int, map[string]any) {
		request := httptest.NewRequest(http.MethodPost, "/api/v1/storage/response-cache/invalidate", bytes.NewBufferString(body))
		request.Header.Set("Content-Type", "application/json")
		request.Header.Set("Authorization", "Bearer operator-key")
		recorder := httptest.NewRecorder()
		mux.ServeHTTP(recorder, request)
		var decoded map[string]any
		_ = json.Unmarshal(recorder.Body.Bytes(), &decoded)
		return recorder.Code, decoded
	}

	code, body := invalidate(`{"selector":{"epoch":"release-a"},"dry_run":false}`)
	if code != http.StatusBadRequest || !strings.Contains(fmt.Sprint(body), "partition") {
		t.Fatalf("invalidate without a partition: status %d, body %v", code, body)
	}
	if got := hits(); got != [2]bool{true, true} || service.CurrentEpoch() != "0" {
		t.Fatalf("hits %v, epoch %s after a refused invalidate", got, service.CurrentEpoch())
	}
	partitionA := `{"selector":{"partition":{"namespace":"tenant-a","decision":"d1"}}`
	code, body = invalidate(partitionA + `,"dry_run":true}`)
	if code != http.StatusOK || body["count_unknown"] != true || hits() != [2]bool{true, true} {
		t.Fatalf("dry run: status %d, body %v, hits %v", code, body, hits())
	}
	epochB := service.ResolveIdentity(tenantB).Partition.Epoch
	code, body = invalidate(partitionA + `,"dry_run":false}`)
	if code != http.StatusOK || body["count_unknown"] != true {
		t.Fatalf("invalidate: status %d, body %v", code, body)
	}
	// The invalidate also empties the in-process front tier, so tenant-b's
	// entry, which this test store keeps nowhere else, is judged by its epoch.
	if got := hits(); got[0] || service.ResolveIdentity(tenantB).Partition.Epoch != epochB || service.CurrentEpoch() != "0" {
		t.Fatalf("after invalidating tenant-a: hits %v, tenant-b epoch %q (was %q), global epoch %s",
			got, service.ResolveIdentity(tenantB).Partition.Epoch, epochB, service.CurrentEpoch())
	}
}
