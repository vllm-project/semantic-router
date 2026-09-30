//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestManagementAuditPagesAcrossMutationKinds(t *testing.T) {
	server := &ClassificationAPIServer{}
	for _, action := range []RouteAuditAction{AuditActionConfigPatch, AuditActionCacheFlush, AuditActionConfigPut, AuditActionCacheInvalidate} {
		server.appendManagementAudit(apiRoute{EndpointMetadata: EndpointMetadata{Method: "POST", Path: "/test"}, AuditAction: action}, "request", managementPrincipal{Role: "operator"}, httptest.NewRequest(http.MethodPost, "/test", nil), http.StatusOK)
	}
	mux := server.setupRoutes()
	read := func(query string) managementAuditResponse {
		t.Helper()
		response := httptest.NewRecorder()
		mux.ServeHTTP(response, httptest.NewRequest(http.MethodGet, apiObservabilityPath+"/audit"+query, nil))
		if response.Code != http.StatusOK {
			t.Fatalf("audit = %d %s", response.Code, response.Body.String())
		}
		var page managementAuditResponse
		if err := json.Unmarshal(response.Body.Bytes(), &page); err != nil {
			t.Fatal(err)
		}
		return page
	}
	first := read("?limit=2")
	if len(first.Entries) != 2 || !first.HasMore || first.NextSequence != 2 || first.Retention != "process" {
		t.Fatalf("first page: %+v", first)
	}
	second := read("?limit=2&after_sequence=2")
	if len(second.Entries) != 2 || second.HasMore || second.Entries[0].PreviousHash != first.Entries[1].Hash {
		t.Fatalf("second page: %+v", second)
	}
	filtered := read("?action=cache.invalidate")
	if len(filtered.Entries) != 1 || filtered.Entries[0].Action != AuditActionCacheInvalidate || filtered.NextSequence != 4 {
		t.Fatalf("filter: %+v", filtered)
	}
	for _, query := range []string{"?limit=0", "?limit=1001", "?after_sequence=-1"} {
		response := httptest.NewRecorder()
		mux.ServeHTTP(response, httptest.NewRequest(http.MethodGet, apiObservabilityPath+"/audit"+query, nil))
		if response.Code != http.StatusBadRequest {
			t.Fatalf("invalid query %s: %d", query, response.Code)
		}
	}
}

func TestManagementAuditBoundedRetentionReportsLostCursor(t *testing.T) {
	server := &ClassificationAPIServer{}
	request := httptest.NewRequest(http.MethodPut, "/test", nil)
	for range maxManagementAuditEntries + 2 {
		server.appendManagementAudit(apiRoute{AuditAction: AuditActionConfigPut}, "request", managementPrincipal{}, request, http.StatusOK)
	}
	page := server.managementAuditPage(1, 2, "")
	if len(server.managementAuditEntries) != maxManagementAuditEntries || !page.Truncated || page.OldestSequence != 3 || len(page.Entries) != 2 || page.Entries[0].Sequence != 3 || page.Entries[1].Sequence != 4 {
		t.Fatalf("retention: %+v", page)
	}
	last := server.managementAuditPage(maxManagementAuditEntries, 2, "")
	if len(last.Entries) != 2 || last.HasMore || last.NextSequence != maxManagementAuditEntries+2 || last.Entries[1].PreviousHash != last.Entries[0].Hash {
		t.Fatalf("wrapped page lost ordering or chain: %+v", last)
	}
}

func TestManagementAuditRequiresItsOwnPermission(t *testing.T) {
	t.Setenv("VSR_MGMT_AUDIT_TEST_TOKEN", "audit-test-token")
	server := testManagementAPIServer(t, config.ManagementAPIConfig{Auth: config.ManagementAPIAuthConfig{
		Mode:   config.ManagementAuthModeBearer,
		Tokens: []config.ManagementAPITokenRef{{Env: "VSR_MGMT_AUDIT_TEST_TOKEN", Role: "metrics-only"}},
		Roles:  map[string][]string{"metrics-only": {"metrics.read"}},
	}})
	request := httptest.NewRequest(http.MethodGet, apiObservabilityPath+"/audit", nil)
	request.Header.Set("Authorization", "Bearer audit-test-token")
	response := httptest.NewRecorder()
	server.setupRoutes().ServeHTTP(response, request)
	if response.Code != http.StatusForbidden {
		t.Fatalf("audit permission: %d", response.Code)
	}
}
