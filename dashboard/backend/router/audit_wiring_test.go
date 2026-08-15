package router

import (
	"context"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	auth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

// TestAuditMutationRecordsConfigDeploy verifies that the config deploy route is
// wrapped so that an authenticated deploy writes an audit row carrying the
// actor, action, resource and method (issue #2825: privileged config changes
// were never recorded in the audit log).
func TestAuditMutationRecordsConfigDeploy(t *testing.T) {
	t.Parallel()

	store, storeErr := auth.NewStore(filepath.Join(t.TempDir(), "auth.db"))
	if storeErr != nil {
		t.Fatalf("NewStore() error = %v", storeErr)
	}
	t.Cleanup(func() { _ = store.Close() })

	svc := auth.NewService(store, "test-secret", 1)
	ctx := context.Background()
	if err := svc.EnsureBootstrapAdmin(ctx, "admin@example.com", "secret-password", "Admin"); err != nil {
		t.Fatalf("EnsureBootstrapAdmin() error = %v", err)
	}
	token, _, err := svc.Login(ctx, "admin@example.com", "secret-password")
	if err != nil {
		t.Fatalf("Login() error = %v", err)
	}

	mux := http.NewServeMux()
	deployCalled := false
	mux.HandleFunc("/api/router/config/deploy", auditMutation(
		svc,
		fixedAuditAction("config.deploy"),
		"router/config",
		func(w http.ResponseWriter, _ *http.Request) {
			deployCalled = true
			w.WriteHeader(http.StatusOK)
		},
	))
	handler := wrapWithAuth(mux, svc)

	req := httptest.NewRequest(http.MethodPost, "/api/router/config/deploy", strings.NewReader("{}"))
	req.Header.Set("Authorization", "Bearer "+token)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if !deployCalled {
		t.Fatal("deploy handler was not called")
	}
	if rec.Code != http.StatusOK {
		t.Fatalf("deploy status = %d, want %d", rec.Code, http.StatusOK)
	}

	logs, err := store.ListAuditLogs(ctx, "", "config.deploy", "", 10, 0)
	if err != nil {
		t.Fatalf("ListAuditLogs() error = %v", err)
	}
	if len(logs) != 1 {
		t.Fatalf("audit rows for config.deploy = %d, want 1", len(logs))
	}
	row := logs[0]
	if row.Action != "config.deploy" {
		t.Errorf("row action = %q, want config.deploy", row.Action)
	}
	if row.Resource != "router/config" {
		t.Errorf("row resource = %q, want router/config", row.Resource)
	}
	if row.Method != http.MethodPost {
		t.Errorf("row method = %q, want POST", row.Method)
	}
	if row.UserID == "" {
		t.Error("row user id is empty, expected the acting user")
	}
	if row.StatusCode != http.StatusOK {
		t.Errorf("row status code = %d, want 200", row.StatusCode)
	}
}

// TestAuditMutationSkipsReads verifies that read-only methods on a wrapped route
// are not recorded in the audit log, keeping the log a record of writes only.
func TestAuditMutationSkipsReads(t *testing.T) {
	t.Parallel()

	store, storeErr := auth.NewStore(filepath.Join(t.TempDir(), "auth.db"))
	if storeErr != nil {
		t.Fatalf("NewStore() error = %v", storeErr)
	}
	t.Cleanup(func() { _ = store.Close() })

	svc := auth.NewService(store, "test-secret", 1)
	ctx := context.Background()
	if err := svc.EnsureBootstrapAdmin(ctx, "admin@example.com", "secret-password", "Admin"); err != nil {
		t.Fatalf("EnsureBootstrapAdmin() error = %v", err)
	}
	token, _, err := svc.Login(ctx, "admin@example.com", "secret-password")
	if err != nil {
		t.Fatalf("Login() error = %v", err)
	}

	mux := http.NewServeMux()
	readCalled := false
	mux.HandleFunc("/api/router/config/yaml", auditMutation(
		svc,
		fixedAuditAction("config.update"),
		"router/config",
		func(w http.ResponseWriter, _ *http.Request) {
			readCalled = true
			w.WriteHeader(http.StatusOK)
		},
	))
	handler := wrapWithAuth(mux, svc)

	req := httptest.NewRequest(http.MethodGet, "/api/router/config/yaml", nil)
	req.Header.Set("Authorization", "Bearer "+token)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if !readCalled {
		t.Fatal("read handler was not called")
	}
	logs, err := store.ListAuditLogs(ctx, "", "", "", 10, 0)
	if err != nil {
		t.Fatalf("ListAuditLogs() error = %v", err)
	}
	if len(logs) != 0 {
		t.Fatalf("audit rows after read-only request = %d, want 0", len(logs))
	}
}

// TestAuditMutationUnavailableStorePassesThrough verifies that a nil auth
// service leaves the handler unwrapped instead of panicking.
func TestAuditMutationUnavailableStorePassesThrough(t *testing.T) {
	t.Parallel()

	nextCalled := false
	handler := auditMutation(
		nil,
		fixedAuditAction("config.deploy"),
		"router/config",
		func(w http.ResponseWriter, _ *http.Request) {
			nextCalled = true
			w.WriteHeader(http.StatusOK)
		},
	)

	req := httptest.NewRequest(http.MethodPost, "/api/router/config/deploy", nil)
	rec := httptest.NewRecorder()
	handler(rec, req)

	if !nextCalled {
		t.Fatal("handler was not called when auth service is nil")
	}
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusOK)
	}
}
