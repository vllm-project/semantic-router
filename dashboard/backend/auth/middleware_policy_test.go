package auth

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

func newPolicyTestRoutes(t *testing.T, handler http.Handler) *PolicyMux {
	t.Helper()
	routes := NewPolicyMux()
	routes.HandleGroup([]RouteContract{
		PublicRoute("/api/status", http.MethodGet),
		ProtectedRoute("/api/router/config/all", PermConfigRead, SensitivitySecret, ResourceOwnerConfig, http.MethodGet),
		ProtectedRoute("/api/logs", PermLogsRead, SensitivitySensitive, ResourceOwnerObservability, http.MethodGet),
		ProtectedBoundedRoute("/api/router/v1/chat/completions", PermInferenceRun, SensitivitySecret, ResourceOwnerInference, 64, http.MethodPost),
		ProtectedMutationRoute("/api/router/config/update", PermConfigWrite, "config.update", SensitivitySecret, ResourceOwnerConfig, 64, http.MethodPost),
		ProtectedMutationRoute("/api/router/config/deploy", PermConfigDeploy, "config.deploy", SensitivitySecret, ResourceOwnerConfig, 64, http.MethodPost),
		ProtectedRoute("/api/router/api/v1/observability/replays", PermReplayRead, SensitivitySecret, ResourceOwnerReplay, http.MethodGet),
		ProtectedMutationRoute("/api/router/api/v1/observability/outcomes", PermFeedbackSubmit, "feedback.submit", SensitivitySensitive, ResourceOwnerFeedback, 64, http.MethodPost),
		ProtectedRoute("/api/ml-pipeline/jobs", PermMlPipeline, SensitivitySensitive, ResourceOwnerML, http.MethodGet),
		ProtectedRoute("/api/openclaw/status", PermOpenClawRead, SensitivitySensitive, ResourceOwnerOpenClaw, http.MethodGet),
		SessionRoute("/api/auth/me", SensitivitySensitive, ResourceOwnerAuth, http.MethodGet),
	}, handler)
	routes.HandleFallback("/", handler)
	return routes
}

func TestAuthenticateRequestDeniesUnknownProtectedRoutes(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	admin := newTestUser(t, svc, "unknown-route@example.com", RoleAdmin, "active")
	called := false
	routes := newPolicyTestRoutes(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		called = true
		w.WriteHeader(http.StatusNoContent)
	}))
	handler := AuthenticateRequest(svc, routes)(routes)

	for _, tc := range []struct {
		name         string
		method, path string
		want         int
		wantHandler  bool
	}{
		{name: "unregistered api path", method: http.MethodGet, path: "/api/unregistered", want: http.StatusForbidden},
		{name: "unregistered embedded path", method: http.MethodGet, path: "/embedded/unknown/", want: http.StatusForbidden},
		{name: "deeper than a registered pattern", method: http.MethodGet, path: "/api/router/config/all/extra", want: http.StatusForbidden},
		{name: "undeclared method", method: http.MethodDelete, path: "/api/router/config/all", want: http.StatusMethodNotAllowed},
		{name: "public route", method: http.MethodGet, path: "/api/status", want: http.StatusNoContent, wantHandler: true},
		{name: "static fallback", method: http.MethodGet, path: "/dashboard", want: http.StatusNoContent, wantHandler: true},
		{name: "registered route", method: http.MethodGet, path: "/api/router/config/all", want: http.StatusNoContent, wantHandler: true},
		{name: "session-only route", method: http.MethodGet, path: "/api/auth/me", want: http.StatusNoContent, wantHandler: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			called = false
			recorder := httptest.NewRecorder()
			handler.ServeHTTP(recorder, newAuthenticatedRequest(t, svc, admin, tc.method, tc.path, ""))
			if recorder.Code != tc.want || called != tc.wantHandler {
				t.Fatalf("status = %d called = %v, want %d/%v", recorder.Code, called, tc.want, tc.wantHandler)
			}
		})
	}

	// Unknown protected paths are denied before any credential is examined.
	called = false
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/api/unregistered", nil))
	if recorder.Code != http.StatusForbidden || called {
		t.Fatalf("anonymous unknown route: status = %d called = %v", recorder.Code, called)
	}
}

func TestIndependentPermissionGrantAndRevoke(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	routes := newPolicyTestRoutes(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	handler := AuthenticateRequest(svc, routes)(routes)

	domains := []struct {
		permission   string
		method, path string
	}{
		{PermConfigRead, http.MethodGet, "/api/router/config/all"},
		{PermConfigDeploy, http.MethodPost, "/api/router/config/deploy"},
		{PermReplayRead, http.MethodGet, "/api/router/api/v1/observability/replays"},
		{PermFeedbackSubmit, http.MethodPost, "/api/router/api/v1/observability/outcomes"},
		{PermInferenceRun, http.MethodPost, "/api/router/v1/chat/completions"},
		{PermMlPipeline, http.MethodGet, "/api/ml-pipeline/jobs"},
		{PermOpenClawRead, http.MethodGet, "/api/openclaw/status"},
	}
	// A reader with every default grant removed proves each permission is granted
	// and revoked on its own: granting one domain unlocks only its route.
	user := newTestUser(t, svc, "independent@example.com", RoleRead, "active")
	if _, err := svc.store.db.Exec(`DELETE FROM role_permissions WHERE role = ?`, RoleRead); err != nil {
		t.Fatal(err)
	}
	serve := func(method, path string) int {
		recorder := httptest.NewRecorder()
		handler.ServeHTTP(recorder, newAuthenticatedRequest(t, svc, user, method, path, `{}`))
		return recorder.Code
	}
	for _, granted := range domains {
		if err := svc.store.SetUserPermission(context.Background(), user.ID, granted.permission, true); err != nil {
			t.Fatal(err)
		}
		for _, probe := range domains {
			want := http.StatusForbidden
			if probe.permission == granted.permission {
				want = http.StatusNoContent
			}
			if got := serve(probe.method, probe.path); got != want {
				t.Fatalf("granted %s: %s %s = %d, want %d", granted.permission, probe.method, probe.path, got, want)
			}
		}
		if err := svc.store.SetUserPermission(context.Background(), user.ID, granted.permission, false); err != nil {
			t.Fatal(err)
		}
		if got := serve(granted.method, granted.path); got != http.StatusForbidden {
			t.Fatalf("revoked %s still admitted: %d", granted.permission, got)
		}
	}
}

func TestUserLevelRevocationOverridesRoleGrant(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "revoked-reader@example.com", RoleRead, "active")
	if err := svc.store.SetUserPermission(context.Background(), user.ID, PermConfigRead, false); err != nil {
		t.Fatal(err)
	}
	perms, err := svc.store.GetEffectivePermissions(context.Background(), user.Role, user.ID)
	if err != nil {
		t.Fatal(err)
	}
	if perms[PermConfigRead] || !perms[PermEvalRead] {
		t.Fatalf("effective permissions = %v, want config.read revoked and evaluation.read kept", perms)
	}
}

func TestInferencePermissionIsIndependentFromConfigRead(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "chat@example.com", RoleRead, "active")
	if _, err := svc.store.db.Exec(`DELETE FROM role_permissions WHERE role = ? AND permission_key = ?`, RoleRead, PermInferenceRun); err != nil {
		t.Fatal(err)
	}
	called := false
	routes := newPolicyTestRoutes(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		called = true
		w.WriteHeader(http.StatusNoContent)
	}))
	handler := AuthenticateRequest(svc, routes)(routes)

	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, newAuthenticatedRequest(t, svc, user, http.MethodPost, "/api/router/v1/chat/completions", `{}`))
	if recorder.Code != http.StatusForbidden || called {
		t.Fatalf("config.read authorized inference: status = %d called = %v", recorder.Code, called)
	}
	recorder = httptest.NewRecorder()
	handler.ServeHTTP(recorder, newAuthenticatedRequest(t, svc, user, http.MethodGet, "/api/router/config/all", ""))
	if recorder.Code != http.StatusNoContent {
		t.Fatalf("config read denied alongside inference: %d", recorder.Code)
	}
}

func TestMutationRequestBodyIsBounded(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	admin := newTestUser(t, svc, "bounded@example.com", RoleAdmin, "active")
	called := false
	routes := newPolicyTestRoutes(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		if _, err := io.ReadAll(r.Body); err != nil {
			http.Error(w, err.Error(), http.StatusRequestEntityTooLarge)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	handler := AuthenticateRequest(svc, routes)(routes)

	for _, tc := range []struct {
		name string
		path string
		want int
	}{
		{name: "revalidated mutation rejected before handler", path: "/api/router/config/update", want: http.StatusRequestEntityTooLarge},
		{name: "bounded inference rejected while streaming", path: "/api/router/v1/chat/completions", want: http.StatusRequestEntityTooLarge},
	} {
		t.Run(tc.name, func(t *testing.T) {
			called = false
			recorder := httptest.NewRecorder()
			handler.ServeHTTP(recorder, newAuthenticatedRequest(t, svc, admin, http.MethodPost, tc.path, strings.Repeat("x", 65)))
			if recorder.Code != tc.want {
				t.Fatalf("status = %d, want %d", recorder.Code, tc.want)
			}
			if tc.path == "/api/router/config/update" && called {
				t.Fatal("oversized revalidated mutation reached the handler")
			}
		})
	}

	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, newAuthenticatedRequest(t, svc, admin, http.MethodPost, "/api/router/config/update", strings.Repeat("x", 64)))
	if recorder.Code != http.StatusNoContent || !called {
		t.Fatalf("body at the limit: status = %d called = %v", recorder.Code, called)
	}
}

// A request is admitted with its permissions intact, its body stalls, the actor
// is demoted while the body is paused, and the body then completes. The
// middleware must re-resolve the actor after the body arrives so the handler,
// and therefore the side effect, never runs.
func TestMutationRechecksPermissionAfterPausedBody(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	writer := newTestUser(t, svc, "paused-writer@example.com", RoleWrite, "active")
	var mu sync.Mutex
	sideEffects := 0
	routes := newPolicyTestRoutes(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		sideEffects++
		mu.Unlock()
		w.WriteHeader(http.StatusNoContent)
	}))
	handler := AuthenticateRequest(svc, routes)(routes)

	bodyReader, bodyWriter := io.Pipe()
	request := newAuthenticatedRequest(t, svc, writer, http.MethodPost, "/api/router/config/update", "")
	request.Body = bodyReader
	request.ContentLength = -1
	recorder := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		defer close(done)
		handler.ServeHTTP(recorder, request)
	}()

	// The first chunk proves the request was admitted and the body is being read.
	if _, err := bodyWriter.Write([]byte(`{"partial":`)); err != nil {
		t.Fatalf("write first chunk: %v", err)
	}
	if _, err := svc.store.UpdateUserRoleOrStatus(context.Background(), writer.ID, RoleRead, ""); err != nil {
		t.Fatalf("demote actor: %v", err)
	}
	if _, err := bodyWriter.Write([]byte(`true}`)); err != nil {
		t.Fatalf("write final chunk: %v", err)
	}
	_ = bodyWriter.Close()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("request did not complete")
	}

	mu.Lock()
	defer mu.Unlock()
	if recorder.Code != http.StatusForbidden || sideEffects != 0 {
		t.Fatalf("status = %d side effects = %d, want %d and none", recorder.Code, sideEffects, http.StatusForbidden)
	}
}

// The same barrier at the handler boundary: a session revoked after the body
// was read but before the handler commits is rejected by the revalidator.
func TestRevalidateRequestRejectsSessionRevokedBeforeCommit(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	admin := newTestUser(t, svc, "revoked-before-commit@example.com", RoleAdmin, "active")
	token, err := svc.issueToken(admin)
	if err != nil {
		t.Fatal(err)
	}
	committed := false
	routes := newPolicyTestRoutes(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Simulate a slow validation phase during which the session is revoked.
		if revokeErr := svc.RevokeToken(r.Context(), token); revokeErr != nil {
			t.Errorf("revoke token: %v", revokeErr)
		}
		if RejectRevokedMutation(w, r) {
			return
		}
		committed = true
		w.WriteHeader(http.StatusNoContent)
	}))
	handler := AuthenticateRequest(svc, routes)(routes)

	request := httptest.NewRequest(http.MethodPost, "/api/router/config/update", strings.NewReader(`{}`))
	request.Header.Set("Authorization", "Bearer "+token)
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusForbidden || committed {
		t.Fatalf("status = %d committed = %v, want %d and no commit", recorder.Code, committed, http.StatusForbidden)
	}

	// Read routes do not carry a revalidator, and direct handler invocations
	// are never rejected by it.
	plain := httptest.NewRequest(http.MethodGet, "/api/router/config/all", nil)
	if RejectRevokedMutation(httptest.NewRecorder(), plain) {
		t.Fatal("request without a revalidator was rejected")
	}
	if err := RevalidateRequest(plain); err == nil {
		t.Fatal("RevalidateRequest without a revalidator returned nil")
	}
	injected := plain.WithContext(WithPermissionRevalidator(plain.Context(), func(context.Context) error { return errors.New("revoked") }))
	if !RejectRevokedMutation(httptest.NewRecorder(), injected) {
		t.Fatal("injected failing revalidator did not reject")
	}
}

func TestRequiredAuditRoutesWriteContractAuditRows(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	admin := newTestUser(t, svc, "audited@example.com", RoleAdmin, "active")
	routes := newPolicyTestRoutes(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusAccepted)
	}))
	handler := AuthenticateRequest(svc, routes)(routes)

	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, newAuthenticatedRequest(t, svc, admin, http.MethodPost, "/api/router/config/deploy", `{}`))
	if recorder.Code != http.StatusAccepted {
		t.Fatalf("status = %d", recorder.Code)
	}
	logs, total, err := svc.store.QueryAuditLogs(context.Background(), AuditLogListOptions{Limit: 10})
	if err != nil {
		t.Fatal(err)
	}
	if total != 1 || len(logs) != 1 || logs[0].Action != "config.deploy" || logs[0].UserID != admin.ID ||
		logs[0].StatusCode != http.StatusAccepted || logs[0].Resource != string(ResourceOwnerConfig) {
		t.Fatalf("audit rows = %+v (total %d)", logs, total)
	}
}
