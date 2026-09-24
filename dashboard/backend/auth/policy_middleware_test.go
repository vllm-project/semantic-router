package auth

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

func TestPolicyMuxRejectsUnmappedProtectedRoute(t *testing.T) {
	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "policy-known@example.com", RoleRead, "active")
	mux := NewPolicyMux()
	mux.HandlePolicyFunc(ProtectedRoute("/api/known", PermConfigRead, SensitivityOperational, ResourceOwnerConfig, http.MethodGet),
		func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusNoContent) })
	mux.HandleFallback("/", http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) }))
	mux.Seal()
	handler := AuthenticateRequest(svc, mux)(mux)
	for _, path := range []string{"/api/unknown", "/api/known/extra"} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, newAuthenticatedRequest(t, svc, user, http.MethodGet, path, ""))
		if response.Code != http.StatusForbidden {
			t.Errorf("%s returned %d", path, response.Code)
		}
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, newAuthenticatedRequest(t, svc, user, http.MethodGet, "/api/known", ""))
	if response.Code != http.StatusNoContent {
		t.Fatalf("known route returned %d", response.Code)
	}
}

func TestPolicyMuxRejectsRegistrationWithoutPolicy(t *testing.T) {
	mux := NewPolicyMux()
	defer func() {
		if recover() == nil {
			t.Fatal("unbound protected route registration did not panic")
		}
	}()
	mux.HandleFunc("/api/unbound", func(http.ResponseWriter, *http.Request) {})
}

func TestIndependentPermissionGrantAndRevoke(t *testing.T) {
	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "policy-grants@example.com", RoleRead, "active")
	for _, permission := range []string{PermConfigRead, PermReplayRead, PermFeedbackSubmit, PermInferenceRun, PermMlPipeline, PermOpenClaw} {
		if _, err := svc.store.db.Exec(`INSERT INTO user_permissions(user_id, permission_key, allowed) VALUES(?,?,0)
ON CONFLICT(user_id, permission_key) DO UPDATE SET allowed=0`, user.ID, permission); err != nil {
			t.Fatal(err)
		}
		perms, err := svc.store.GetEffectivePermissions(t.Context(), RoleRead, user.ID)
		if err != nil || perms[permission] {
			t.Fatalf("%s revoke: perms=%v err=%v", permission, perms, err)
		}
		_, err = svc.store.db.Exec(`UPDATE user_permissions SET allowed=1 WHERE user_id=? AND permission_key=?`, user.ID, permission)
		if err != nil {
			t.Fatal(err)
		}
		perms, err = svc.store.GetEffectivePermissions(t.Context(), RoleRead, user.ID)
		if err != nil || !perms[permission] {
			t.Fatalf("%s grant: perms=%v err=%v", permission, perms, err)
		}
	}
}

func TestMutationRejectsRevocationWhileBodyIsPaused(t *testing.T) {
	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "policy-paused@example.com", RoleRead, "active")
	mux := NewPolicyMux()
	called := make(chan struct{}, 1)
	mux.HandlePolicyFunc(ProtectedMutationRoute("/api/probe", PermInferenceRun, "probe.run", SensitivitySensitive, ResourceOwnerInference, 1024, http.MethodPost),
		func(w http.ResponseWriter, _ *http.Request) {
			called <- struct{}{}
			w.WriteHeader(http.StatusNoContent)
		})
	mux.Seal()
	request := newAuthenticatedRequest(t, svc, user, http.MethodPost, "/api/probe", "")
	body := &pausedBody{entered: make(chan struct{}), release: make(chan struct{}), body: strings.NewReader("{}")}
	request.Body = body
	response := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		AuthenticateRequest(svc, mux)(mux).ServeHTTP(response, request)
		close(done)
	}()
	<-body.entered
	if _, err := svc.store.db.Exec(`INSERT INTO user_permissions(user_id, permission_key, allowed) VALUES(?,?,0)
ON CONFLICT(user_id, permission_key) DO UPDATE SET allowed=0`, user.ID, PermInferenceRun); err != nil {
		t.Fatal(err)
	}
	close(body.release)
	<-done
	if response.Code != http.StatusForbidden {
		t.Fatalf("status=%d, want forbidden", response.Code)
	}
	select {
	case <-called:
		t.Fatal("revoked mutation reached handler")
	default:
	}
}

type pausedBody struct {
	entered chan struct{}
	release chan struct{}
	body    *strings.Reader
	once    sync.Once
}

func (b *pausedBody) Read(p []byte) (int, error) {
	b.once.Do(func() { close(b.entered); <-b.release })
	return b.body.Read(p)
}

func (b *pausedBody) Close() error { return nil }

var _ io.ReadCloser = (*pausedBody)(nil)
