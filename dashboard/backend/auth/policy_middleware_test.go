package auth

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/gorilla/websocket"
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

func TestOpenClawReadAndManagePermissionsStayIndependent(t *testing.T) {
	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "openclaw-read-only@example.com", RoleRead, "active")
	mux := NewPolicyMux()
	allow := func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusNoContent) }
	mux.HandlePolicyFunc(Route("/api/openclaw/rooms/{id}/messages",
		ReadPolicy(http.MethodGet, PermOpenClawRead, SensitivitySensitive, ResourceOwnerOpenClaw),
		MutationPolicy(http.MethodPost, PermOpenClaw, "openclaw.room.message", SensitivitySecret, ResourceOwnerOpenClaw, 1024),
	), allow)
	for _, contract := range []RouteContract{
		ProtectedRoute("/api/openclaw/rooms/{id}/ws", PermOpenClaw, SensitivitySecret, ResourceOwnerOpenClaw, http.MethodGet),
		ProtectedRoute("/api/openclaw/token", PermOpenClaw, SensitivitySecret, ResourceOwnerOpenClaw, http.MethodGet),
		ProtectedRoute("/embedded/openclaw/", PermOpenClaw, SensitivitySecret, ResourceOwnerOpenClaw, http.MethodGet),
	} {
		mux.HandlePolicyFunc(contract, allow)
	}
	mux.Seal()
	handler := AuthenticateRequest(svc, mux)(mux)
	requestStatus := func(method, path string) int {
		t.Helper()
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, newAuthenticatedRequest(t, svc, user, method, path, `{}`))
		return response.Code
	}
	readPath := "/api/openclaw/rooms/room-1/messages"
	managedPaths := []string{
		"/api/openclaw/rooms/room-1/messages",
		"/api/openclaw/rooms/room-1/ws",
		"/api/openclaw/token",
		"/embedded/openclaw/worker-1/",
	}
	managedMethods := []string{http.MethodPost, http.MethodGet, http.MethodGet, http.MethodGet}
	if got := requestStatus(http.MethodGet, readPath); got != http.StatusNoContent {
		t.Fatalf("room read with read permission returned %d", got)
	}
	for index, path := range managedPaths {
		if got := requestStatus(managedMethods[index], path); got != http.StatusForbidden {
			t.Errorf("read-only user reached %s %s: %d", managedMethods[index], path, got)
		}
	}
	if _, err := svc.store.db.Exec(`INSERT INTO user_permissions(user_id, permission_key, allowed) VALUES(?,?,1)
ON CONFLICT(user_id, permission_key) DO UPDATE SET allowed=1`, user.ID, PermOpenClaw); err != nil {
		t.Fatal(err)
	}
	for index, path := range managedPaths {
		if got := requestStatus(managedMethods[index], path); got != http.StatusNoContent {
			t.Errorf("granted manager denied %s %s: %d", managedMethods[index], path, got)
		}
	}
	if _, err := svc.store.db.Exec(`UPDATE user_permissions SET allowed=0 WHERE user_id=? AND permission_key=?`, user.ID, PermOpenClaw); err != nil {
		t.Fatal(err)
	}
	for index, path := range managedPaths {
		if got := requestStatus(managedMethods[index], path); got != http.StatusForbidden {
			t.Errorf("revoked manager reached %s %s: %d", managedMethods[index], path, got)
		}
	}
	if got := requestStatus(http.MethodGet, readPath); got != http.StatusNoContent {
		t.Fatalf("revoking manage also revoked independent room read: %d", got)
	}
}

func TestAuditedWebSocketHandshakeRecordsSwitchingProtocols(t *testing.T) {
	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "openclaw-websocket@example.com", RoleWrite, "active")
	mux := NewPolicyMux()
	mux.HandlePolicyFunc(Route("/api/openclaw/rooms/{id}/ws", RoutePolicy{
		Method: http.MethodGet, Permission: PermOpenClaw,
		AuditMode: AuditRequired, AuditAction: "openclaw.room.ws.connect",
		Sensitivity: SensitivitySecret, ResourceOwner: ResourceOwnerOpenClaw,
	}), func(w http.ResponseWriter, r *http.Request) {
		conn, err := (&websocket.Upgrader{}).Upgrade(w, r, nil)
		if err == nil {
			_ = conn.Close()
		}
	})
	mux.Seal()
	server := httptest.NewServer(AuthenticateRequest(svc, mux)(mux))
	t.Cleanup(server.Close)
	token, err := svc.issueToken(user)
	if err != nil {
		t.Fatal(err)
	}
	conn, response, err := websocket.DefaultDialer.Dial(
		"ws"+strings.TrimPrefix(server.URL, "http")+"/api/openclaw/rooms/room-1/ws",
		http.Header{"Authorization": []string{"Bearer " + token}},
	)
	if response != nil && response.Body != nil {
		defer response.Body.Close()
	}
	if err != nil {
		t.Fatalf("audited websocket handshake: %v", err)
	}
	_ = conn.Close()
	if response.StatusCode != http.StatusSwitchingProtocols {
		t.Fatalf("handshake status = %d", response.StatusCode)
	}
	var recordedStatus int
	if err := svc.store.db.QueryRow(`SELECT status_code FROM user_audit_logs WHERE action = ? ORDER BY id DESC LIMIT 1`,
		"openclaw.room.ws.connect").Scan(&recordedStatus); err != nil {
		t.Fatal(err)
	}
	if recordedStatus != http.StatusSwitchingProtocols {
		t.Fatalf("audited handshake status = %d, want 101", recordedStatus)
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
