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

func TestProtectedRouteRequiresAuditActionAtRegistration(t *testing.T) {
	incomplete := Route("/api/incomplete", RoutePolicy{
		Method: http.MethodGet, Permission: PermConfigRead,
		AuditMode: AuditNone, Sensitivity: SensitivitySensitive, ResourceOwner: ResourceOwnerConfig,
	})
	if err := ValidateRouteContract(incomplete); err == nil {
		t.Fatal("protected route without an audit action was accepted")
	}
	func() {
		defer func() {
			if recover() == nil {
				t.Error("protected route without an audit action did not fail registration")
			}
		}()
		NewPolicyMux().HandlePolicyFunc(incomplete, func(http.ResponseWriter, *http.Request) {})
	}()
	secretWithoutAudit := Route("/api/secret", RoutePolicy{
		Method: http.MethodGet, Permission: PermConfigRead, AuditMode: AuditNone,
		AuditAction: "config.read", Sensitivity: SensitivitySecret, ResourceOwner: ResourceOwnerConfig,
	})
	if err := ValidateRouteContract(secretWithoutAudit); err == nil {
		t.Fatal("secret read without auditing was accepted")
	}
	for _, contract := range []RouteContract{
		ProtectedRoute("/api/config", PermConfigRead, SensitivitySensitive, ResourceOwnerConfig, http.MethodGet),
		ProtectedBoundedRoute("/api/config/preview", PermConfigDeploy, SensitivitySensitive, ResourceOwnerConfig, 1024, http.MethodPost),
	} {
		if err := ValidateRouteContract(contract); err != nil {
			t.Fatalf("complete route %q was rejected: %v", contract.Pattern, err)
		}
		if contract.Policies[0].AuditAction == "" {
			t.Errorf("route %q has no audit action", contract.Pattern)
		}
	}
	bounded := ProtectedBoundedRoute("/api/config/preview", PermConfigDeploy, SensitivitySensitive, ResourceOwnerConfig, 1024, http.MethodPost)
	if policy := bounded.Policies[0]; policy.AuditMode != AuditNone || !policy.Revalidate {
		t.Fatalf("read-style POST lacks live revalidation or unexpectedly emits audit writes: %+v", policy)
	}
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

func TestIndependentPermissionsAtProtectedRoutes(t *testing.T) {
	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "policy-route-grants@example.com", RoleRead, "active")
	type routeCase struct {
		method, path, permission string
		owner                    ResourceOwner
	}
	routes := []routeCase{
		{http.MethodPost, "/api/router/config/update", PermConfigWrite, ResourceOwnerConfig},
		{http.MethodGet, "/api/router/api/v1/observability/replays/record-1", PermReplayRead, ResourceOwnerReplay},
		{http.MethodPost, "/api/router/api/v1/observability/outcomes", PermFeedbackSubmit, ResourceOwnerFeedback},
		{http.MethodPost, "/api/router/v1/chat/completions", PermInferenceRun, ResourceOwnerInference},
		{http.MethodPost, "/api/ml-pipeline/train", PermMlPipeline, ResourceOwnerML},
		{http.MethodPost, "/api/openclaw/rooms", PermOpenClaw, ResourceOwnerOpenClaw},
	}
	mux := NewPolicyMux()
	allow := func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusNoContent) }
	for _, route := range routes {
		if route.method == http.MethodGet {
			mux.HandlePolicyFunc(ProtectedRoute(route.path, route.permission, SensitivitySecret, route.owner, route.method), allow)
			continue
		}
		mux.HandlePolicyFunc(ProtectedMutationRoute(route.path, route.permission, route.permission+".test", SensitivitySensitive, route.owner, 1024, route.method), allow)
	}
	mux.Seal()
	handler := AuthenticateRequest(svc, mux)(mux)
	for _, route := range routes {
		if _, err := svc.store.db.Exec(`INSERT INTO user_permissions(user_id, permission_key, allowed) VALUES(?,?,0)
ON CONFLICT(user_id, permission_key) DO UPDATE SET allowed=0`, user.ID, route.permission); err != nil {
			t.Fatal(err)
		}
	}
	requestStatus := func(route routeCase) int {
		t.Helper()
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, newAuthenticatedRequest(t, svc, user, route.method, route.path, `{}`))
		return response.Code
	}
	for _, granted := range routes {
		if _, err := svc.store.db.Exec(`UPDATE user_permissions SET allowed=1 WHERE user_id=? AND permission_key=?`, user.ID, granted.permission); err != nil {
			t.Fatal(err)
		}
		for _, route := range routes {
			want := http.StatusForbidden
			if route.permission == granted.permission {
				want = http.StatusNoContent
			}
			if got := requestStatus(route); got != want {
				t.Errorf("grant %s: %s %s returned %d, want %d", granted.permission, route.method, route.path, got, want)
			}
		}
		if _, err := svc.store.db.Exec(`UPDATE user_permissions SET allowed=0 WHERE user_id=? AND permission_key=?`, user.ID, granted.permission); err != nil {
			t.Fatal(err)
		}
		if got := requestStatus(granted); got != http.StatusForbidden {
			t.Errorf("revoked %s still reached %s: %d", granted.permission, granted.path, got)
		}
	}
}

func TestSecretReadProducesRouteAudit(t *testing.T) {
	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "policy-read-audit@example.com", RoleRead, "active")
	mux := NewPolicyMux()
	contract := ProtectedRoute("/api/replays/record-1", PermReplayRead, SensitivitySecret, ResourceOwnerReplay, http.MethodGet)
	mux.HandlePolicyFunc(contract, func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusNoContent) })
	mux.Seal()
	response := httptest.NewRecorder()
	AuthenticateRequest(svc, mux)(mux).ServeHTTP(response, newAuthenticatedRequest(t, svc, user, http.MethodGet, contract.Pattern, ""))
	if response.Code != http.StatusNoContent {
		t.Fatalf("secret read returned %d", response.Code)
	}
	var action string
	if err := svc.store.db.QueryRow(`SELECT action FROM user_audit_logs WHERE user_id=? ORDER BY id DESC LIMIT 1`, user.ID).Scan(&action); err != nil {
		t.Fatal(err)
	}
	if action != contract.Policies[0].AuditAction {
		t.Fatalf("audit action=%q, want %q", action, contract.Policies[0].AuditAction)
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

func TestReadStylePostRejectsRevocationWhileBodyIsPaused(t *testing.T) {
	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "policy-paused-preview@example.com", RoleRead, "active")
	mux := NewPolicyMux()
	policy := ReadPolicy(http.MethodPost, PermEvalRun, SensitivitySensitive, ResourceOwnerEvaluation)
	policy.MaxBodyBytes = 1024
	called := make(chan struct{}, 1)
	mux.HandlePolicyFunc(Route("/api/preview", policy), func(w http.ResponseWriter, _ *http.Request) {
		called <- struct{}{}
		w.WriteHeader(http.StatusNoContent)
	})
	mux.Seal()
	request := newAuthenticatedRequest(t, svc, user, http.MethodPost, "/api/preview", "")
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
ON CONFLICT(user_id, permission_key) DO UPDATE SET allowed=0`, user.ID, PermEvalRun); err != nil {
		t.Fatal(err)
	}
	close(body.release)
	<-done
	if response.Code != http.StatusForbidden {
		t.Fatalf("status=%d, want forbidden", response.Code)
	}
	select {
	case <-called:
		t.Fatal("revoked preview reached handler")
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
