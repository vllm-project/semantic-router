package auth

import (
	"database/sql"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

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

func TestPolicyLookupUsesEscapedPathLikeServeMux(t *testing.T) {
	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "policy-escaped-path@example.com", RoleRead, "active")
	mux := NewPolicyMux()
	mux.HandlePolicyFunc(ProtectedRoute("/api/items/{id}", PermConfigRead,
		SensitivityOperational, ResourceOwnerConfig, http.MethodGet),
		func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusNoContent) })
	mux.HandlePolicyFunc(ProtectedRoute("/api/items/a/{id}", PermUsersManage,
		SensitivitySensitive, ResourceOwnerAuth, http.MethodGet),
		func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusAccepted) })
	mux.Seal()
	handler := AuthenticateRequest(svc, mux)(mux)

	encoded := newAuthenticatedRequest(t, svc, user, http.MethodGet, "/api/items/a%2Fb", "")
	policy, lookup := mux.LookupRoutePolicy(encoded.Method, encoded.URL.EscapedPath())
	if lookup != RouteFound || policy.Permission != PermConfigRead {
		t.Fatalf("encoded slash lookup = %+v, %v; want config.read", policy, lookup)
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, encoded)
	if response.Code != http.StatusNoContent {
		t.Fatalf("encoded slash dispatch returned %d, want 204", response.Code)
	}

	response = httptest.NewRecorder()
	handler.ServeHTTP(response, newAuthenticatedRequest(t, svc, user, http.MethodGet, "/api/items/a/b", ""))
	if response.Code != http.StatusForbidden {
		t.Fatalf("literal slash reached the privileged route: status=%d", response.Code)
	}
}

func TestRegisteredCORSPreflightDoesNotAuthorizeActualRequest(t *testing.T) {
	svc := newTestAuthService(t)
	mux := NewPolicyMux()
	mux.HandlePolicyFunc(ProtectedRoute("/api/config", PermConfigRead,
		SensitivityOperational, ResourceOwnerConfig, http.MethodGet),
		func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodOptions {
				w.WriteHeader(http.StatusNoContent)
				return
			}
			w.WriteHeader(http.StatusOK)
		})
	mux.Seal()
	handler := AuthenticateRequest(svc, mux)(mux)

	preflight := httptest.NewRequest(http.MethodOptions, "/api/config", nil)
	preflight.Header.Set("Origin", "https://example.test")
	preflight.Header.Set("Access-Control-Request-Method", http.MethodGet)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, preflight)
	if response.Code != http.StatusNoContent {
		t.Fatalf("credential-free registered preflight returned %d, want 204", response.Code)
	}
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/api/config", nil))
	if response.Code != http.StatusUnauthorized {
		t.Fatalf("unauthenticated actual request returned %d, want 401", response.Code)
	}
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodOptions, "/api/unknown", nil))
	if response.Code != http.StatusForbidden {
		t.Fatalf("unknown route preflight returned %d, want 403", response.Code)
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
	streaming := ProtectedStreamingMutationRoute("/api/upload", PermMlPipeline, "ml.upload",
		SensitivitySensitive, ResourceOwnerML, NoBodyLimit, http.MethodPost)
	if err := ValidateRouteContract(streaming); err == nil {
		t.Fatal("unbounded streaming mutation was accepted")
	}
}

func TestIndependentPermissionGrantAndRevoke(t *testing.T) {
	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "policy-grants@example.com", RoleRead, "active")
	for _, permission := range []string{PermConfigRead, PermReplayRead, PermFeedbackSubmit, PermInferenceRun, PermMlPipeline, PermMcpManage} {
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
		{http.MethodPost, "/api/mcp/servers", PermMcpManage, ResourceOwnerTools},
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

func TestReadAndManagePermissionsStayIndependent(t *testing.T) {
	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "mcp-read-only@example.com", RoleRead, "active")
	mux := NewPolicyMux()
	allow := func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusNoContent) }
	mux.HandlePolicyFunc(Route("/api/mcp/servers",
		ReadPolicy(http.MethodGet, PermMcpRead, SensitivitySensitive, ResourceOwnerTools),
		MutationPolicy(http.MethodPost, PermMcpManage, "mcp.server.create", SensitivitySecret, ResourceOwnerTools, 1024),
	), allow)
	mux.HandlePolicyFunc(ProtectedMutationRoute("/api/mcp/servers/{id}/connect", PermMcpManage, "mcp.server.connect",
		SensitivitySensitive, ResourceOwnerTools, 1024, http.MethodPost), allow)
	mux.Seal()
	handler := AuthenticateRequest(svc, mux)(mux)
	requestStatus := func(method, path string) int {
		t.Helper()
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, newAuthenticatedRequest(t, svc, user, method, path, `{}`))
		return response.Code
	}
	readPath := "/api/mcp/servers"
	managedPaths := []string{"/api/mcp/servers", "/api/mcp/servers/server-1/connect"}
	if got := requestStatus(http.MethodGet, readPath); got != http.StatusNoContent {
		t.Fatalf("server list with read permission returned %d", got)
	}
	for _, path := range managedPaths {
		if got := requestStatus(http.MethodPost, path); got != http.StatusForbidden {
			t.Errorf("read-only user reached POST %s: %d", path, got)
		}
	}
	if _, err := svc.store.db.Exec(`INSERT INTO user_permissions(user_id, permission_key, allowed) VALUES(?,?,1)
ON CONFLICT(user_id, permission_key) DO UPDATE SET allowed=1`, user.ID, PermMcpManage); err != nil {
		t.Fatal(err)
	}
	for _, path := range managedPaths {
		if got := requestStatus(http.MethodPost, path); got != http.StatusNoContent {
			t.Errorf("granted manager denied POST %s: %d", path, got)
		}
	}
	if _, err := svc.store.db.Exec(`UPDATE user_permissions SET allowed=0 WHERE user_id=? AND permission_key=?`, user.ID, PermMcpManage); err != nil {
		t.Fatal(err)
	}
	for _, path := range managedPaths {
		if got := requestStatus(http.MethodPost, path); got != http.StatusForbidden {
			t.Errorf("revoked manager reached POST %s: %d", path, got)
		}
	}
	if got := requestStatus(http.MethodGet, readPath); got != http.StatusNoContent {
		t.Fatalf("revoking manage also revoked the independent read: %d", got)
	}
}

func TestAuditedWebSocketHandshakeRecordsSwitchingProtocols(t *testing.T) {
	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "audited-websocket@example.com", RoleWrite, "active")
	mux := NewPolicyMux()
	mux.HandlePolicyFunc(Route("/api/probe/{id}/ws", RoutePolicy{
		Method: http.MethodGet, Permission: PermLogsRead,
		AuditMode: AuditRequired, AuditAction: "probe.ws.connect",
		Sensitivity: SensitivitySecret, ResourceOwner: ResourceOwnerObservability,
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
		"ws"+strings.TrimPrefix(server.URL, "http")+"/api/probe/probe-1/ws",
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
	recordedStatus := waitForAuditStatus(t, svc.store.db, "probe.ws.connect")
	if recordedStatus != http.StatusSwitchingProtocols {
		t.Fatalf("audited handshake status = %d, want 101", recordedStatus)
	}
}

// waitForAuditStatus polls for the newest audit row of one action. The record
// is written by the server goroutine after the upgraded handler returns, so a
// single query races the insert and fails with "sql: no rows in result set"
// when the query wins (#4582).
func waitForAuditStatus(t *testing.T, db *sql.DB, action string) int {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		var status int
		err := db.QueryRow(`SELECT status_code FROM user_audit_logs WHERE action = ? ORDER BY id DESC LIMIT 1`,
			action).Scan(&status)
		if err == nil {
			return status
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("no audit row for %s arrived within the deadline", action)
	return 0
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

func TestStreamingMutationBoundsUploadAndRechecksBeforeCommit(t *testing.T) {
	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "streaming-upload@example.com", RoleRead, "active")
	mux := NewPolicyMux()
	const limit = 6 << 20
	enteredHandler := false
	committed := false
	mux.HandlePolicyFunc(ProtectedStreamingMutationRoute("/api/upload", PermInferenceRun,
		"upload.create", SensitivitySensitive, ResourceOwnerInference, limit, http.MethodPost),
		func(w http.ResponseWriter, r *http.Request) {
			enteredHandler = true
			payload, readErr := io.ReadAll(r.Body)
			if readErr != nil {
				http.Error(w, "Invalid upload", http.StatusBadRequest)
				return
			}
			if len(payload) != 5<<20 || RejectRevokedMutation(w, r) {
				return
			}
			committed = true
			w.WriteHeader(http.StatusCreated)
		})
	mux.Seal()
	handler := AuthenticateRequest(svc, mux)(mux)

	payload := strings.Repeat("x", 5<<20)
	request := newAuthenticatedRequest(t, svc, user, http.MethodPost, "/api/upload", "")
	body := &handlerReadBody{Reader: strings.NewReader(payload), handlerEntered: &enteredHandler}
	request.Body = body
	request.ContentLength = int64(len(payload))
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusCreated || !committed || body.readBeforeHandler {
		t.Fatalf("streamed upload: status=%d committed=%t readBeforeHandler=%t",
			response.Code, committed, body.readBeforeHandler)
	}

	enteredHandler, committed = false, false
	oversized := newAuthenticatedRequest(t, svc, user, http.MethodPost, "/api/upload", "")
	oversized.Body = io.NopCloser(strings.NewReader(strings.Repeat("x", limit+1)))
	oversized.ContentLength = limit + 1
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, oversized)
	if response.Code != http.StatusRequestEntityTooLarge || enteredHandler || committed {
		t.Fatalf("oversized upload: status=%d enteredHandler=%t committed=%t",
			response.Code, enteredHandler, committed)
	}
}

func TestStreamingMutationRejectsPermissionRevokedDuringUpload(t *testing.T) {
	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "streaming-revoked@example.com", RoleRead, "active")
	mux := NewPolicyMux()
	committed := make(chan struct{}, 1)
	mux.HandlePolicyFunc(ProtectedStreamingMutationRoute("/api/upload", PermInferenceRun,
		"upload.create", SensitivitySensitive, ResourceOwnerInference, 1024, http.MethodPost),
		func(w http.ResponseWriter, r *http.Request) {
			if _, readErr := io.ReadAll(r.Body); readErr != nil {
				http.Error(w, "Invalid upload", http.StatusBadRequest)
				return
			}
			if RejectRevokedMutation(w, r) {
				return
			}
			committed <- struct{}{}
			w.WriteHeader(http.StatusCreated)
		})
	mux.Seal()
	request := newAuthenticatedRequest(t, svc, user, http.MethodPost, "/api/upload", "")
	body := &pausedBody{entered: make(chan struct{}), release: make(chan struct{}), body: strings.NewReader("upload")}
	request.Body = body
	request.ContentLength = int64(len("upload"))
	response := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		AuthenticateRequest(svc, mux)(mux).ServeHTTP(response, request)
		close(done)
	}()
	<-body.entered
	if _, updateErr := svc.store.db.Exec(`INSERT INTO user_permissions(user_id, permission_key, allowed) VALUES(?,?,0)
ON CONFLICT(user_id, permission_key) DO UPDATE SET allowed=0`, user.ID, PermInferenceRun); updateErr != nil {
		t.Fatal(updateErr)
	}
	close(body.release)
	<-done
	if response.Code != http.StatusForbidden {
		t.Fatalf("revoked upload returned %d, want 403", response.Code)
	}
	select {
	case <-committed:
		t.Fatal("revoked upload committed")
	default:
	}
}

type handlerReadBody struct {
	io.Reader
	handlerEntered    *bool
	readBeforeHandler bool
}

func (b *handlerReadBody) Read(p []byte) (int, error) {
	if !*b.handlerEntered {
		b.readBeforeHandler = true
	}
	return b.Reader.Read(p)
}

func (b *handlerReadBody) Close() error { return nil }

func TestReadStylePostRejectsRevocationWhileBodyIsPaused(t *testing.T) {
	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "policy-paused-preview@example.com", RoleRead, "active")
	if _, err := svc.store.db.Exec(`INSERT INTO user_permissions(user_id, permission_key, allowed) VALUES(?,?,1)
ON CONFLICT(user_id, permission_key) DO UPDATE SET allowed=1`, user.ID, PermEvalRun); err != nil {
		t.Fatal(err)
	}
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
