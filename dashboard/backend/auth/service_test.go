package auth

import (
	"context"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/console"
)

func TestResolveBootstrapSessionCreatesAdminSessionAndCookie(t *testing.T) {
	service, stores := newTestAuthService(t, Config{
		Mode:          ModeBootstrap,
		BootstrapRole: console.ConsoleRoleAdmin,
		SessionTTL:    2 * time.Hour,
	})

	req := httptest.NewRequest(http.MethodGet, "/api/auth/session", nil)
	w := httptest.NewRecorder()

	session, err := service.ResolveSession(w, req)
	if err != nil {
		t.Fatalf("ResolveSession() error = %v", err)
	}

	if !session.Authenticated {
		t.Fatal("expected authenticated bootstrap session")
	}
	if session.EffectiveRole != console.ConsoleRoleAdmin {
		t.Fatalf("expected admin bootstrap role, got %q", session.EffectiveRole)
	}
	if session.Session.ID == "" {
		t.Fatal("expected session id to be persisted")
	}

	cookies := w.Result().Cookies()
	if len(cookies) != 1 || cookies[0].Name != service.SessionCookieName() {
		t.Fatalf("expected auth cookie %q, got %#v", service.SessionCookieName(), cookies)
	}

	gotSession, err := stores.Sessions.GetSession(context.Background(), session.Session.ID)
	if err != nil {
		t.Fatalf("GetSession() error = %v", err)
	}
	if gotSession == nil || gotSession.UserID != session.User.ID {
		t.Fatalf("expected persisted bootstrap session for %q, got %#v", session.User.ID, gotSession)
	}

	bindings, err := stores.RoleBindings.ListRoleBindings(context.Background(), console.RoleBindingFilter{
		PrincipalType: console.PrincipalTypeUser,
		PrincipalID:   session.User.ID,
		ScopeType:     console.ScopeTypeGlobal,
		Limit:         8,
	})
	if err != nil {
		t.Fatalf("ListRoleBindings() error = %v", err)
	}
	if len(bindings) != 1 || bindings[0].Role != console.ConsoleRoleAdmin {
		t.Fatalf("expected persisted bootstrap admin binding, got %#v", bindings)
	}
}

func TestResolveProxySessionUsesHeaderRolesAndCookieFallback(t *testing.T) {
	service, _ := newTestAuthService(t, Config{
		Mode:             ModeProxy,
		SessionTTL:       time.Hour,
		ProxyUserHeader:  "X-Forwarded-User",
		ProxyRolesHeader: "X-Forwarded-Roles",
	})

	req := httptest.NewRequest(http.MethodGet, "/api/auth/session", nil)
	req.Header.Set("X-Forwarded-User", "alice@example.com")
	req.Header.Set("X-Forwarded-Roles", "editor, operator")
	w := httptest.NewRecorder()

	firstSession, err := service.ResolveSession(w, req)
	if err != nil {
		t.Fatalf("ResolveSession() error = %v", err)
	}
	if firstSession.EffectiveRole != console.ConsoleRoleOperator {
		t.Fatalf("expected operator role from proxy headers, got %q", firstSession.EffectiveRole)
	}
	if !firstSession.Capabilities.CanDeployConfig {
		t.Fatal("expected operator role to grant deploy capability")
	}
	if !firstSession.Capabilities.CanRunMLPipeline {
		t.Fatal("expected operator role to grant ML pipeline capability")
	}

	fallbackReq := httptest.NewRequest(http.MethodGet, "/api/auth/session", nil)
	for _, cookie := range w.Result().Cookies() {
		fallbackReq.AddCookie(cookie)
	}
	fallbackWriter := httptest.NewRecorder()

	fallbackSession, err := service.ResolveSession(fallbackWriter, fallbackReq)
	if err != nil {
		t.Fatalf("ResolveSession() fallback error = %v", err)
	}
	if fallbackSession.User.ID != firstSession.User.ID {
		t.Fatalf("expected fallback session for %q, got %q", firstSession.User.ID, fallbackSession.User.ID)
	}
	if fallbackSession.EffectiveRole != console.ConsoleRoleOperator {
		t.Fatalf("expected operator role on cookie fallback, got %q", fallbackSession.EffectiveRole)
	}
}

func TestRequireRoleRejectsInsufficientRole(t *testing.T) {
	service, _ := newTestAuthService(t, Config{
		Mode:             ModeProxy,
		ProxyUserHeader:  "X-Forwarded-User",
		ProxyRolesHeader: "X-Forwarded-Roles",
	})

	req := httptest.NewRequest(http.MethodPost, "/api/router/config/deploy", nil)
	req.Header.Set("X-Forwarded-User", "viewer@example.com")
	req.Header.Set("X-Forwarded-Roles", "viewer")
	w := httptest.NewRecorder()

	service.RequireRole(console.ConsoleRoleOperator, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})).ServeHTTP(w, req)

	if w.Code != http.StatusForbidden {
		t.Fatalf("expected status 403, got %d", w.Code)
	}
}

func newTestAuthService(t *testing.T, cfg Config) (*Service, *console.Stores) {
	t.Helper()

	stores, err := console.OpenStore(console.StoreConfig{
		Backend:    console.StoreBackendSQLite,
		SQLitePath: filepath.Join(t.TempDir(), "console.db"),
	})
	if err != nil {
		t.Fatalf("OpenStore() error = %v", err)
	}
	t.Cleanup(func() {
		_ = stores.Lifecycle.Close()
	})

	service, err := New(cfg, stores)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return service, stores
}
