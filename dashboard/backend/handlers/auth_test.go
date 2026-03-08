package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	dashboardauth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/console"
)

func TestAuthSessionHandler(t *testing.T) {
	service, _ := newAuthHandlerService(t)
	req := httptest.NewRequest(http.MethodGet, "/api/auth/session", nil)
	w := httptest.NewRecorder()

	AuthSessionHandler(service)(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var response AuthSessionResponse
	if err := json.NewDecoder(w.Body).Decode(&response); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if !response.Authenticated {
		t.Fatal("expected authenticated auth session response")
	}
	if response.EffectiveRole != string(console.ConsoleRoleAdmin) {
		t.Fatalf("expected admin effective role, got %q", response.EffectiveRole)
	}
	if len(w.Result().Cookies()) == 0 {
		t.Fatal("expected auth session handler to set a session cookie")
	}
}

func newAuthHandlerService(t *testing.T) (*dashboardauth.Service, *console.Stores) {
	t.Helper()

	stores, err := console.OpenStore(console.StoreConfig{
		Backend:    console.StoreBackendSQLite,
		SQLitePath: t.TempDir() + "/console.db",
	})
	if err != nil {
		t.Fatalf("OpenStore() error = %v", err)
	}
	t.Cleanup(func() {
		_ = stores.Lifecycle.Close()
	})

	service, err := dashboardauth.New(dashboardauth.Config{
		Mode:          dashboardauth.ModeBootstrap,
		BootstrapRole: console.ConsoleRoleAdmin,
	}, stores)
	if err != nil {
		t.Fatalf("auth.New() error = %v", err)
	}
	return service, stores
}
