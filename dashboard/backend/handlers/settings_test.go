package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
)

func TestSettingsHandlerReflectsEffectiveReadonlyMode(t *testing.T) {
	t.Parallel()
	t.Run("marks read users as readonly even when dashboard is globally writable", testSettingsReadUser)
	t.Run("keeps write users writable until global readonly is enabled", testSettingsWriteUser)
}

func requestSettings(t *testing.T, cfg *config.Config, authContext auth.AuthContext) SettingsResponse {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, "/api/settings", nil)
	req = req.WithContext(auth.WithAuthContext(req.Context(), authContext))
	recorder := httptest.NewRecorder()
	SettingsHandler(cfg).ServeHTTP(recorder, req)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
	}
	if cacheControl := recorder.Header().Get("Cache-Control"); cacheControl != "no-store" {
		t.Fatalf("Cache-Control = %q, want no-store", cacheControl)
	}
	var response SettingsResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response error = %v", err)
	}
	return response
}

func testSettingsReadUser(t *testing.T) {
	t.Parallel()
	response := requestSettings(t, &config.Config{
		RouterAPIURL:    "http://router:8080",
		RouterPublicURL: "https://router.example.test",
		FleetSimURL:     "http://fleet-sim:8000",
	}, auth.AuthContext{
		UserID: "user-read-1",
		Role:   auth.RoleRead,
		Perms:  map[string]bool{auth.PermConfigRead: true},
	})
	if !response.ReadonlyMode {
		t.Fatalf("readonlyMode = false, want true")
	}
	if response.ServerReadonly {
		t.Fatalf("unexpected server capability response: %#v", response)
	}
	if !response.FleetSimEnabled {
		t.Fatalf("fleetSimEnabled = false, want true")
	}
	if response.RouterEvalURL != "http://router:8080/api/v1/eval" {
		t.Fatalf("routerEvalEndpoint = %q, want %q", response.RouterEvalURL, "http://router:8080/api/v1/eval")
	}
	if response.RouterPublicURL != "https://router.example.test" {
		t.Fatalf("routerPublicUrl = %q", response.RouterPublicURL)
	}
}

func testSettingsWriteUser(t *testing.T) {
	t.Parallel()
	authContext := auth.AuthContext{
		UserID: "user-write-1",
		Role:   auth.RoleWrite,
		Perms: map[string]bool{
			auth.PermConfigRead:  true,
			auth.PermConfigWrite: true,
		},
	}
	response := requestSettings(t, &config.Config{}, authContext)
	if response.ReadonlyMode || response.ServerReadonly {
		t.Fatalf("unexpected writable server capability response: %#v", response)
	}
	if response.FleetSimEnabled || response.RouterEvalURL != fallbackRouterEvalEndpoint {
		t.Fatalf("unexpected optional service response: %#v", response)
	}

	response = requestSettings(t, &config.Config{
		ReadonlyMode: true,
	}, authContext)
	if !response.ReadonlyMode || !response.ServerReadonly {
		t.Fatalf("global readonly was not enforced: %#v", response)
	}
}
