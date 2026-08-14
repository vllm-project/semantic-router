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
	t.Run("keeps package import available when only runtime config is read-only", testSettingsRuntimeReadonly)
	t.Run("keeps config editing available when only the package store is read-only", testSettingsStoreReadonly)
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
		RuntimeConfigWritable: true,
		RecipeStoreWritable:   true,
		RouterAPIURL:          "http://router:8080",
		FleetSimURL:           "http://fleet-sim:8000",
	}, auth.AuthContext{
		UserID: "user-read-1",
		Role:   auth.RoleRead,
		Perms:  map[string]bool{auth.PermConfigRead: true},
	})
	if !response.ReadonlyMode {
		t.Fatalf("readonlyMode = false, want true")
	}
	if response.ServerReadonly || !response.RuntimeConfigWritable || !response.RecipeStoreWritable {
		t.Fatalf("unexpected server capability response: %#v", response)
	}
	if !response.FleetSimEnabled {
		t.Fatalf("fleetSimEnabled = false, want true")
	}
	if response.RouterEvalURL != "http://router:8080/api/v1/eval" {
		t.Fatalf("routerEvalEndpoint = %q, want %q", response.RouterEvalURL, "http://router:8080/api/v1/eval")
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
	response := requestSettings(t, &config.Config{
		RuntimeConfigWritable: true,
		RecipeStoreWritable:   true,
	}, authContext)
	if response.ReadonlyMode || response.ServerReadonly || !response.RuntimeConfigWritable || !response.RecipeStoreWritable {
		t.Fatalf("unexpected writable server capability response: %#v", response)
	}
	if response.FleetSimEnabled || response.RouterEvalURL != fallbackRouterEvalEndpoint {
		t.Fatalf("unexpected optional service response: %#v", response)
	}

	response = requestSettings(t, &config.Config{
		ReadonlyMode:          true,
		RuntimeConfigWritable: true,
		RecipeStoreWritable:   true,
	}, authContext)
	if !response.ReadonlyMode || !response.ServerReadonly || !response.RuntimeConfigWritable || !response.RecipeStoreWritable {
		t.Fatalf("global readonly did not remain independent from physical capabilities: %#v", response)
	}
}

func adminSettingsAuthContext(userID string) auth.AuthContext {
	return auth.AuthContext{
		UserID: userID,
		Role:   auth.RoleAdmin,
		Perms: map[string]bool{
			auth.PermConfigWrite:  true,
			auth.PermConfigDeploy: true,
		},
	}
}

func testSettingsRuntimeReadonly(t *testing.T) {
	t.Parallel()
	response := requestSettings(t, &config.Config{
		RuntimeConfigWritable: false,
		RecipeStoreWritable:   true,
	}, adminSettingsAuthContext("admin-1"))
	if !response.ReadonlyMode || response.ServerReadonly || response.RuntimeConfigWritable || !response.RecipeStoreWritable {
		t.Fatalf("runtime-only readonly capability response = %#v", response)
	}
}

func testSettingsStoreReadonly(t *testing.T) {
	t.Parallel()
	response := requestSettings(t, &config.Config{
		RuntimeConfigWritable: true,
		RecipeStoreWritable:   false,
	}, adminSettingsAuthContext("admin-2"))
	if response.ReadonlyMode || response.ServerReadonly || !response.RuntimeConfigWritable || response.RecipeStoreWritable {
		t.Fatalf("store-only readonly capability response = %#v", response)
	}
}
