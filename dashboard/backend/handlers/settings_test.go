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

	t.Run("marks read users as readonly even when dashboard is globally writable", func(t *testing.T) {
		t.Parallel()

		req := httptest.NewRequest(http.MethodGet, "/api/settings", nil)
		req = req.WithContext(auth.WithAuthContext(req.Context(), auth.AuthContext{
			UserID: "user-read-1",
			Role:   auth.RoleRead,
			Perms: map[string]bool{
				auth.PermConfigRead: true,
			},
		}))

		recorder := httptest.NewRecorder()
		SettingsHandler(&config.Config{
			ReadonlyMode: false,
			SetupMode:    false,
			RouterAPIURL: "http://router:8080",
			FleetSimURL:  "http://fleet-sim:8000",
		}).ServeHTTP(recorder, req)

		if recorder.Code != http.StatusOK {
			t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
		}

		var response SettingsResponse
		if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
			t.Fatalf("decode response error = %v", err)
		}
		if !response.ReadonlyMode {
			t.Fatalf("readonlyMode = false, want true")
		}
		if !response.FleetSimEnabled {
			t.Fatalf("fleetSimEnabled = false, want true")
		}
		if response.RouterEvalURL != "http://router:8080/api/v1/eval" {
			t.Fatalf("routerEvalEndpoint = %q, want %q", response.RouterEvalURL, "http://router:8080/api/v1/eval")
		}
	})

	t.Run("keeps write users writable until global readonly is enabled", func(t *testing.T) {
		t.Parallel()

		req := httptest.NewRequest(http.MethodGet, "/api/settings", nil)
		req = req.WithContext(auth.WithAuthContext(req.Context(), auth.AuthContext{
			UserID: "user-write-1",
			Role:   auth.RoleWrite,
			Perms: map[string]bool{
				auth.PermConfigRead:  true,
				auth.PermConfigWrite: true,
			},
		}))

		recorder := httptest.NewRecorder()
		SettingsHandler(&config.Config{ReadonlyMode: false, SetupMode: false}).ServeHTTP(recorder, req)

		var response SettingsResponse
		if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
			t.Fatalf("decode response error = %v", err)
		}
		if response.ReadonlyMode {
			t.Fatalf("readonlyMode = true, want false")
		}
		if response.FleetSimEnabled {
			t.Fatalf("fleetSimEnabled = true, want false when simulator URL is empty")
		}
		if response.RouterEvalURL != fallbackRouterEvalEndpoint {
			t.Fatalf("routerEvalEndpoint = %q, want %q", response.RouterEvalURL, fallbackRouterEvalEndpoint)
		}

		recorder = httptest.NewRecorder()
		SettingsHandler(&config.Config{ReadonlyMode: true, SetupMode: false}).ServeHTTP(recorder, req)
		if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
			t.Fatalf("decode response error = %v", err)
		}
		if !response.ReadonlyMode {
			t.Fatalf("readonlyMode = false, want true when global readonly is enabled")
		}
	})
}
