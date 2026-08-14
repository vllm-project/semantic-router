package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
)

// activatedResolver resolves to "not in setup mode", which is what these
// read-only cases previously got from cfg.SetupMode.
func activatedResolver(t *testing.T) *setupmode.Resolver {
	t.Helper()
	return setupmode.New(createActivatedSetupConfig(t, t.TempDir()), false)
}

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
		}, activatedResolver(t)).ServeHTTP(recorder, req)

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
		SettingsHandler(&config.Config{ReadonlyMode: false, SetupMode: false}, activatedResolver(t)).ServeHTTP(recorder, req)

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
		SettingsHandler(&config.Config{ReadonlyMode: true, SetupMode: false}, activatedResolver(t)).ServeHTTP(recorder, req)
		if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
			t.Fatalf("decode response error = %v", err)
		}
		if !response.ReadonlyMode {
			t.Fatalf("readonlyMode = false, want true when global readonly is enabled")
		}
	})
}

// SettingsResponse.SetupMode is kept but now reads the resolver. cfg.SetupMode is
// the legacy flag frozen at startup, so these cases set the two to opposite
// values in both directions and prove the config file wins.
func TestSettingsHandlerReportsResolvedSetupModeNotTheLegacyFlag(t *testing.T) {
	t.Parallel()

	settingsSetupMode := func(t *testing.T, cfg *config.Config, resolver *setupmode.Resolver) bool {
		t.Helper()
		recorder := httptest.NewRecorder()
		SettingsHandler(cfg, resolver).ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/api/settings", nil))
		if recorder.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200; body=%s", recorder.Code, recorder.Body.String())
		}
		var response SettingsResponse
		if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
			t.Fatalf("decode response error = %v", err)
		}
		return response.SetupMode
	}

	t.Run("config declares setup mode while the legacy flag is false", func(t *testing.T) {
		t.Parallel()

		configPath := createBootstrapSetupConfig(t, t.TempDir())
		got := settingsSetupMode(t, &config.Config{SetupMode: false}, setupmode.New(configPath, false))

		if !got {
			t.Fatalf("setupMode = false, want true: /api/settings must report the config file, not cfg.SetupMode")
		}
	})

	t.Run("config is activated while the legacy flag is still true", func(t *testing.T) {
		t.Parallel()

		configPath := createActivatedSetupConfig(t, t.TempDir())
		got := settingsSetupMode(t, &config.Config{SetupMode: true}, setupmode.New(configPath, true))

		if got {
			t.Fatalf("setupMode = true, want false: a stale legacy flag must not show through /api/settings")
		}
	})
}
