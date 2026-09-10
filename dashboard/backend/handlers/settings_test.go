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
	t.Run("marks read users as readonly even when dashboard is globally writable", testSettingsReadUser)
	t.Run("keeps write users writable until global readonly is enabled", testSettingsWriteUser)
	t.Run("keeps package import available when only runtime config is read-only", testSettingsRuntimeReadonly)
	t.Run("keeps config editing available when only the package store is read-only", testSettingsStoreReadonly)
}

func TestSettingsHandlerReportsFrozenEvaluationAvailability(t *testing.T) {
	t.Parallel()
	authContext := adminSettingsAuthContext("evaluation-admin")

	available := requestSettings(t, &config.Config{
		EvaluationAvailable: true,
	}, authContext)
	if !available.EvaluationAvailable || available.EvaluationUnavailableReason != "" {
		t.Fatalf("available Evaluation response = %#v", available)
	}

	unavailable := requestSettings(t, &config.Config{
		EvaluationUnavailableReason: "Evaluation could not be initialized.",
	}, authContext)
	if unavailable.EvaluationAvailable || unavailable.EvaluationUnavailableReason != "Evaluation could not be initialized." {
		t.Fatalf("unavailable Evaluation response = %#v", unavailable)
	}
}

func requestSettings(t *testing.T, cfg *config.Config, authContext auth.AuthContext) SettingsResponse {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, "/api/settings", nil)
	req = req.WithContext(auth.WithAuthContext(req.Context(), authContext))
	recorder := httptest.NewRecorder()
	SettingsHandler(cfg, activatedResolver(t)).ServeHTTP(recorder, req)
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
	if response.RouterEvalURL != fallbackRouterEvalEndpoint {
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
