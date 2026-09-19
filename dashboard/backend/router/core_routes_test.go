package router

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/recipe"
	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
)

func TestRegisterRecipeRoutesPassesStoreToRecipeService(t *testing.T) {
	directory := filepath.Join("..", "..", "..", "config", "recipes", "accuracy")
	configPath := filepath.Join(directory, "config.yaml")
	t.Setenv("VLLM_SR_ACTIVE_RECIPE_DIR", directory)
	t.Setenv(recipe.ManagementCredentialEnv, "")

	store := recipe.NewStore(recipe.StoreOptions{
		Root:       filepath.Join(t.TempDir(), "recipe-store"),
		ConfigPath: configPath,
	})
	token, err := store.EnsureManagementCredential()
	if err != nil {
		t.Fatalf("EnsureManagementCredential(): %v", err)
	}
	configBytes, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("ReadFile(config): %v", err)
	}
	hash := sha256.Sum256(configBytes)
	var authenticated []string
	router := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer "+token {
			http.Error(w, "missing service credential", http.StatusUnauthorized)
			return
		}
		authenticated = append(authenticated, r.URL.Path)
		switch r.URL.Path {
		case "/api/v1/config/hash":
			_, _ = fmt.Fprintf(w, `{"source_config_hash":%q,"generated_runtime_hash":%q,"active_runtime_hash":%q,"activation_status":"active"}`, hex.EncodeToString(hash[:]), hex.EncodeToString(hash[:]), hex.EncodeToString(hash[:]))
		case "/api/v1/routing/preview":
			_, _ = w.Write([]byte(`{
  "requested_model":"vllm-sr/auto",
  "selected_model":"gpt55-worker",
  "selection_status":"selected",
  "selection_method":"static",
  "recipe":"default",
  "routing_decision":"accuracy_direct",
  "decision_result":{
    "decision_name":"accuracy_direct",
    "algorithm":"static",
    "plugins":[],
    "matched_signals":{}
  },
  "recommended_models":["gpt55-worker"],
  "eval_trace":[{"decision_name":"accuracy_direct","matched":true}]
}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer router.Close()

	mux := http.NewServeMux()
	registerRecipeRoutes(mux, &config.Config{
		AbsConfigPath: configPath,
		ConfigDir:     filepath.Dir(directory),
		RouterAPIURL:  router.URL,
	}, store)

	descriptorResponse := httptest.NewRecorder()
	mux.ServeHTTP(descriptorResponse, httptest.NewRequest(http.MethodGet, "/api/recipe", nil))
	if descriptorResponse.Code != http.StatusOK {
		t.Fatalf("GET /api/recipe status=%d body=%s", descriptorResponse.Code, descriptorResponse.Body.String())
	}
	var descriptor struct {
		Digests struct {
			Recipe string `json:"recipe"`
		} `json:"digests"`
	}
	if err := json.NewDecoder(descriptorResponse.Body).Decode(&descriptor); err != nil {
		t.Fatalf("decode descriptor: %v", err)
	}

	request := httptest.NewRequest(http.MethodPost, "/api/recipe/probes/accuracy_direct/direct_explanation/validate", nil)
	request.Header.Set("If-Match", `"`+descriptor.Digests.Recipe+`"`)
	validationResponse := httptest.NewRecorder()
	mux.ServeHTTP(validationResponse, request)
	if validationResponse.Code != http.StatusOK {
		t.Fatalf("POST /api/recipe/probes/.../validate status=%d body=%s", validationResponse.Code, validationResponse.Body.String())
	}
	if got, want := authenticated, []string{"/api/v1/config/hash", "/api/v1/routing/preview", "/api/v1/config/hash"}; !slices.Equal(got, want) {
		t.Fatalf("authenticated Router requests = %v, want %v", got, want)
	}
}

func TestRegisterRecipeRoutesExposesUnmanagedDescriptor(t *testing.T) {
	t.Setenv("VLLM_SR_ACTIVE_RECIPE_DIR", "")
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.yaml"), []byte("version: v0.3\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(config): %v", err)
	}
	mux := http.NewServeMux()
	registerRecipeRoutes(mux, &config.Config{ConfigDir: dir, RouterAPIURL: "http://router.invalid"})
	recorder := httptest.NewRecorder()
	mux.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/api/recipe", nil))
	if recorder.Code != http.StatusOK {
		t.Fatalf("GET /api/recipe status = %d, body=%s", recorder.Code, recorder.Body.String())
	}
	var body struct {
		Managed bool `json:"managed"`
	}
	if err := json.NewDecoder(recorder.Body).Decode(&body); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if body.Managed {
		t.Fatal("bare config reported as managed recipe")
	}
}

func TestRegisterRecipePackageRoutesEnforceCanonicalPathsAndMethods(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "runtime-config.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.3\nrouting:\n  modelCards: []\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("VLLM_SR_ACTIVE_RECIPE_DIR", "")
	t.Setenv("VLLM_SR_RECIPE_STORE_DIR", filepath.Join(dir, "recipe-store"))
	mux := http.NewServeMux()
	registerRecipeRoutes(mux, &config.Config{
		ConfigDir:             dir,
		AbsConfigPath:         configPath,
		RouterAPIURL:          "http://router.invalid",
		RuntimeConfigWritable: true,
		RecipeStoreWritable:   true,
	})
	tests := []struct {
		method string
		path   string
		body   string
		status int
	}{
		{method: http.MethodGet, path: "/api/recipe/packages", status: http.StatusOK},
		{method: http.MethodPost, path: "/api/recipe/packages", status: http.StatusMethodNotAllowed},
		{method: http.MethodGet, path: "/api/recipe/packages/extra", status: http.StatusNotFound},
		{method: http.MethodPost, path: "/api/recipe/import", body: `{"unknown":true}`, status: http.StatusBadRequest},
		{method: http.MethodPost, path: "/api/recipe/import/anything", body: `{}`, status: http.StatusNotFound},
		{method: http.MethodPost, path: "/api/recipe/activate/anything", body: `{}`, status: http.StatusNotFound},
		{method: http.MethodPost, path: "/api/recipe/deactivate/anything", body: `{}`, status: http.StatusNotFound},
		{method: http.MethodPost, path: "/api/recipe/deactivate", status: http.StatusOK},
		{method: http.MethodPost, path: "/api/recipe/deactivate/", body: `{}`, status: http.StatusOK},
	}
	for _, test := range tests {
		response := httptest.NewRecorder()
		mux.ServeHTTP(response, httptest.NewRequest(test.method, test.path, strings.NewReader(test.body)))
		if response.Code != test.status {
			t.Fatalf("%s %s status=%d want=%d body=%s", test.method, test.path, response.Code, test.status, response.Body.String())
		}
		if !strings.Contains(response.Header().Get("Cache-Control"), "no-store") {
			t.Fatalf("%s %s missing no-store", test.method, test.path)
		}
	}
}

func TestRegisterCoreRoutesExposesReadOnlyModelCatalogEndpoint(t *testing.T) {
	t.Parallel()

	mux := http.NewServeMux()
	cfg := &config.Config{ConfigDir: t.TempDir(), PythonPath: "python3"}
	registerCoreRoutes(mux, cfg, setupmode.New(cfg.AbsConfigPath, cfg.SetupMode))

	response := httptest.NewRecorder()
	mux.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/api/models/catalog", nil))
	if response.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST /api/models/catalog status=%d want=%d body=%s", response.Code, http.StatusMethodNotAllowed, response.Body.String())
	}
}

func TestRegisterConfigRoutesKeepsInferenceVerificationAvailableInReadonlyMode(t *testing.T) {
	t.Parallel()

	mux := http.NewServeMux()
	registerConfigRoutes(mux, &config.Config{
		AbsConfigPath:         "active-config.yaml",
		ReadonlyMode:          true,
		RuntimeConfigWritable: false,
	})

	response := httptest.NewRecorder()
	mux.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/api/models/verify", nil))
	if response.Code != http.StatusMethodNotAllowed {
		t.Fatalf("GET /api/models/verify status=%d want=%d body=%s", response.Code, http.StatusMethodNotAllowed, response.Body.String())
	}
	if response.Header().Get("Allow") != http.MethodPost {
		t.Fatalf("GET /api/models/verify Allow=%q", response.Header().Get("Allow"))
	}
}

func TestRecipeActivationStartupRecoveryHonorsRuntimeMutationCapabilities(t *testing.T) {
	tests := []struct {
		name      string
		config    config.Config
		wantCalls int
	}{
		{name: "global readonly", config: config.Config{ReadonlyMode: true, RuntimeConfigWritable: true}},
		{name: "runtime config readonly", config: config.Config{RuntimeConfigWritable: false}},
		{name: "runtime config writable", config: config.Config{RuntimeConfigWritable: true}, wantCalls: 1},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			assertRecipeActivationStartupRecovery(t, &test.config, test.wantCalls)
		})
	}
}

func assertRecipeActivationStartupRecovery(t *testing.T, cfg *config.Config, wantCalls int) {
	t.Helper()
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	journalPath := filepath.Join(dir, "activation.json")
	writeCoreRouteTestFile(t, configPath, "original config\n")
	writeCoreRouteTestFile(t, journalPath, "pending journal\n")

	calls := 0
	recoverRecipeActivationOnStartup(cfg, func(context.Context) error {
		calls++
		if err := os.WriteFile(configPath, []byte("recovered config\n"), 0o600); err != nil {
			return err
		}
		return os.WriteFile(journalPath, []byte("recovered journal\n"), 0o600)
	})

	if calls != wantCalls {
		t.Fatalf("recovery calls = %d, want %d", calls, wantCalls)
	}
	wantConfig, wantJournal := "original config\n", "pending journal\n"
	if wantCalls == 1 {
		wantConfig, wantJournal = "recovered config\n", "recovered journal\n"
	}
	assertCoreRouteTestFile(t, configPath, wantConfig)
	assertCoreRouteTestFile(t, journalPath, wantJournal)
}

func TestRegisterRecipeRoutesKeepsImportAvailableWhenRuntimeConfigReadonly(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	writeCoreRouteTestFile(t, configPath, "version: v0.3\n")
	t.Setenv("VLLM_SR_ACTIVE_RECIPE_DIR", "")
	t.Setenv("VLLM_SR_RECIPE_STORE_DIR", filepath.Join(dir, "recipe-store"))

	mux := http.NewServeMux()
	registerRecipeRoutes(mux, &config.Config{
		ConfigDir:             dir,
		AbsConfigPath:         configPath,
		RouterAPIURL:          "http://router.invalid",
		RuntimeConfigWritable: false,
		RecipeStoreWritable:   true,
	})
	response := httptest.NewRecorder()
	mux.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/api/recipe/import", strings.NewReader(`{"unknown":true}`)))
	if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), `"error":"invalid_request"`) {
		t.Fatalf("runtime-readonly import status=%d body=%s", response.Code, response.Body.String())
	}
}

func writeCoreRouteTestFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("WriteFile(%s): %v", path, err)
	}
}

func assertCoreRouteTestFile(t *testing.T, path, want string) {
	t.Helper()
	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile(%s): %v", path, err)
	}
	if string(content) != want {
		t.Fatalf("%s = %q, want %q", path, content, want)
	}
}

func TestRuntimeConfigCapabilityGuardsLocalWriteRoutesButNotKBS(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.3\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	routerAPI := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/storage/knowledge-bases/example" || r.Method != http.MethodPut {
			t.Fatalf("unexpected KBS proxy request: %s %s", r.Method, r.URL.Path)
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer routerAPI.Close()

	cfg := &config.Config{
		AbsConfigPath:         configPath,
		ConfigDir:             dir,
		RouterAPIURL:          routerAPI.URL,
		RuntimeConfigWritable: false,
		RecipeStoreWritable:   true,
	}
	mux := http.NewServeMux()
	registerHealthAndSetupRoutes(mux, cfg, setupmode.New(cfg.AbsConfigPath, cfg.SetupMode))
	registerConfigRoutes(mux, cfg)

	for _, target := range []struct {
		method string
		path   string
	}{
		{method: http.MethodPost, path: "/api/setup/activate"},
		{method: http.MethodPost, path: "/api/router/config/update"},
		{method: http.MethodPost, path: "/api/router/config/deploy"},
		{method: http.MethodPost, path: "/api/router/config/rollback"},
		{method: http.MethodPost, path: "/api/router/config/global/update"},
		{method: http.MethodPost, path: "/api/router/config/global/raw/update"},
	} {
		response := httptest.NewRecorder()
		mux.ServeHTTP(response, httptest.NewRequest(target.method, target.path, strings.NewReader(`{}`)))
		if response.Code != http.StatusForbidden {
			t.Fatalf("%s %s status=%d want=%d body=%s", target.method, target.path, response.Code, http.StatusForbidden, response.Body.String())
		}
	}

	for _, target := range []struct {
		method string
		path   string
	}{
		{method: http.MethodGet, path: "/api/router/config/defaults"},
		{method: http.MethodPost, path: "/api/router/config/defaults/update"},
	} {
		response := httptest.NewRecorder()
		mux.ServeHTTP(response, httptest.NewRequest(target.method, target.path, nil))
		if response.Code != http.StatusNotFound {
			t.Fatalf("removed alias %s %s status=%d want=%d", target.method, target.path, response.Code, http.StatusNotFound)
		}
	}

	response := httptest.NewRecorder()
	mux.ServeHTTP(response, httptest.NewRequest(http.MethodPut, "/api/router/api/v1/storage/knowledge-bases/example", strings.NewReader(`{}`)))
	if response.Code != http.StatusNoContent {
		t.Fatalf("KBS proxy status=%d want=%d body=%s", response.Code, http.StatusNoContent, response.Body.String())
	}
}

func TestResolveToolsDBPathUsesRouterContractPath(t *testing.T) {
	configDir := t.TempDir()
	configPath := filepath.Join(configDir, "config.yaml")
	if err := os.WriteFile(configPath, []byte(`
version: v0.3
global:
  integrations:
    tools:
      tools_db_path: "/tmp/custom-tools.json"
`), 0o644); err != nil {
		t.Fatalf("WriteFile(config): %v", err)
	}

	got := resolveToolsDBPath(&config.Config{
		AbsConfigPath: configPath,
		ConfigDir:     configDir,
	})
	if got != "/tmp/custom-tools.json" {
		t.Fatalf("resolveToolsDBPath() = %q, want %q", got, "/tmp/custom-tools.json")
	}
}

func TestResolveToolsDBPathFallsBackWhenRouterContractCannotParse(t *testing.T) {
	configDir := t.TempDir()
	configPath := filepath.Join(configDir, "config.yaml")
	if err := os.WriteFile(configPath, []byte("routing: ["), 0o644); err != nil {
		t.Fatalf("WriteFile(config): %v", err)
	}

	got := resolveToolsDBPath(&config.Config{
		AbsConfigPath: configPath,
		ConfigDir:     configDir,
	})
	// Was filepath.Join(configDir, "config", ...), which repeated the config
	// directory: ConfigDir is already the directory holding config.yaml, so the
	// fallback named <dir>/config/config/tools_db.json in a real checkout. The
	// default is relative to the project root, which is ConfigDir's parent —
	// the same reading of ConfigDir that mlTrainingDir already uses.
	want := filepath.Join(filepath.Dir(configDir), defaultToolsDBPath)
	if got != want {
		t.Fatalf("resolveToolsDBPath() = %q, want %q", got, want)
	}
}
