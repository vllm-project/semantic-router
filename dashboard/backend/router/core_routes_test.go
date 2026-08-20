package router

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
)

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
	if err := os.WriteFile(configPath, []byte("version: v0.3\n"), 0o600); err != nil {
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
		if r.URL.Path != "/config/kbs/example" || r.Method != http.MethodPost {
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
	registerSecurityPolicyRoutes(mux, cfg)

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
		{method: http.MethodPost, path: "/api/router/config/defaults/update"},
		{method: http.MethodPut, path: "/api/security/policy"},
		{method: http.MethodPost, path: "/api/security/policy/preview"},
	} {
		response := httptest.NewRecorder()
		mux.ServeHTTP(response, httptest.NewRequest(target.method, target.path, strings.NewReader(`{}`)))
		if response.Code != http.StatusForbidden {
			t.Fatalf("%s %s status=%d want=%d body=%s", target.method, target.path, response.Code, http.StatusForbidden, response.Body.String())
		}
	}

	response := httptest.NewRecorder()
	mux.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/api/router/config/kbs/example", strings.NewReader(`{}`)))
	if response.Code != http.StatusNoContent {
		t.Fatalf("KBS proxy status=%d want=%d body=%s", response.Code, http.StatusNoContent, response.Body.String())
	}
}

func TestResolveEvaluationProjectRootFallsBackToWorkingDirectoryRepo(t *testing.T) {
	repoRoot := t.TempDir()
	scriptPath := filepath.Join(repoRoot, "src", "training", "model_eval", "mmlu_pro_vllm_eval.py")
	if err := os.MkdirAll(filepath.Dir(scriptPath), 0o755); err != nil {
		t.Fatalf("MkdirAll(script dir): %v", err)
	}
	if err := os.WriteFile(scriptPath, []byte("print('ok')\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(script): %v", err)
	}
	signalScriptPath := filepath.Join(repoRoot, "src", "training", "model_eval", "signal_eval.py")
	if err := os.WriteFile(signalScriptPath, []byte("print('ok')\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(signal script): %v", err)
	}
	if err := os.MkdirAll(filepath.Join(repoRoot, "dashboard", "backend"), 0o755); err != nil {
		t.Fatalf("MkdirAll(dashboard/backend): %v", err)
	}

	workDir := filepath.Join(repoRoot, "dashboard", "backend")
	previousWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd(): %v", err)
	}
	chdirErr := os.Chdir(workDir)
	if chdirErr != nil {
		t.Fatalf("Chdir(%s): %v", workDir, chdirErr)
	}
	t.Cleanup(func() {
		_ = os.Chdir(previousWD)
	})

	externalConfigDir := filepath.Join(t.TempDir(), "config")
	mkdirErr := os.MkdirAll(externalConfigDir, 0o755)
	if mkdirErr != nil {
		t.Fatalf("MkdirAll(config dir): %v", mkdirErr)
	}

	cfg := &config.Config{ConfigDir: externalConfigDir}
	got := resolveEvaluationProjectRoot(cfg)
	gotResolved, err := filepath.EvalSymlinks(got)
	if err != nil {
		t.Fatalf("EvalSymlinks(got): %v", err)
	}
	wantResolved, err := filepath.EvalSymlinks(repoRoot)
	if err != nil {
		t.Fatalf("EvalSymlinks(repoRoot): %v", err)
	}
	if gotResolved != wantResolved {
		t.Fatalf("resolveEvaluationProjectRoot() = %q, want %q", got, repoRoot)
	}
}

func TestResolveEvaluationProjectRootRecognizesRuntimeAppLayout(t *testing.T) {
	appRoot := t.TempDir()
	scriptDir := filepath.Join(appRoot, "src", "training", "model_eval")
	if err := os.MkdirAll(scriptDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(script dir): %v", err)
	}
	for _, scriptName := range []string{"mmlu_pro_vllm_eval.py", "signal_eval.py"} {
		scriptPath := filepath.Join(scriptDir, scriptName)
		if err := os.WriteFile(scriptPath, []byte("print('ok')\n"), 0o644); err != nil {
			t.Fatalf("WriteFile(%s): %v", scriptName, err)
		}
	}

	previousWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd(): %v", err)
	}
	chdirErr := os.Chdir(appRoot)
	if chdirErr != nil {
		t.Fatalf("Chdir(%s): %v", appRoot, chdirErr)
	}
	t.Cleanup(func() {
		_ = os.Chdir(previousWD)
	})

	cfg := &config.Config{ConfigDir: string(filepath.Separator)}
	got := resolveEvaluationProjectRoot(cfg)
	gotResolved, err := filepath.EvalSymlinks(got)
	if err != nil {
		t.Fatalf("EvalSymlinks(got): %v", err)
	}
	wantResolved, err := filepath.EvalSymlinks(appRoot)
	if err != nil {
		t.Fatalf("EvalSymlinks(appRoot): %v", err)
	}
	if gotResolved != wantResolved {
		t.Fatalf("resolveEvaluationProjectRoot() = %q, want %q", got, appRoot)
	}
}

func TestResolveToolsDBPathUsesRouterContractPath(t *testing.T) {
	configDir := t.TempDir()
	configPath := filepath.Join(configDir, "config.yaml")
	if err := os.WriteFile(configPath, []byte(`
version: "0.3"
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
	want := filepath.Join(configDir, "config", "tools_db.json")
	if got != want {
		t.Fatalf("resolveToolsDBPath() = %q, want %q", got, want)
	}
}
