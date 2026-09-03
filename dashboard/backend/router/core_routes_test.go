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

	dashboardauth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/evaluationplane"
	"github.com/vllm-project/semantic-router/dashboard/backend/recipe"
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

func TestRegisterEvaluationPlaneRoutesExposeCurrentContract(t *testing.T) {
	root := t.TempDir()
	t.Setenv("VLLM_SR_SOURCE_REVISION", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
	configPath := filepath.Join(root, "config.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.3\nrouting:\n  modelCards: []\n"), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}
	mux := http.NewServeMux()
	cfg := &config.Config{
		EvaluationEnabled: true,
		EvaluationDataDir: filepath.Join(root, "evaluation"),
		PythonPath:        "python3",
		AbsConfigPath:     configPath,
		RouterAPIURL:      "http://router.internal",
		EnvoyURL:          "http://envoy.internal",
	}
	registerEvaluationRoutes(mux, cfg)
	if !cfg.EvaluationAvailable || cfg.EvaluationUnavailableReason != "" {
		t.Fatalf("evaluation availability = (%t, %q), want ready", cfg.EvaluationAvailable, cfg.EvaluationUnavailableReason)
	}

	catalog := httptest.NewRecorder()
	mux.ServeHTTP(catalog, httptest.NewRequest(http.MethodGet, "/api/evaluation/v1/catalog", nil))
	if catalog.Code != http.StatusOK || strings.Contains(catalog.Body.String(), "router.internal") || strings.Contains(catalog.Body.String(), "envoy.internal") {
		t.Fatalf("catalog status=%d body=%s", catalog.Code, catalog.Body.String())
	}

	create := httptest.NewRecorder()
	mux.ServeHTTP(create, evaluationAdminRequest(httptest.NewRequest(http.MethodPost, "/api/evaluation/v1/runs", strings.NewReader(`{
		"client_request_id":"202f2f29-c28f-461d-b860-30352c1ab3f9",
		"name":"route test","description":"","suite_ids":["evaluation-smoke"],"track_ids":["routing"],
		"mode":"replay","target_id":"fixture","change_profile":"schema_adapter",
		"sample_limit":4,"concurrency":1,"seed":17
	}`))))
	if create.Code != http.StatusCreated {
		t.Fatalf("create status=%d body=%s", create.Code, create.Body.String())
	}
	ledgerResponse := httptest.NewRecorder()
	mux.ServeHTTP(ledgerResponse, evaluationAdminRequest(
		httptest.NewRequest(http.MethodGet, "/api/evaluation/v1/runs", nil),
	))
	var ledger evaluationplane.RunLedger
	if ledgerResponse.Code != http.StatusOK || json.NewDecoder(ledgerResponse.Body).Decode(&ledger) != nil || !ledger.LedgerComplete || len(ledger.Runs) != 1 {
		t.Fatalf("run ledger route status=%d body=%s", ledgerResponse.Code, ledgerResponse.Body.String())
	}

	proxyCalls := 0
	mux.HandleFunc("/api/", func(w http.ResponseWriter, _ *http.Request) {
		proxyCalls++
		w.WriteHeader(http.StatusBadGateway)
	})
	for _, unknownPath := range []string{
		"/api/evaluation",
		"/api/evaluation/",
		"/api/evaluation/tasks",
		"/api/evaluation/tasks/obsolete-run",
		"/api/evaluation/datasets",
		"/api/evaluation/v1/unknown",
	} {
		response := httptest.NewRecorder()
		mux.ServeHTTP(response, httptest.NewRequest(http.MethodGet, unknownPath, nil))
		if response.Code != http.StatusNotFound {
			t.Fatalf("unknown Evaluation route %s status=%d, want 404", unknownPath, response.Code)
		}
		if response.Header().Get("Cache-Control") != "private, no-store" {
			t.Fatalf("unknown Evaluation route %s Cache-Control=%q", unknownPath, response.Header().Get("Cache-Control"))
		}
	}
	if proxyCalls != 0 {
		t.Fatalf("unknown Evaluation routes reached /api/ fallback %d times", proxyCalls)
	}
}

func TestRegisterEvaluationPlaneRoutesFreezeUnavailableState(t *testing.T) {
	t.Run("disabled", func(t *testing.T) {
		cfg := &config.Config{EvaluationEnabled: false}
		if service := registerEvaluationRoutes(http.NewServeMux(), cfg); service != nil {
			t.Fatal("disabled Evaluation returned a service")
		}
		if cfg.EvaluationAvailable || cfg.EvaluationUnavailableReason == "" {
			t.Fatalf("disabled availability = (%t, %q)", cfg.EvaluationAvailable, cfg.EvaluationUnavailableReason)
		}
	})

	t.Run("initialization failure", func(t *testing.T) {
		root := t.TempDir()
		blockedDataDir := filepath.Join(root, "not-a-directory")
		if err := os.WriteFile(blockedDataDir, []byte("blocked"), 0o600); err != nil {
			t.Fatalf("write blocking file: %v", err)
		}
		cfg := &config.Config{
			EvaluationEnabled: true,
			EvaluationDataDir: blockedDataDir,
			PythonPath:        "python3",
		}
		if service := registerEvaluationRoutes(http.NewServeMux(), cfg); service != nil {
			t.Fatal("failed Evaluation initialization returned a service")
		}
		if cfg.EvaluationAvailable || cfg.EvaluationUnavailableReason == "" {
			t.Fatalf("failed availability = (%t, %q)", cfg.EvaluationAvailable, cfg.EvaluationUnavailableReason)
		}
		if strings.Contains(cfg.EvaluationUnavailableReason, blockedDataDir) {
			t.Fatalf("public reason leaked server path: %q", cfg.EvaluationUnavailableReason)
		}
	})
}

func TestEvaluationNamespaceBoundaryRemainsClosedWhenDisabled(t *testing.T) {
	mux := http.NewServeMux()
	registerEvaluationRoutes(mux, &config.Config{EvaluationEnabled: false})
	proxyCalls := 0
	mux.HandleFunc("/api/", func(w http.ResponseWriter, _ *http.Request) {
		proxyCalls++
		w.WriteHeader(http.StatusBadGateway)
	})

	response := httptest.NewRecorder()
	mux.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/api/evaluation/tasks/obsolete-run/start", nil))
	if response.Code != http.StatusNotFound {
		t.Fatalf("disabled Evaluation namespace status=%d, want 404 body=%s", response.Code, response.Body.String())
	}
	if response.Header().Get("Cache-Control") != "private, no-store" {
		t.Fatalf("disabled Evaluation namespace Cache-Control=%q", response.Header().Get("Cache-Control"))
	}
	if proxyCalls != 0 {
		t.Fatalf("disabled Evaluation namespace reached /api/ fallback %d times", proxyCalls)
	}
}

func TestEvaluationRoutesFailClosedWhenOnlyManagementCredentialExists(t *testing.T) {
	root := t.TempDir()
	configPath := filepath.Join(root, "config.yaml")
	configYAML := `version: v0.3
global:
  router:
    auto_model_names: [test-mom]
providers:
  defaults:
    default_model: model-fast
  models:
    - name: model-fast
      backend_refs: [{endpoint: fast.models.test:8000}]
    - name: model-strong
      backend_refs: [{endpoint: strong.models.test:8000}]
routing:
  modelCards:
    - {name: model-fast, modality: text}
    - {name: model-strong, modality: text}
  decisions:
    - name: route
      rules: {}
      modelRefs: [{model: model-fast}, {model: model-strong}]
`
	if err := os.WriteFile(configPath, []byte(configYAML), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}
	t.Setenv(recipe.ManagementCredentialEnv, "")
	store := recipe.NewStore(recipe.StoreOptions{
		Root: filepath.Join(root, "recipe-store"), ConfigPath: configPath,
	})
	_, credentialErr := store.EnsureManagementCredential()
	if credentialErr != nil {
		t.Fatalf("EnsureManagementCredential: %v", credentialErr)
	}
	mux := http.NewServeMux()
	evaluationDir := filepath.Join(root, "evaluation")
	t.Setenv("VLLM_SR_SOURCE_REVISION", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
	registerEvaluationRoutes(mux, &config.Config{
		EvaluationEnabled: true, EvaluationDataDir: evaluationDir, PythonPath: "python3",
		AbsConfigPath: configPath, RouterAPIURL: "http://router.internal",
	}, store)
	catalogResponse := httptest.NewRecorder()
	mux.ServeHTTP(catalogResponse, httptest.NewRequest(http.MethodGet, "/api/evaluation/v1/catalog", nil))
	if catalogResponse.Code != http.StatusOK {
		t.Fatalf("catalog status=%d body=%s", catalogResponse.Code, catalogResponse.Body.String())
	}
	var catalog evaluationplane.Catalog
	if err := json.NewDecoder(catalogResponse.Body).Decode(&catalog); err != nil {
		t.Fatalf("decode catalog: %v", err)
	}
	var mixture *evaluationplane.CatalogTarget
	for index := range catalog.Targets {
		if catalog.Targets[index].Kind == "mixture-of-models" {
			mixture = &catalog.Targets[index]
			break
		}
	}
	if mixture == nil || mixture.Labels["router_auth"] != "dedicated-evaluation-credential-unavailable" || len(mixture.TrackIDs) != 0 {
		t.Fatalf("Mixture target did not fail closed without a dedicated evaluation credential: %#v", mixture)
	}

	createBody := strings.Replace(`{
		"client_request_id":"d032dcf7-c76b-493c-9b50-2d159448a637",
		"name":"live routing","description":"","suite_ids":["live-mom-core"],"track_ids":["routing"],
		"mode":"live","target_id":"__MOM_TARGET__","change_profile":"recipe",
		"sample_limit":4,"concurrency":1,"seed":17
	}`, "__MOM_TARGET__", mixture.ID, 1)
	create := httptest.NewRecorder()
	mux.ServeHTTP(create, evaluationAdminRequest(httptest.NewRequest(http.MethodPost, "/api/evaluation/v1/runs", strings.NewReader(createBody))))
	if create.Code != http.StatusBadRequest || !strings.Contains(create.Body.String(), "target cannot execute") {
		t.Fatalf("create status=%d body=%s, want fail-closed 400", create.Code, create.Body.String())
	}
	runs, err := os.ReadDir(filepath.Join(evaluationDir, "runs"))
	if err != nil || len(runs) != 0 {
		t.Fatalf("rejected run persisted a bundle: entries=%v err=%v", runs, err)
	}
}

func evaluationAdminRequest(request *http.Request) *http.Request {
	return request.WithContext(dashboardauth.WithAuthContext(request.Context(), dashboardauth.AuthContext{
		UserID: "evaluation-route-test", Role: dashboardauth.RoleAdmin,
	}))
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
