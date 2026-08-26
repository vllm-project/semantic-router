package router

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
)

func TestRegisterCoreRoutesDoesNotExposeDashboardRecipeAuthority(t *testing.T) {
	t.Parallel()
	mux := http.NewServeMux()
	registerCoreRoutes(mux, &config.Config{ConfigDir: t.TempDir(), PythonPath: "python3"}, nil)

	for _, target := range []struct {
		method string
		path   string
	}{
		{method: http.MethodGet, path: "/api/recipe"},
		{method: http.MethodGet, path: "/api/recipe/packages"},
		{method: http.MethodPost, path: "/api/recipe/import"},
		{method: http.MethodPost, path: "/api/recipe/activate"},
		{method: http.MethodPost, path: "/api/recipe/deactivate"},
		{method: http.MethodGet, path: "/api/recipe/probes"},
	} {
		t.Run(target.method+" "+target.path, func(t *testing.T) {
			response := httptest.NewRecorder()
			mux.ServeHTTP(response, httptest.NewRequest(target.method, target.path, nil))
			if response.Code != http.StatusNotFound {
				t.Fatalf("status=%d want=%d body=%s", response.Code, http.StatusNotFound, response.Body.String())
			}
		})
	}
}

func TestRegisterCoreRoutesDoesNotExposeLegacyModelDiscoveryEndpoint(t *testing.T) {
	t.Parallel()
	mux := http.NewServeMux()
	registerCoreRoutes(mux, &config.Config{ConfigDir: t.TempDir(), PythonPath: "python3"}, nil)
	response := httptest.NewRecorder()
	mux.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/api/models/discover", nil))
	if response.Code != http.StatusNotFound {
		t.Fatalf("POST legacy model discovery status=%d want=%d body=%s", response.Code, http.StatusNotFound, response.Body.String())
	}
}

func TestRegisterCoreRoutesDoesNotExposePackagedModelCatalogEndpoint(t *testing.T) {
	t.Parallel()
	mux := http.NewServeMux()
	registerCoreRoutes(mux, &config.Config{ConfigDir: t.TempDir(), PythonPath: "python3"}, nil)
	response := httptest.NewRecorder()
	mux.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/api/models/catalog", nil))
	if response.Code != http.StatusNotFound {
		t.Fatalf("GET packaged Model catalog status=%d want=%d body=%s", response.Code, http.StatusNotFound, response.Body.String())
	}
}

func TestRegisterCoreRoutesDoesNotExposeDashboardToolExecution(t *testing.T) {
	t.Parallel()
	mux := http.NewServeMux()
	registerCoreRoutes(mux, &config.Config{ConfigDir: t.TempDir(), PythonPath: "python3"}, nil)

	for _, path := range []string{
		"/api/tools-db",
		"/api/tools/web-search",
		"/api/tools/open-web",
		"/api/tools/weather",
		"/api/tools/fetch-raw",
	} {
		t.Run(path, func(t *testing.T) {
			response := httptest.NewRecorder()
			mux.ServeHTTP(response, httptest.NewRequest(http.MethodPost, path, nil))
			if response.Code != http.StatusNotFound {
				t.Fatalf("POST %s status=%d want=%d body=%s", path, response.Code, http.StatusNotFound, response.Body.String())
			}
		})
	}
}

func TestRegisterCoreRoutesDoesNotExposeRuntimeConfigAuthority(t *testing.T) {
	t.Parallel()
	mux := http.NewServeMux()
	registerCoreRoutes(mux, &config.Config{ConfigDir: t.TempDir(), PythonPath: "python3"}, nil)

	for _, target := range []struct {
		method string
		path   string
	}{
		{method: http.MethodGet, path: "/api/setup/state"},
		{method: http.MethodPost, path: "/api/setup/validate"},
		{method: http.MethodPost, path: "/api/setup/import-remote"},
		{method: http.MethodPost, path: "/api/setup/activate"},
		{method: http.MethodGet, path: "/api/setup/presets"},
		{method: http.MethodGet, path: "/api/router/config/all"},
		{method: http.MethodGet, path: "/api/router/config/yaml"},
		{method: http.MethodPost, path: "/api/router/config/update"},
		{method: http.MethodPost, path: "/api/router/config/deploy"},
		{method: http.MethodPost, path: "/api/router/config/deploy/preview"},
		{method: http.MethodPost, path: "/api/router/config/rollback"},
		{method: http.MethodGet, path: "/api/router/config/versions"},
		{method: http.MethodGet, path: "/api/router/config/deployments"},
		{method: http.MethodGet, path: "/api/router/config/active-projection"},
		{method: http.MethodGet, path: "/api/router/config/global"},
		{method: http.MethodPost, path: "/api/router/config/global/update"},
		{method: http.MethodGet, path: "/api/router/config/global/raw"},
		{method: http.MethodPost, path: "/api/router/config/global/raw/update"},
		{method: http.MethodGet, path: "/api/router/config/defaults"},
		{method: http.MethodPost, path: "/api/router/config/defaults/update"},
		{method: http.MethodPost, path: "/api/router/config/nl/generate"},
	} {
		t.Run(target.method+" "+target.path, func(t *testing.T) {
			response := httptest.NewRecorder()
			mux.ServeHTTP(response, httptest.NewRequest(target.method, target.path, nil))
			if response.Code != http.StatusNotFound {
				t.Fatalf("status=%d want=%d body=%s", response.Code, http.StatusNotFound, response.Body.String())
			}
		})
	}
}

func TestRegisterCoreRoutesDoesNotExposeTopologyEvaluationProxy(t *testing.T) {
	t.Parallel()
	mux := http.NewServeMux()
	registerCoreRoutes(mux, &config.Config{ConfigDir: t.TempDir(), PythonPath: "python3"}, nil)

	response := httptest.NewRecorder()
	mux.ServeHTTP(
		response,
		httptest.NewRequest(http.MethodPost, "/api/topology/test-query", nil),
	)
	if response.Code != http.StatusNotFound {
		t.Fatalf(
			"POST topology evaluation proxy status=%d want=%d body=%s",
			response.Code,
			http.StatusNotFound,
			response.Body.String(),
		)
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
