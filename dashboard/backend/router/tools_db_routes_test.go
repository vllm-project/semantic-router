package router

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/routercontract"
)

func TestToolsDBRouteFollowsSavedConfig(t *testing.T) {
	root := t.TempDir()
	t.Setenv("VLLM_SR_CONFIG_BASE_DIR", root)
	configPath := filepath.Join(root, "runtime.yaml")
	first := filepath.Join(root, "first.json")
	second := filepath.Join(t.TempDir(), "second.json")
	for path, name := range map[string]string{first: "first", second: "second"} {
		if err := os.WriteFile(path, []byte(fmt.Sprintf(`[{"name":%q}]`, name)), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	saveConfig := func(path string) {
		t.Helper()
		body := fmt.Sprintf("version: v0.3\nglobal:\n  integrations:\n    tools:\n      tools_db_path: %q\n", path)
		if err := os.WriteFile(configPath, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	saveConfig("first.json")
	mux := http.NewServeMux()
	registerToolRoutes(mux, &config.Config{AbsConfigPath: configPath, ConfigDir: root, ConfigBaseDir: root})
	assertDatabase := func(name string) {
		t.Helper()
		response := httptest.NewRecorder()
		mux.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/api/tools-db", nil))
		want := fmt.Sprintf(`[{"name":%q}]`, name)
		if response.Code != http.StatusOK || strings.TrimSpace(response.Body.String()) != want {
			t.Fatalf("GET = %d %s; want %s", response.Code, response.Body.String(), want)
		}
	}
	assertDatabase("first")
	// ConfigPage refreshes /api/tools-db when the persisted tools_db_path changes.
	saveConfig(second)
	assertDatabase("second")
}

func TestToolsDBRouteResourcePathContract(t *testing.T) {
	for _, tc := range []struct {
		name, configured, contents       string
		invalidConfig, missing, absolute bool
		wantStatus                       int
	}{
		{name: "relative", configured: "catalogue/tools.json", contents: `[{"name":"fixture"}]`, wantStatus: http.StatusOK},
		{name: "canonical-default", contents: `[]`, wantStatus: http.StatusOK},
		{name: "unparsable-default", invalidConfig: true, contents: `[]`, wantStatus: http.StatusOK},
		{name: "absolute", absolute: true, contents: `[]`, wantStatus: http.StatusOK},
		{name: "missing", missing: true, wantStatus: http.StatusNotFound},
		{name: "invalid-json", contents: `[`, wantStatus: http.StatusInternalServerError},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// A root literally named config must not be guessed to mean its parent.
			root := filepath.Join(t.TempDir(), "config")
			if err := os.MkdirAll(root, 0o755); err != nil {
				t.Fatal(err)
			}
			t.Chdir(t.TempDir())
			t.Setenv("VLLM_SR_CONFIG_BASE_DIR", root)
			configured := tc.configured
			if tc.absolute {
				configured = filepath.Join(t.TempDir(), "external.json")
			}
			path := configured
			if path == "" {
				path = defaultToolsDBPath
			}
			if !filepath.IsAbs(path) {
				path = filepath.Join(root, path)
			}
			if !tc.missing {
				if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, []byte(tc.contents), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			configPath := filepath.Join(t.TempDir(), "runtime.yaml")
			body := "version: v0.3\nglobal:\n  integrations:\n    tools:\n      enabled: true\n"
			if configured != "" {
				body += fmt.Sprintf("      tools_db_path: %q\n", configured)
			}
			if tc.invalidConfig {
				body = "global: ["
			}
			if err := os.WriteFile(configPath, []byte(body), 0o600); err != nil {
				t.Fatal(err)
			}
			if !tc.invalidConfig {
				if _, err := routercontract.ReadToolSelection(configPath); err != nil {
					t.Fatalf("normal canonical fixture did not parse: %v", err)
				}
			}
			cfg := &config.Config{AbsConfigPath: configPath, ConfigDir: t.TempDir(), ConfigBaseDir: root}
			mux := http.NewServeMux()
			registerToolRoutes(mux, cfg)
			response := httptest.NewRecorder()
			mux.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/api/tools-db", nil))
			if response.Code != tc.wantStatus {
				t.Fatalf("GET = %d: %s", response.Code, response.Body.String())
			}
			if tc.wantStatus == http.StatusOK && strings.TrimSpace(response.Body.String()) != tc.contents {
				t.Fatalf("served %q, want configured file %q", response.Body.String(), tc.contents)
			}
			// This API is read-only; a normal PUT must keep the selected file intact.
			response = httptest.NewRecorder()
			mux.ServeHTTP(response, httptest.NewRequest(http.MethodPut, "/api/tools-db", strings.NewReader(`[]`)))
			if response.Code != http.StatusMethodNotAllowed {
				t.Fatalf("PUT = %d", response.Code)
			}
			if !tc.missing {
				data, err := os.ReadFile(path)
				if err != nil || string(data) != tc.contents {
					t.Fatalf("GET/PUT changed tools database: %q, %v", data, err)
				}
			}
		})
	}
}

func TestToolsDBRouteKeepsExplicitContainerAssetRoot(t *testing.T) {
	root := t.TempDir()
	t.Chdir(root)
	t.Setenv("VLLM_SR_CONFIG_BASE_DIR", root)
	configPath := filepath.Join(root, ".vllm-sr", "runtime-config.yaml")
	toolsPath := filepath.Join(root, "config", "tools_db.json")
	for _, path := range []string{configPath, toolsPath} {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(configPath, []byte("version: v0.3\nglobal:\n  integrations:\n    tools:\n      enabled: true\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := routercontract.ReadToolSelection(configPath); err != nil {
		t.Fatalf("normal canonical fixture did not parse: %v", err)
	}
	if err := os.WriteFile(toolsPath, []byte(`[{"name":"fixture"}]`), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg := &config.Config{AbsConfigPath: configPath, ConfigDir: root, ConfigBaseDir: root}
	mux := http.NewServeMux()
	registerToolRoutes(mux, cfg)
	response := httptest.NewRecorder()
	mux.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/api/tools-db", nil))
	if response.Code != http.StatusOK {
		t.Fatalf("GET /api/tools-db = %d: %s", response.Code, response.Body.String())
	}
}

func TestToolsDBRouteReadsRepositoryConfigFromDevWorkingDirectory(t *testing.T) {
	repoRoot, err := filepath.Abs("../../..")
	if err != nil {
		t.Fatal(err)
	}
	t.Chdir(filepath.Join(repoRoot, "dashboard", "backend"))
	t.Setenv("VLLM_SR_CONFIG_BASE_DIR", repoRoot)
	cfg := &config.Config{
		AbsConfigPath: filepath.Join(repoRoot, "config", "config.yaml"),
		ConfigDir:     filepath.Join(repoRoot, "config"),
		ConfigBaseDir: repoRoot,
	}
	mux := http.NewServeMux()
	registerToolRoutes(mux, cfg)
	response := httptest.NewRecorder()
	mux.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/api/tools-db", nil))
	if response.Code != http.StatusOK {
		t.Fatalf("GET /api/tools-db = %d: %s", response.Code, response.Body.String())
	}
	expected, err := os.ReadFile(filepath.Join(repoRoot, "config", "runtime", "tools", "tools_db.json"))
	if err != nil {
		t.Fatal(err)
	}
	var actualJSON, expectedJSON any
	if err := json.Unmarshal(response.Body.Bytes(), &actualJSON); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(expected, &expectedJSON); err != nil {
		t.Fatal(err)
	}
	actual, _ := json.Marshal(actualJSON)
	want, _ := json.Marshal(expectedJSON)
	if string(actual) != string(want) {
		t.Fatal("GET /api/tools-db did not serve the canonical configured database")
	}
}
