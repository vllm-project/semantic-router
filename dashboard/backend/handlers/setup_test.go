package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
)

func createBootstrapSetupConfig(t *testing.T, dir string) string {
	t.Helper()

	configPath := filepath.Join(dir, "config.yaml")
	config := map[string]interface{}{
		"version": "v0.3",
		"listeners": []map[string]interface{}{
			{
				"name":    "http-8899",
				"address": "0.0.0.0",
				"port":    8899,
				"timeout": "300s",
			},
		},
		"setup": map[string]interface{}{
			"mode":  true,
			"state": "bootstrap",
		},
	}

	data, err := yaml.Marshal(config)
	if err != nil {
		t.Fatalf("failed to marshal bootstrap config: %v", err)
	}
	if err := os.WriteFile(configPath, data, 0o644); err != nil {
		t.Fatalf("failed to write bootstrap config: %v", err)
	}
	return configPath
}

// createActivatedSetupConfig writes the same config without the setup block,
// the shape the file has after activation.
func createActivatedSetupConfig(t *testing.T, dir string) string {
	t.Helper()

	configPath := filepath.Join(dir, "config.yaml")
	config := map[string]interface{}{
		"version": "v0.3",
		"listeners": []map[string]interface{}{
			{
				"name":    "http-8899",
				"address": "0.0.0.0",
				"port":    8899,
				"timeout": "300s",
			},
		},
	}

	data, err := yaml.Marshal(config)
	if err != nil {
		t.Fatalf("failed to marshal activated config: %v", err)
	}
	if err := os.WriteFile(configPath, data, 0o644); err != nil {
		t.Fatalf("failed to write activated config: %v", err)
	}
	return configPath
}

// createCanonicallyInvalidSetupConfig writes a config with a readable setup
// block and a type error in listeners.
//
// setupmode decodes only the setup block, so it resolves this file cleanly,
// while readSetupConfigFile decodes the full schema and fails. That is the only
// way to reach SetupStateHandler's read-error branch with a good resolution.
func createCanonicallyInvalidSetupConfig(t *testing.T, dir string, setupMode bool) string {
	t.Helper()

	configPath := filepath.Join(dir, "config.yaml")
	body := "version: v0.3\nlisteners: \"not-a-list\"\n"
	if setupMode {
		body += "setup:\n  mode: true\n  state: bootstrap\n"
	}
	if err := os.WriteFile(configPath, []byte(body), 0o644); err != nil {
		t.Fatalf("failed to write canonically invalid config: %v", err)
	}
	return configPath
}

// decodeSetupState also returns the raw body, because reason is omitempty and
// only the raw body can prove it is absent.
func decodeSetupState(t *testing.T, w *httptest.ResponseRecorder) (SetupStateResponse, string) {
	t.Helper()

	raw := w.Body.String()
	var resp SetupStateResponse
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		t.Fatalf("failed to decode setup state %q: %v", raw, err)
	}
	return resp, raw
}

// getSetupState issues GET /api/setup/state against the given resolver.
func getSetupState(t *testing.T, configPath string, resolver *setupmode.Resolver) (SetupStateResponse, string) {
	t.Helper()

	w := httptest.NewRecorder()
	SetupStateHandler(configPath, resolver)(w, httptest.NewRequest(http.MethodGet, "/api/setup/state", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("setup state status = %d, want 200; body=%s", w.Code, w.Body.String())
	}
	return decodeSetupState(t, w)
}

func TestSummarizeSetupConfigIncludesRecipeOwnedSignalsAndDecisions(t *testing.T) {
	var config setupConfigFile
	err := yaml.Unmarshal([]byte(`
version: v0.3
providers:
  models:
    - name: shared-model
routing:
  modelCards:
    - name: shared-model
  decisions: []
entrypoints:
  - model_names: [vllm-sr/balanced]
    recipe: balanced
recipes:
  - name: balanced
    routing:
      signals:
        keywords:
          - name: balanced-keyword
            operator: OR
            keywords: [balanced]
      decisions:
        - name: balanced-route
          priority: 100
          rules:
            operator: AND
            conditions:
              - type: keyword
                name: balanced-keyword
          modelRefs:
            - model: shared-model
              use_reasoning: false
  - name: private
    routing:
      signals:
        pii:
          - name: private-pii
            threshold: 0.8
      decisions:
        - name: private-route
          priority: 100
          rules:
            operator: AND
            conditions:
              - type: pii
                name: private-pii
          modelRefs:
            - model: shared-model
              use_reasoning: false
`), &config)
	if err != nil {
		t.Fatalf("unmarshal config: %v", err)
	}

	summary := summarizeSetupConfig(&config.CanonicalConfig)

	if summary.Models != 1 || summary.Decisions != 2 || summary.Signals != 2 {
		t.Fatalf("summary = %+v, want models=1 decisions=2 signals=2", summary)
	}

	merged := mergeSetupCanonicalConfig(setupConfigFile{}.CanonicalConfig, config.CanonicalConfig)
	if len(merged.Entrypoints) != 1 || len(merged.Recipes) != 2 {
		t.Fatalf(
			"setup merge dropped scoped routing: entrypoints=%d recipes=%d",
			len(merged.Entrypoints),
			len(merged.Recipes),
		)
	}
}

func createValidSetupPatch() map[string]interface{} {
	return map[string]interface{}{
		"providers": map[string]interface{}{
			"defaults": map[string]interface{}{
				"default_model": "test-model",
			},
			"models": []map[string]interface{}{
				{
					"name": "test-model",
					"backend_refs": []map[string]interface{}{
						{
							"name":     "primary",
							"endpoint": "host.docker.internal:8000",
							"protocol": "http",
							"weight":   1,
						},
					},
				},
			},
		},
		"routing": map[string]interface{}{
			"modelCards": []map[string]interface{}{
				{
					"name":     "test-model",
					"modality": "text",
				},
			},
			"signals": map[string]interface{}{
				"domains": []map[string]interface{}{
					{
						"name":        "other",
						"description": "General requests",
					},
				},
				"keywords": []map[string]interface{}{
					{
						"name":           "test_keywords",
						"operator":       "OR",
						"keywords":       []string{"test"},
						"case_sensitive": false,
					},
				},
			},
			"decisions": []map[string]interface{}{
				{
					"name":        "default_route",
					"description": "Default setup route",
					"priority":    100,
					"rules": map[string]interface{}{
						"operator": "AND",
						"conditions": []map[string]interface{}{
							{
								"type": "domain",
								"name": "other",
							},
							{
								"type": "keyword",
								"name": "test_keywords",
							},
						},
					},
					"modelRefs": []map[string]interface{}{
						{
							"model":         "test-model",
							"use_reasoning": false,
						},
					},
				},
			},
		},
	}
}

func mustJSONRaw(t *testing.T, value interface{}) json.RawMessage {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("failed to marshal JSON payload: %v", err)
	}
	return json.RawMessage(data)
}

func TestSetupStateHandler(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createBootstrapSetupConfig(t, tempDir)

	req := httptest.NewRequest(http.MethodGet, "/api/setup/state", nil)
	w := httptest.NewRecorder()

	SetupStateHandler(configPath, setupmode.New(configPath, false))(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var resp SetupStateResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if !resp.SetupMode {
		t.Fatalf("expected setupMode=true")
	}
	if resp.ListenerPort != 8899 {
		t.Fatalf("expected listenerPort=8899, got %d", resp.ListenerPort)
	}
	if resp.Models != 0 || resp.Decisions != 0 {
		t.Fatalf("expected empty bootstrap counts, got models=%d decisions=%d", resp.Models, resp.Decisions)
	}
}

func TestSetupValidateHandler(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createBootstrapSetupConfig(t, tempDir)

	body, err := json.Marshal(SetupConfigRequest{Config: mustJSONRaw(t, createValidSetupPatch())})
	if err != nil {
		t.Fatalf("failed to marshal request: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/setup/validate", bytes.NewReader(body))
	w := httptest.NewRecorder()

	SetupValidateHandler(configPath, setupmode.New(configPath, false))(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp SetupValidateResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if !resp.Valid {
		t.Fatalf("expected valid=true")
	}
	if !resp.CanActivate {
		t.Fatalf("expected canActivate=true")
	}
	if resp.Models != 1 || resp.Decisions != 1 {
		t.Fatalf("expected models=1 and decisions=1, got models=%d decisions=%d", resp.Models, resp.Decisions)
	}
	if resp.Signals != 2 {
		t.Fatalf("expected signals=2, got %d", resp.Signals)
	}
	var configMap map[string]interface{}
	if err := json.Unmarshal(resp.Config, &configMap); err != nil {
		t.Fatalf("failed to decode validated config: %v", err)
	}
	if _, hasSetup := configMap["setup"]; hasSetup {
		t.Fatalf("validated config should not contain setup marker")
	}
}

func TestSetupValidateHandlerUsesConfigDirectoryForRelativeKBAssets(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createBootstrapSetupConfig(t, tempDir)

	kbDir := filepath.Join(tempDir, "custom-kb")
	if err := os.MkdirAll(kbDir, 0o755); err != nil {
		t.Fatalf("failed to create custom kb directory: %v", err)
	}
	if err := os.WriteFile(filepath.Join(kbDir, "labels.json"), []byte(`{
  "labels": {
    "safe": {
      "description": "Safe content",
      "exemplars": ["hello world"]
    }
  }
}`), 0o644); err != nil {
		t.Fatalf("failed to write custom kb labels manifest: %v", err)
	}

	patch := createValidSetupPatch()
	patch["global"] = map[string]interface{}{
		"model_catalog": map[string]interface{}{
			"kbs": []map[string]interface{}{
				{
					"name": "custom_kb",
					"source": map[string]interface{}{
						"path":     "custom-kb/",
						"manifest": "labels.json",
					},
					"threshold": 0.55,
				},
			},
		},
	}

	body, err := json.Marshal(SetupConfigRequest{Config: mustJSONRaw(t, patch)})
	if err != nil {
		t.Fatalf("failed to marshal request: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/setup/validate", bytes.NewReader(body))
	w := httptest.NewRecorder()

	SetupValidateHandler(configPath, setupmode.New(configPath, false))(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}
}

func TestSetupImportRemoteHandler(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createBootstrapSetupConfig(t, tempDir)

	remoteConfigServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-yaml")
		_, _ = w.Write([]byte(`
version: v0.3
providers:
  defaults:
    default_model: remote-model
  models:
    - name: remote-model
      backend_refs:
        - name: primary
          endpoint: remote.example.com
          protocol: https
          weight: 100
routing:
  modelCards:
    - name: remote-model
      modality: text
  signals:
    domains:
      - name: remote-domain
        description: Remote domain signal
  decisions:
    - name: remote-route
      description: Remote route
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: domain
            name: remote-domain
      modelRefs:
        - model: remote-model
          use_reasoning: false
`))
	}))
	defer remoteConfigServer.Close()

	body, err := json.Marshal(SetupImportRemoteRequest{URL: remoteConfigServer.URL})
	if err != nil {
		t.Fatalf("failed to marshal request: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/setup/import-remote", bytes.NewReader(body))
	w := httptest.NewRecorder()

	SetupImportRemoteHandler(configPath, setupmode.New(configPath, false))(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp SetupImportRemoteResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if resp.SourceURL != remoteConfigServer.URL {
		t.Fatalf("expected sourceUrl=%q, got %q", remoteConfigServer.URL, resp.SourceURL)
	}
	if resp.Models != 1 || resp.Decisions != 1 || resp.Signals != 1 {
		t.Fatalf("expected counts 1/1/1, got models=%d decisions=%d signals=%d", resp.Models, resp.Decisions, resp.Signals)
	}
	if !resp.CanActivate {
		t.Fatalf("expected canActivate=true")
	}
	var importedConfig map[string]interface{}
	if err := json.Unmarshal(resp.Config, &importedConfig); err != nil {
		t.Fatalf("failed to decode imported config: %v", err)
	}
	if providers, ok := importedConfig["providers"].(map[string]interface{}); !ok {
		t.Fatalf("expected imported config providers map, got %#v", importedConfig["providers"])
	} else if defaults, ok := providers["defaults"].(map[string]interface{}); !ok || defaults["default_model"] != "remote-model" {
		t.Fatalf("expected imported config providers.defaults.default_model=remote-model, got %#v", importedConfig["providers"])
	}
	if routing, ok := importedConfig["routing"].(map[string]interface{}); !ok || routing["modelCards"] == nil {
		t.Fatalf("expected imported config routing.modelCards to be preserved, got %#v", importedConfig["routing"])
	}
}

func TestSetupImportRemoteHandlerUsesConfigDirectoryForRelativeKBAssets(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createBootstrapSetupConfig(t, tempDir)

	kbDir := filepath.Join(tempDir, "remote-kb")
	if err := os.MkdirAll(kbDir, 0o755); err != nil {
		t.Fatalf("failed to create remote kb directory: %v", err)
	}
	if err := os.WriteFile(filepath.Join(kbDir, "labels.json"), []byte(`{
  "labels": {
    "safe": {
      "description": "Safe content",
      "exemplars": ["hello world"]
    }
  }
}`), 0o644); err != nil {
		t.Fatalf("failed to write remote kb labels manifest: %v", err)
	}

	remoteConfigServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-yaml")
		_, _ = w.Write([]byte(`
version: v0.3
providers:
  defaults:
    default_model: remote-model
  models:
    - name: remote-model
      backend_refs:
        - name: primary
          endpoint: remote.example.com
          protocol: https
          weight: 100
routing:
  modelCards:
    - name: remote-model
      modality: text
  decisions:
    - name: remote-route
      description: Remote route
      priority: 100
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: remote-model
          use_reasoning: false
global:
  model_catalog:
    kbs:
      - name: remote_kb
        source:
          path: remote-kb/
          manifest: labels.json
        threshold: 0.55
`))
	}))
	defer remoteConfigServer.Close()

	body, err := json.Marshal(SetupImportRemoteRequest{URL: remoteConfigServer.URL})
	if err != nil {
		t.Fatalf("failed to marshal request: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/setup/import-remote", bytes.NewReader(body))
	w := httptest.NewRecorder()

	SetupImportRemoteHandler(configPath, setupmode.New(configPath, false))(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}
}

func TestSetupActivateHandler(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createBootstrapSetupConfig(t, tempDir)

	body, err := json.Marshal(SetupConfigRequest{Config: mustJSONRaw(t, createValidSetupPatch())})
	if err != nil {
		t.Fatalf("failed to marshal request: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/setup/activate", bytes.NewReader(body))
	w := httptest.NewRecorder()

	SetupActivateHandler(configPath, false, tempDir, setupmode.New(configPath, false))(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	configData, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("failed to read activated config: %v", err)
	}

	var configMap map[string]interface{}
	if err := yaml.Unmarshal(configData, &configMap); err != nil {
		t.Fatalf("failed to parse activated config: %v", err)
	}

	if _, hasSetup := configMap["setup"]; hasSetup {
		t.Fatalf("setup marker should be removed after activation")
	}

	globalConfig, ok := configMap["global"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected activated config to include explicit global defaults, got %#v", configMap["global"])
	}
	modelCatalog, ok := globalConfig["model_catalog"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected global.model_catalog in activated config, got %#v", globalConfig["model_catalog"])
	}
	embeddings, ok := modelCatalog["embeddings"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected global.model_catalog.embeddings in activated config, got %#v", modelCatalog["embeddings"])
	}
	semantic, ok := embeddings["semantic"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected global.model_catalog.embeddings.semantic in activated config, got %#v", embeddings["semantic"])
	}
	// Mirrors pkg/config/canonical_defaults global.model_catalog.embeddings.semantic.mmbert_model_path
	if semantic["mmbert_model_path"] != "models/mmbert-embed-32k-2d-matryoshka" {
		t.Fatalf("expected explicit mmbert default path, got %#v", semantic["mmbert_model_path"])
	}

	if info, err := os.Stat(filepath.Join(tempDir, ".vllm-sr")); err != nil || !info.IsDir() {
		t.Fatalf(".vllm-sr output directory should exist after activation: %v", err)
	}
}

func TestSetupActivateHandlerStartsCreatedSplitRuntimeContainers(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createBootstrapSetupConfig(t, tempDir)
	fakeDocker := writeFakeLifecycleDockerCLI(t)

	t.Setenv("PATH", filepath.Dir(fakeDocker.path)+":"+os.Getenv("PATH"))
	t.Setenv(routerContainerNameEnv, "lane-a-vllm-sr-router-container")
	t.Setenv(envoyContainerNameEnv, "lane-a-vllm-sr-envoy-container")
	t.Setenv(dashboardContainerNameEnv, "lane-a-vllm-sr-dashboard-container")
	t.Setenv("TEST_DOCKER_LOG_FILE", fakeDocker.logPath)
	t.Setenv("TEST_ROUTER_CONTAINER", "lane-a-vllm-sr-router-container")
	t.Setenv("TEST_ROUTER_STATUS_FILE", fakeDocker.routerStatusPath)
	t.Setenv("TEST_ENVOY_CONTAINER", "lane-a-vllm-sr-envoy-container")
	t.Setenv("TEST_ENVOY_STATUS_FILE", fakeDocker.envoyStatusPath)

	if writeErr := os.WriteFile(fakeDocker.routerStatusPath, []byte("created\n"), 0o644); writeErr != nil {
		t.Fatalf("failed to seed router status: %v", writeErr)
	}
	if writeErr := os.WriteFile(fakeDocker.envoyStatusPath, []byte("created\n"), 0o644); writeErr != nil {
		t.Fatalf("failed to seed envoy status: %v", writeErr)
	}

	body, err := json.Marshal(SetupConfigRequest{Config: mustJSONRaw(t, createValidSetupPatch())})
	if err != nil {
		t.Fatalf("failed to marshal request: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/setup/activate", bytes.NewReader(body))
	w := httptest.NewRecorder()

	SetupActivateHandler(configPath, false, tempDir, setupmode.New(configPath, false))(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	logData, err := os.ReadFile(fakeDocker.logPath)
	if err != nil {
		t.Fatalf("failed to read docker log: %v", err)
	}
	logText := string(logData)
	if !strings.Contains(logText, "start lane-a-vllm-sr-router-container") {
		t.Fatalf("expected router start, got %q", logText)
	}
	if !strings.Contains(logText, "start lane-a-vllm-sr-envoy-container") {
		t.Fatalf("expected envoy start, got %q", logText)
	}
	if strings.Contains(logText, "supervisorctl") {
		t.Fatalf("split runtime should not use supervisorctl, got %q", logText)
	}
}

func TestSetupActivateHandlerRefreshesSplitEnvoyConfigBeforeStartingCreatedContainers(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createBootstrapSetupConfig(t, tempDir)
	fakeDocker := writeFakeLifecycleDockerCLI(t)

	t.Setenv("PATH", filepath.Dir(fakeDocker.path)+":"+os.Getenv("PATH"))
	t.Setenv(routerContainerNameEnv, "lane-a-vllm-sr-router-container")
	t.Setenv(envoyContainerNameEnv, "lane-a-vllm-sr-envoy-container")
	t.Setenv(dashboardContainerNameEnv, "lane-a-vllm-sr-dashboard-container")
	t.Setenv("TEST_DOCKER_LOG_FILE", fakeDocker.logPath)
	t.Setenv("TEST_ROUTER_CONTAINER", "lane-a-vllm-sr-router-container")
	t.Setenv("TEST_ROUTER_STATUS_FILE", fakeDocker.routerStatusPath)
	t.Setenv("TEST_ENVOY_CONTAINER", "lane-a-vllm-sr-envoy-container")
	t.Setenv("TEST_ENVOY_STATUS_FILE", fakeDocker.envoyStatusPath)

	runtimeDir := filepath.Join(tempDir, ".vllm-sr")
	if err := os.MkdirAll(runtimeDir, 0o755); err != nil {
		t.Fatalf("failed to create runtime dir: %v", err)
	}

	runtimeConfigPath := filepath.Join(runtimeDir, "runtime-config.yaml")
	envoyConfigPath := filepath.Join(runtimeDir, "envoy.yaml")
	if err := os.WriteFile(envoyConfigPath, []byte("# stale bootstrap config\nbootstrap_only: true\n"), 0o644); err != nil {
		t.Fatalf("failed to seed stale envoy config: %v", err)
	}

	t.Setenv("VLLM_SR_RUNTIME_CONFIG_PATH", runtimeConfigPath)
	t.Setenv("VLLM_SR_ENVOY_CONFIG_PATH", envoyConfigPath)
	pythonBinary := "python3"
	if _, err := exec.LookPath(pythonBinary); err != nil {
		pythonBinary = "python"
	}
	t.Setenv("VLLM_SR_PYTHON_BIN", pythonBinary)
	repoRoot, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatalf("resolve repo root: %v", err)
	}
	t.Setenv("VLLM_SR_CLI_PATH", filepath.Join(repoRoot, "src", "vllm-sr"))

	if writeErr := os.WriteFile(fakeDocker.routerStatusPath, []byte("created\n"), 0o644); writeErr != nil {
		t.Fatalf("failed to seed router status: %v", writeErr)
	}
	if writeErr := os.WriteFile(fakeDocker.envoyStatusPath, []byte("created\n"), 0o644); writeErr != nil {
		t.Fatalf("failed to seed envoy status: %v", writeErr)
	}

	body, err := json.Marshal(SetupConfigRequest{Config: mustJSONRaw(t, createValidSetupPatch())})
	if err != nil {
		t.Fatalf("failed to marshal request: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/setup/activate", bytes.NewReader(body))
	w := httptest.NewRecorder()

	SetupActivateHandler(configPath, false, tempDir, setupmode.New(configPath, false))(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	envoyConfigData, err := os.ReadFile(envoyConfigPath)
	if err != nil {
		t.Fatalf("failed to read envoy config: %v", err)
	}
	envoyConfigText := string(envoyConfigData)
	if strings.Contains(envoyConfigText, "bootstrap_only") {
		t.Fatalf("expected setup activation to replace stale envoy config, got:\n%s", envoyConfigText)
	}
	if !strings.Contains(envoyConfigText, "host.docker.internal") {
		t.Fatalf("expected refreshed envoy config to include activated backend endpoint, got:\n%s", envoyConfigText)
	}
	if !strings.Contains(envoyConfigText, "test_model_cluster") {
		t.Fatalf("expected refreshed envoy config to include activated model cluster, got:\n%s", envoyConfigText)
	}
}

// --- Resolved setup state on /api/setup/state (#2795) ----------------------

// The happy path must not carry a reason. Asserted on the raw body, because a
// decoded struct cannot tell an empty string from an omitted key.
func TestSetupStateHandlerOmitsReasonOnCleanResolution(t *testing.T) {
	configPath := createBootstrapSetupConfig(t, t.TempDir())

	resp, raw := getSetupState(t, configPath, setupmode.New(configPath, true))

	if !resp.SetupMode {
		t.Fatalf("setupMode = false, want true; body=%s", raw)
	}
	if resp.Reason != "" {
		t.Fatalf("reason = %q, want empty on a clean resolution", resp.Reason)
	}
	if strings.Contains(raw, "reason") {
		t.Fatalf("raw body contains a reason key, want it omitted entirely: %s", raw)
	}
}

// A stale DASHBOARD_SETUP_MODE against an activated config is the invisible
// case from the issue. The state stays false and the reason makes it visible.
func TestSetupStateHandlerExplainsStaleLegacyFlag(t *testing.T) {
	configPath := createActivatedSetupConfig(t, t.TempDir())

	resp, raw := getSetupState(t, configPath, setupmode.New(configPath, true))

	if resp.SetupMode {
		t.Fatalf("setupMode = true from the legacy flag alone, want false; body=%s", raw)
	}
	if resp.Reason == "" {
		t.Fatalf("reason is empty for a conflicting legacy flag; body=%s", raw)
	}
	if !strings.Contains(resp.Reason, "DASHBOARD_SETUP_MODE") {
		t.Fatalf("reason %q does not name the stale input", resp.Reason)
	}
	if !strings.Contains(raw, `"reason"`) {
		t.Fatalf("raw body is missing the reason key: %s", raw)
	}
}

// An unreadable config used to produce a 500, which the frontend swallowed into
// "not in setup mode" with no explanation. Answer 200 with the reason instead.
func TestSetupStateHandlerAnswers200WithReasonWhenConfigUnreadable(t *testing.T) {
	missingPath := filepath.Join(t.TempDir(), "does-not-exist.yaml")

	w := httptest.NewRecorder()
	SetupStateHandler(missingPath, setupmode.New(missingPath, false))(
		w, httptest.NewRequest(http.MethodGet, "/api/setup/state", nil))

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 with a diagnostic reason; body=%s", w.Code, w.Body.String())
	}

	resp, raw := decodeSetupState(t, w)
	if resp.SetupMode {
		t.Fatalf("setupMode = true for an unreadable config, want false (fail closed); body=%s", raw)
	}
	if !strings.Contains(resp.Reason, "unreadable") {
		t.Fatalf("reason = %q, want it to mention the config being unreadable", resp.Reason)
	}
}

// The write endpoints share the gate, so on an activated config they must all
// refuse.
func TestSetupWriteEndpointsGateOnResolvedState(t *testing.T) {
	configPath := createActivatedSetupConfig(t, t.TempDir())
	resolver := setupmode.New(configPath, true)

	body, err := json.Marshal(SetupConfigRequest{Config: mustJSONRaw(t, createValidSetupPatch())})
	if err != nil {
		t.Fatalf("failed to marshal request: %v", err)
	}

	t.Run("validate", func(t *testing.T) {
		w := httptest.NewRecorder()
		SetupValidateHandler(configPath, resolver)(
			w, httptest.NewRequest(http.MethodPost, "/api/setup/validate", bytes.NewReader(body)))

		if w.Code != http.StatusBadRequest {
			t.Fatalf("status = %d, want 400; body=%s", w.Code, w.Body.String())
		}
		if !strings.Contains(w.Body.String(), "setup mode is not active") {
			t.Fatalf("body = %q, want it to say setup mode is not active", w.Body.String())
		}
	})

	t.Run("import-remote", func(t *testing.T) {
		w := httptest.NewRecorder()
		SetupImportRemoteHandler(configPath, resolver)(
			w, httptest.NewRequest(http.MethodPost, "/api/setup/import-remote", strings.NewReader(`{"url":"https://example.com/c.yaml"}`)))

		if w.Code != http.StatusBadRequest {
			t.Fatalf("status = %d, want 400; body=%s", w.Code, w.Body.String())
		}
		if !strings.Contains(w.Body.String(), "setup mode is not active") {
			t.Fatalf("body = %q, want it to say setup mode is not active", w.Body.String())
		}
	})

	t.Run("activate", func(t *testing.T) {
		w := httptest.NewRecorder()
		SetupActivateHandler(configPath, false, filepath.Dir(configPath), resolver)(
			w, httptest.NewRequest(http.MethodPost, "/api/setup/activate", bytes.NewReader(body)))

		if w.Code != http.StatusBadRequest {
			t.Fatalf("status = %d, want 400; body=%s", w.Code, w.Body.String())
		}
		if !strings.Contains(w.Body.String(), "setup mode is not active") {
			t.Fatalf("body = %q, want it to say setup mode is not active", w.Body.String())
		}
	})
}

// Proves the Invalidate call works.
//
// Activation writes the config then answers the request, and on coarse mtime
// granularity the identity check cannot see the change on its own. The same
// resolver is reused and nothing restarts, so a stale cached resolution would
// show up as /api/setup/state still reporting true after activation.
func TestSetupActivateHandlerFlipsSetupStateWithinOneRequest(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createBootstrapSetupConfig(t, tempDir)
	// The legacy flag stays true across activation, as it does in a real
	// deployment: the CLI sets it at launch and nothing clears it.
	resolver := setupmode.New(configPath, true)

	before, rawBefore := getSetupState(t, configPath, resolver)
	if !before.SetupMode {
		t.Fatalf("setupMode = false before activation, want true; body=%s", rawBefore)
	}
	if before.Reason != "" {
		t.Fatalf("reason = %q before activation, want empty (flag and config agree)", before.Reason)
	}

	body, err := json.Marshal(SetupConfigRequest{Config: mustJSONRaw(t, createValidSetupPatch())})
	if err != nil {
		t.Fatalf("failed to marshal request: %v", err)
	}

	w := httptest.NewRecorder()
	SetupActivateHandler(configPath, false, tempDir, resolver)(
		w, httptest.NewRequest(http.MethodPost, "/api/setup/activate", bytes.NewReader(body)))
	if w.Code != http.StatusOK {
		t.Fatalf("activate status = %d, want 200; body=%s", w.Code, w.Body.String())
	}

	after, rawAfter := getSetupState(t, configPath, resolver)
	if after.SetupMode {
		t.Fatalf("setupMode = true after activation; the cached resolution was not invalidated; body=%s", rawAfter)
	}
	// The flag now disagrees with the config, so the response also explains
	// why the environment value lost.
	if !strings.Contains(after.Reason, "DASHBOARD_SETUP_MODE") {
		t.Fatalf("reason = %q after activation, want it to name the now-stale legacy flag", after.Reason)
	}

	// The write endpoints closed in the same moment, through the same resolver.
	vw := httptest.NewRecorder()
	SetupValidateHandler(configPath, resolver)(
		vw, httptest.NewRequest(http.MethodPost, "/api/setup/validate", bytes.NewReader(body)))
	if vw.Code != http.StatusBadRequest {
		t.Fatalf("validate status = %d after activation, want 400; body=%s", vw.Code, vw.Body.String())
	}
}

// Covers the branch where the two decoders disagree: the resolver answers
// cleanly from the setup block while this handler cannot decode the full schema.
//
// The response must carry the resolved value, not a hardcoded false. Reporting
// false while the bootstrap gate reads true is the invisible-open-door split
// this change removes. The resolution is clean, so its Reason is empty and the
// handler supplies its own.
func TestSetupStateHandlerReportsResolvedStateWhenCanonicalDecodeFails(t *testing.T) {
	t.Run("setup mode active: state must agree with the bootstrap gate", func(t *testing.T) {
		configPath := createCanonicallyInvalidSetupConfig(t, t.TempDir(), true)
		// legacyFlag=true matches the config, so the resolution is clean.
		resolver := setupmode.New(configPath, true)

		if !resolver.Active() {
			t.Fatalf("precondition failed: resolver must read the setup block of a canonically invalid config")
		}
		if reason := resolver.Resolve().Reason; reason != "" {
			t.Fatalf("precondition failed: resolution should be clean, got reason %q", reason)
		}

		resp, raw := getSetupState(t, configPath, resolver)

		if !resp.SetupMode {
			t.Fatalf("setupMode = false while the bootstrap gate is open; the surfaces disagree. body=%s", raw)
		}
		if resp.Reason != unreadableConfigReason {
			t.Fatalf("reason = %q, want the handler's own explanation %q", resp.Reason, unreadableConfigReason)
		}
		if !strings.Contains(raw, `"reason"`) {
			t.Fatalf("raw body is missing the reason key: %s", raw)
		}
		// The rest of the payload is unavailable, so it must be empty.
		if resp.ListenerPort != 0 || resp.Models != 0 || resp.Decisions != 0 || resp.CanActivate {
			t.Fatalf("expected an empty payload alongside the reason, got %+v", resp)
		}
	})

	t.Run("setup mode inactive", func(t *testing.T) {
		configPath := createCanonicallyInvalidSetupConfig(t, t.TempDir(), false)
		resolver := setupmode.New(configPath, false)

		resp, raw := getSetupState(t, configPath, resolver)

		if resp.SetupMode {
			t.Fatalf("setupMode = true for a config with no setup block; body=%s", raw)
		}
		if resp.Reason != unreadableConfigReason {
			t.Fatalf("reason = %q, want %q", resp.Reason, unreadableConfigReason)
		}
	})

	// The reason is served unauthenticated, so it must disclose neither the
	// config location nor its contents.
	t.Run("reason discloses neither path nor contents", func(t *testing.T) {
		dir := t.TempDir()
		configPath := createCanonicallyInvalidSetupConfig(t, dir, true)

		resp, _ := getSetupState(t, configPath, setupmode.New(configPath, true))

		for _, secret := range []string{configPath, dir, "not-a-list"} {
			if strings.Contains(resp.Reason, secret) {
				t.Fatalf("reason %q discloses %q", resp.Reason, secret)
			}
		}
	})
}

// The write endpoints gate first, then read. When the gate passes but the read
// fails they must report the read failure, not an inactive gate, which would
// send an operator looking in the wrong place.
func TestSetupWriteEndpointsReportUnreadableConfigWhileSetupModeIsActive(t *testing.T) {
	configPath := createCanonicallyInvalidSetupConfig(t, t.TempDir(), true)
	resolver := setupmode.New(configPath, true)

	body, err := json.Marshal(SetupConfigRequest{Config: mustJSONRaw(t, createValidSetupPatch())})
	if err != nil {
		t.Fatalf("failed to marshal request: %v", err)
	}

	w := httptest.NewRecorder()
	SetupValidateHandler(configPath, resolver)(
		w, httptest.NewRequest(http.MethodPost, "/api/setup/validate", bytes.NewReader(body)))

	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body=%s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "failed to read existing config") {
		t.Fatalf("body = %q, want it to report the read failure rather than an inactive gate", w.Body.String())
	}
}
