package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
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

	SetupStateHandler(configPath)(w, req)

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

	SetupValidateHandler(configPath)(w, req)

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

	SetupValidateHandler(configPath)(w, req)

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

	SetupImportRemoteHandler(configPath)(w, req)

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

	SetupImportRemoteHandler(configPath)(w, req)

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

	SetupActivateHandler(configPath, false, tempDir)(w, req)

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
	if semantic["mmbert_model_path"] != "models/mom-embedding-ultra" {
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

	if err := os.WriteFile(fakeDocker.routerStatusPath, []byte("created\n"), 0o644); err != nil {
		t.Fatalf("failed to seed router status: %v", err)
	}
	if err := os.WriteFile(fakeDocker.envoyStatusPath, []byte("created\n"), 0o644); err != nil {
		t.Fatalf("failed to seed envoy status: %v", err)
	}

	body, err := json.Marshal(SetupConfigRequest{Config: mustJSONRaw(t, createValidSetupPatch())})
	if err != nil {
		t.Fatalf("failed to marshal request: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/setup/activate", bytes.NewReader(body))
	w := httptest.NewRecorder()

	SetupActivateHandler(configPath, false, tempDir)(w, req)

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
