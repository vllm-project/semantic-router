package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"gopkg.in/yaml.v3"
)

func createValidPythonCLIConfig(t *testing.T, dir string) string {
	configPath := filepath.Join(dir, "config.yaml")
	validConfig := `
version: v0.1
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
signals:
  keywords:
    - name: billing_keywords
      operator: contains
      keywords:
        - invoice
decisions:
  - name: billing-route
    description: Route billing questions
    priority: 90
    rules:
      operator: AND
      conditions:
        - type: keyword
          name: billing_keywords
    modelRefs:
      - model: qwen3-4b
providers:
  models:
    - name: qwen3-4b
      endpoints:
        - name: primary
          weight: 100
          endpoint: router.internal:8000
  default_model: qwen3-4b
`
	if err := os.WriteFile(configPath, []byte(validConfig), 0o644); err != nil {
		t.Fatalf("Failed to create Python CLI test config file: %v", err)
	}
	return configPath
}

func copySharedFirstSliceRuntimeConfig(t *testing.T, dir string) string {
	t.Helper()

	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("failed to resolve test file path")
	}

	fixturePath := filepath.Clean(
		filepath.Join(
			filepath.Dir(thisFile),
			"..",
			"..",
			"..",
			"config",
			"testing",
			"td001-first-slice-runtime.yaml",
		),
	)
	fixtureData, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("Failed to read shared runtime fixture: %v", err)
	}

	configPath := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(configPath, fixtureData, 0o644); err != nil {
		t.Fatalf("Failed to copy shared runtime fixture: %v", err)
	}
	return configPath
}

func configureDashboardPythonCLITestEnv(t *testing.T) {
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("failed to resolve test file path")
	}

	cliRoot := filepath.Clean(filepath.Join(filepath.Dir(thisFile), "..", "..", "..", "src", "vllm-sr"))
	t.Setenv("VLLM_SR_CLI_PATH", cliRoot)

	previousDockerStatus := runtimeDockerContainerStatus
	previousInContainer := runtimeIsRunningInContainer
	runtimeDockerContainerStatus = func(string) string { return "not found" }
	runtimeIsRunningInContainer = func() bool { return false }
	t.Cleanup(func() {
		runtimeDockerContainerStatus = previousDockerStatus
		runtimeIsRunningInContainer = previousInContainer
	})
}

func loadYAMLMapForTest(t *testing.T, configPath string) map[string]interface{} {
	t.Helper()

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("Failed to read test config file: %v", err)
	}

	var config map[string]interface{}
	if err := yaml.Unmarshal(data, &config); err != nil {
		t.Fatalf("Failed to parse test config YAML: %v", err)
	}
	return config
}

func TestConfigHandler_PythonCLILoadsCanonicalShape(t *testing.T) {
	tempDir := t.TempDir()
	configPath := copySharedFirstSliceRuntimeConfig(t, tempDir)
	configureDashboardPythonCLITestEnv(t)

	req := httptest.NewRequest(http.MethodGet, "/api/router/config/all", nil)
	w := httptest.NewRecorder()

	handler := ConfigHandler(configPath)
	handler(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected status 200, got %d. Response: %s", w.Code, w.Body.String())
	}

	var result map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&result); err != nil {
		t.Fatalf("Failed to decode config response: %v", err)
	}

	if _, ok := result["providers"]; !ok {
		t.Fatalf("Expected canonical providers block in response, got: %#v", result)
	}
	if _, ok := result["model_config"]; ok {
		t.Fatalf("Did not expect legacy model_config root key in canonical response: %#v", result)
	}
}

func TestUpdateConfigHandler_PythonCLIValidation(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidPythonCLIConfig(t, tempDir)
	configureDashboardPythonCLITestEnv(t)

	t.Run("accepts partial canonical update", func(t *testing.T) {
		updateBody := map[string]interface{}{
			"providers": map[string]interface{}{
				"default_model": "qwen3-4b",
			},
		}

		bodyBytes, _ := json.Marshal(updateBody)
		req := httptest.NewRequest(http.MethodPost, "/api/router/config/update", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		handler := UpdateConfigHandler(configPath, false, tempDir)
		handler(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("Expected status 200, got %d. Response: %s", w.Code, w.Body.String())
		}
	})

	t.Run("rejects unknown canonical top-level block", func(t *testing.T) {
		createValidPythonCLIConfig(t, tempDir)

		updateBody := map[string]interface{}{
			"unknown_block": map[string]interface{}{
				"enabled": true,
			},
		}

		bodyBytes, _ := json.Marshal(updateBody)
		req := httptest.NewRequest(http.MethodPost, "/api/router/config/update", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		handler := UpdateConfigHandler(configPath, false, tempDir)
		handler(w, req)

		if w.Code != http.StatusBadRequest {
			t.Fatalf("Expected status 400, got %d. Response: %s", w.Code, w.Body.String())
		}
		if !contains(w.Body.String(), "Config validation failed") {
			t.Fatalf("Expected validation failure, got: %s", w.Body.String())
		}
	})
}

func TestUpdateConfigHandler_FullCanonicalWriteReplacesLegacyShape(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)
	configureDashboardPythonCLITestEnv(t)

	canonicalSourcePath := createValidPythonCLIConfig(t, t.TempDir())
	updateBody := loadYAMLMapForTest(t, canonicalSourcePath)

	bodyBytes, _ := json.Marshal(updateBody)
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/update", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler := UpdateConfigHandler(configPath, false, tempDir)
	handler(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected status 200, got %d. Response: %s", w.Code, w.Body.String())
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("Failed to read updated config: %v", err)
	}
	configYAML := string(data)

	if !contains(configYAML, "providers:") {
		t.Fatalf("Expected canonical providers block after write, got: %s", configYAML)
	}
	if contains(configYAML, "model_config:") || contains(configYAML, "vllm_endpoints:") {
		t.Fatalf("Did not expect legacy runtime root keys after canonical write, got: %s", configYAML)
	}
}
