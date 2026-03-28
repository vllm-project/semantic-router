package handlers

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

func TestGlobalConfigYAMLHandler_ReturnsEffectiveGlobalConfig(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	configWithGlobal := `
version: v0.3
listeners:
  - name: public
    address: 0.0.0.0
    port: 8801
providers:
  defaults:
    default_model: test-model
    reasoning_families:
      qwen3:
        type: reasoning_effort
        parameter: reasoning_effort
  models:
    - name: test-model
      provider_model_id: test-model
      backend_refs:
        - name: endpoint1
          endpoint: 127.0.0.1:8000
          protocol: http
routing:
  modelCards:
    - name: test-model
  signals:
    domains:
      - name: business
        description: Business
  decisions:
    - name: business-route
      priority: 1
      rules:
        operator: AND
        conditions:
          - type: domain
            name: business
      modelRefs:
        - model: test-model
global:
  router:
    strategy: priority
  services:
    response_api:
      enabled: true
`
	if err := os.WriteFile(configPath, []byte(configWithGlobal), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/api/router/config/global/raw", nil)
	w := httptest.NewRecorder()

	GlobalConfigYAMLHandler(configPath)(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	body := w.Body.String()
	if strings.Contains(body, "\nversion:") || strings.Contains(body, "\nproviders:") || strings.Contains(body, "\nrouting:") {
		t.Fatalf("raw global response should only contain the global block, got:\n%s", body)
	}
	if !strings.Contains(body, "router:") || !strings.Contains(body, "response_api:") {
		t.Fatalf("raw global response missing expected sections, got:\n%s", body)
	}
	if !strings.Contains(body, "stores:") || !strings.Contains(body, "semantic_cache:") {
		t.Fatalf("raw global response should include effective router defaults, got:\n%s", body)
	}
}

func TestGlobalConfigYAMLHandler_ReturnsDefaultsWhenGlobalMissing(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	req := httptest.NewRequest(http.MethodGet, "/api/router/config/global/raw", nil)
	w := httptest.NewRecorder()

	GlobalConfigYAMLHandler(configPath)(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	body := w.Body.String()
	for _, fragment := range []string{
		"router:",
		"services:",
		"stores:",
		"model_catalog:",
		"response_api:",
		"semantic_cache:",
	} {
		if !strings.Contains(body, fragment) {
			t.Fatalf("effective global response missing %q:\n%s", fragment, body)
		}
	}
}

func TestUpdateGlobalConfigYAMLHandler_ReplacesGlobalOverride(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	initial := `
version: v0.3
listeners:
  - name: public
    address: 0.0.0.0
    port: 8801
providers:
  defaults:
    default_model: test-model
    reasoning_families:
      qwen3:
        type: reasoning_effort
        parameter: reasoning_effort
  models:
    - name: test-model
      provider_model_id: test-model
      backend_refs:
        - name: endpoint1
          endpoint: 127.0.0.1:8000
          protocol: http
routing:
  modelCards:
    - name: test-model
  signals:
    domains:
      - name: business
        description: Business
  decisions:
    - name: business-route
      priority: 1
      rules:
        operator: AND
        conditions:
          - type: domain
            name: business
      modelRefs:
        - model: test-model
global:
  router:
    strategy: priority
  services:
    response_api:
      enabled: true
`
	if err := os.WriteFile(configPath, []byte(initial), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	rawUpdate := `
stores:
  semantic_cache:
    enabled: true
    backend_type: memory
router:
  clear_route_cache: true
`
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/global/raw/update", bytes.NewBufferString(rawUpdate))
	req.Header.Set("Content-Type", "text/yaml")
	w := httptest.NewRecorder()

	UpdateGlobalConfigYAMLHandler(configPath, false, tempDir)(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	updatedData, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read updated config: %v", err)
	}
	updated := string(updatedData)
	if !strings.Contains(updated, "semantic_cache:") || !strings.Contains(updated, "clear_route_cache: true") {
		t.Fatalf("raw global save did not persist the new block, got:\n%s", updated)
	}
}

func TestUpdateGlobalConfigYAMLHandler_EmptyBodyClearsGlobalOverride(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	configWithGlobal := `
version: v0.3
listeners:
  - name: public
    address: 0.0.0.0
    port: 8801
providers:
  defaults:
    default_model: test-model
    reasoning_families:
      qwen3:
        type: reasoning_effort
        parameter: reasoning_effort
  models:
    - name: test-model
      provider_model_id: test-model
      backend_refs:
        - name: endpoint1
          endpoint: 127.0.0.1:8000
          protocol: http
routing:
  modelCards:
    - name: test-model
  signals:
    domains:
      - name: business
        description: Business
  decisions:
    - name: business-route
      priority: 1
      rules:
        operator: AND
        conditions:
          - type: domain
            name: business
      modelRefs:
        - model: test-model
global:
  router:
    strategy: priority
`
	if err := os.WriteFile(configPath, []byte(configWithGlobal), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/router/config/global/raw/update", bytes.NewBufferString(""))
	req.Header.Set("Content-Type", "text/yaml")
	w := httptest.NewRecorder()

	UpdateGlobalConfigYAMLHandler(configPath, false, tempDir)(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	updatedData, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read updated config: %v", err)
	}
	if strings.Contains(string(updatedData), "\nglobal:") {
		t.Fatalf("expected empty raw save to remove global override, got:\n%s", string(updatedData))
	}
}
