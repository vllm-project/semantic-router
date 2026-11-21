package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestConfigHandler(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	tests := []struct {
		name           string
		method         string
		expectedStatus int
	}{
		{
			name:           "GET request should succeed",
			method:         http.MethodGet,
			expectedStatus: http.StatusOK,
		},
		{
			name:           "POST request should fail",
			method:         http.MethodPost,
			expectedStatus: http.StatusMethodNotAllowed,
		},
		{
			name:           "PUT request should fail",
			method:         http.MethodPut,
			expectedStatus: http.StatusMethodNotAllowed,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(tt.method, "/api/router/config/all", nil)
			w := httptest.NewRecorder()

			handler := ConfigHandler(configPath, false)
			handler(w, req)

			if w.Code != tt.expectedStatus {
				t.Errorf("Expected status %d, got %d", tt.expectedStatus, w.Code)
			}

			if tt.expectedStatus == http.StatusOK {
				var result interface{}
				if err := json.NewDecoder(w.Body).Decode(&result); err != nil {
					t.Errorf("Response is not valid JSON: %v", err)
				}
			}
		})
	}
}

// A configuration the Router loads must read back through the Dashboard. A
// global router_replay cost_baseline_model names a configured model, which the
// Router resolves on the whole config; the Dashboard, which validates the
// global block alone, must not answer 500 because that block has no models in
// scope. Two surfaces, one verdict.
func TestConfigHandlerReadsAGlobalCostBaselineModel(t *testing.T) {
	config := `version: v0.5
listeners:
  - name: public
    address: 0.0.0.0
    port: 8801
providers:
  defaults:
    model: test-model
  models:
    - name: test-model
      reasoning:
        family: qwen3
      provider_model_id: test-model
      pricing: {currency: USD, prompt_per_1m: 1, completion_per_1m: 2}
      backend_refs:
        - name: endpoint1
          provider: vllm
          endpoint: 127.0.0.1:8000
          protocol: http
          weight: 1
recipes:
  - name: default
    decisions:
      - name: route
        priority: 1
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: test-model
global:
  plugins:
    - type: router_replay
      configuration:
        cost_baseline_model: test-model
`
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(configPath, []byte(config), 0o600); err != nil {
		t.Fatal(err)
	}
	// The Router loads it: the baseline resolves against the whole config.
	if _, err := routerconfig.ParseYAMLBytes([]byte(config)); err != nil {
		t.Fatalf("the Router rejects a config it should load: %v", err)
	}
	w := httptest.NewRecorder()
	ConfigHandler(configPath, false)(w, httptest.NewRequest(http.MethodGet, "/api/router/config/all", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("config read status = %d, want 200: %s", w.Code, w.Body.String())
	}
}

// A model that sets no pricing is priced at its provider's catalog list price,
// and the models page reads that price here. A priced model and a self-hosted
// model are not listed.
func TestCatalogPricesHandlerListsCatalogPricedModels(t *testing.T) {
	config := `version: v0.5
listeners:
  - name: public
    address: 0.0.0.0
    port: 8801
providers:
  defaults:
    model: own
  models:
    - name: listed
      provider_model_id: claude-sonnet-5-5
      api_format: anthropic
      backend_refs:
        - {name: claude, provider: anthropic}
    - name: own
      provider_model_id: claude-sonnet-5-5
      api_format: anthropic
      pricing: {currency: USD, prompt_per_1m: 1, completion_per_1m: 2}
      backend_refs:
        - {name: claude, provider: anthropic}
    - name: local
      provider_model_id: claude-sonnet-5-5
      backend_refs:
        - {name: local, provider: vllm, endpoint: 127.0.0.1:8000}
recipes:
  - name: default
`
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(configPath, []byte(config), 0o600); err != nil {
		t.Fatal(err)
	}
	w := httptest.NewRecorder()
	CatalogPricesHandler(configPath)(w, httptest.NewRequest(http.MethodGet, "/api/router/config/catalog-prices", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", w.Code, w.Body.String())
	}
	var body struct {
		CatalogListPrices map[string]map[string]any `json:"catalog_list_prices"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if len(body.CatalogListPrices) != 1 || body.CatalogListPrices["listed"]["prompt_per_1m"] != 2.0 || body.CatalogListPrices["listed"]["currency"] != "USD" {
		t.Fatalf("catalog prices = %v, want only listed at the anthropic list price", body.CatalogListPrices)
	}
}
