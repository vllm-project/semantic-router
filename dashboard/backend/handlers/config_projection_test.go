package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/configprojection"
)

func openTestConfigProjectionStore(t *testing.T) *configprojection.Store {
	t.Helper()
	store, err := configprojection.Open(filepath.Join(t.TempDir(), "projection.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = store.Close()
	})
	return store
}

func TestActiveConfigProjectionHandlerReturnsActiveDeployment(t *testing.T) {
	store := openTestConfigProjectionStore(t)
	SetConfigProjectionStore(store)

	if err := store.RefreshFromCanonical(configprojection.RefreshInput{
		Version:   "20260101-120000",
		Source:    configprojection.SourceManual,
		YAMLBytes: []byte(testCanonicalYAMLForProjection()),
	}); err != nil {
		t.Fatalf("RefreshFromCanonical: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/api/router/config/active-projection", nil)
	w := httptest.NewRecorder()
	ActiveConfigProjectionHandler()(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}

	var resp configprojection.ActiveProjectionResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Status != configprojection.StatusOK {
		t.Fatalf("unexpected status: %+v", resp)
	}
	if resp.Deployment == nil || resp.Deployment.Version != "20260101-120000" {
		t.Fatalf("unexpected deployment: %+v", resp.Deployment)
	}
}

func TestConfigDeploymentDetailHandlerNotFound(t *testing.T) {
	store := openTestConfigProjectionStore(t)
	SetConfigProjectionStore(store)

	req := httptest.NewRequest(http.MethodGet, "/api/router/config/deployments/missing-version", nil)
	w := httptest.NewRecorder()
	ConfigDeploymentDetailHandler()(w, req)

	if w.Code != http.StatusNotFound {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
}

func testCanonicalYAMLForProjection() string {
	return `
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
      reasoning_family: qwen3
      provider_model_id: test-model
      backend_refs:
        - name: endpoint1
          endpoint: 127.0.0.1:8000
          protocol: http
          weight: 1
routing:
  modelCards:
    - name: test-model
  signals:
    domains:
      - name: business
        description: Business and management related queries
  decisions:
    - name: default-business
      priority: 1
      rules:
        operator: OR
        conditions:
          - type: domain
            name: business
      modelRefs:
        - model: test-model
`
}
