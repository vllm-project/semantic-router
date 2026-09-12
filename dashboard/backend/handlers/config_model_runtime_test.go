package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestUpdateConfigHandlerPreservesModelRuntimeDeclarations(t *testing.T) {
	configPath := createValidTestConfig(t, t.TempDir())
	body := canonicalConfigBody("127.0.0.1:8000")
	body["global"] = map[string]interface{}{"model_catalog": map[string]interface{}{
		"deployments": map[string]interface{}{"local-pii": map[string]interface{}{
			"artifact": "models/checkpoint", "revision": "pinned", "provider": "candle", "device": "cpu", "precision": "fp32",
			"input": map[string]interface{}{"max_tokens": 512, "overflow": "reject"},
		}},
		"admission": map[string]interface{}{"local-pii": map[string]interface{}{"max_concurrency": 2, "max_queue": 3, "queue_timeout_ms": 500, "on_overflow": "shed"}},
	}}
	body["recipes"] = []map[string]interface{}{{"name": "private", "routing": map[string]interface{}{
		"model_bindings": map[string]interface{}{"pii_classifier": map[string]interface{}{
			"deployment": "local-pii", "contract": "token_spans.v1", "adapter": "mmbert32k", "head": "pii", "mapping_path": "mappings/pii.json",
		}},
	}}}
	payload, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/update", bytes.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	UpdateConfigHandler(configPath, false, "")(response, req)
	if response.Code != http.StatusOK {
		t.Fatalf("save failed: %d %s", response.Code, response.Body.String())
	}
	saved, err := readCanonicalConfigFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	deployment := saved.Global.ModelCatalog.Deployments["local-pii"]
	if deployment.Revision != "pinned" || deployment.Precision != "fp32" || deployment.Input.MaxTokens != 512 {
		t.Fatalf("deployment changed: %#v", deployment)
	}
	budget := saved.Global.ModelCatalog.Admission["local-pii"]
	if budget.MaxConcurrency != 2 || budget.MaxQueue != 3 || budget.QueueTimeoutMs != 500 || budget.OnOverflow != "shed" {
		t.Fatalf("admission changed: %#v", budget)
	}
	binding := saved.Recipes[0].Routing.ModelBindings["pii_classifier"]
	if binding.Deployment != "local-pii" || binding.Head != "pii" || binding.MappingPath != "mappings/pii.json" || binding.Contract != "token_spans.v1" {
		t.Fatalf("binding changed: %#v", binding)
	}
	if len(saved.Routing.ModelBindings) != 0 {
		t.Fatal("private binding leaked into default routing")
	}
}
