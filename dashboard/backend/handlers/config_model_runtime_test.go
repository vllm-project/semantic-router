package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestUpdateConfigHandlerPreservesModelRuntimeDeclarations(t *testing.T) {
	const pinned = "0123456789abcdef0123456789abcdef01234567"
	configPath := createValidTestConfig(t, t.TempDir())
	body := canonicalConfigBody("127.0.0.1:8000")
	body["global"] = map[string]interface{}{"model_catalog": map[string]interface{}{
		"deployments": map[string]interface{}{
			"local-decision": map[string]interface{}{
				"artifact": "vllm-sr/checkpoint", "revision": pinned, "provider": "model_runtime", "device": "cpu",
				"input": map[string]interface{}{"max_tokens": 512, "overflow": "window"},
			},
			"local-embedding": map[string]interface{}{"artifact": "vllm-sr/embedding-checkpoint", "provider": "model_runtime", "device": "cpu"},
		},
		"admission": map[string]interface{}{"local-decision": map[string]interface{}{"max_concurrency": 2, "max_queue": 3, "queue_timeout_ms": 500, "on_overflow": "shed"}},
	}}
	body["recipes"] = []map[string]interface{}{{"name": "private", "routing": map[string]interface{}{
		"model_bindings": map[string]interface{}{
			"pii":       "local-decision",
			"jailbreak": "local-decision",
			"embedding": map[string]interface{}{"deployment": "local-embedding", "head": "onnx/model.onnx"},
		},
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
	deployment := saved.Global.ModelCatalog.Deployments["local-decision"]
	if deployment.Revision != pinned || !deployment.IsModelRuntime() || deployment.Input.MaxTokens != 512 || deployment.Input.Overflow != "window" {
		t.Fatalf("deployment changed: %#v", deployment)
	}
	budget := saved.Global.ModelCatalog.Admission["local-decision"]
	if budget.MaxConcurrency != 2 || budget.MaxQueue != 3 || budget.QueueTimeoutMs != 500 || budget.OnOverflow != "shed" {
		t.Fatalf("admission changed: %#v", budget)
	}
	binding := saved.Recipes[0].Routing.ModelBindings["embedding"]
	if binding.Deployment != "local-embedding" || binding.Head != "onnx/model.onnx" || binding.Contract != "" || binding.Adapter != "" {
		t.Fatalf("binding changed: %#v", binding)
	}
	if len(saved.Routing.ModelBindings) != 0 {
		t.Fatal("private binding leaked into default routing")
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	var document map[string]any
	if err := yaml.Unmarshal(data, &document); err != nil {
		t.Fatal(err)
	}
	recipes, ok := document["recipes"].([]any)
	if !ok || len(recipes) != 1 {
		t.Fatalf("recipes = %#v", document["recipes"])
	}
	recipe, ok := recipes[0].(map[string]any)
	if !ok {
		t.Fatalf("recipe = %#v", recipes[0])
	}
	recipeBindings := requireMapValue(t, requireMapValue(t, recipe, "routing"), "model_bindings")
	if recipeBindings["jailbreak"] != "local-decision" || recipeBindings["pii"] != "local-decision" {
		t.Fatalf("recipe binding is not written as its deployment name: %#v", recipeBindings)
	}
	if object := requireMapValue(t, recipeBindings, "embedding"); len(object) != 2 {
		t.Fatalf("a binding with a head writes only its deployment and head: %#v", object)
	}
}
