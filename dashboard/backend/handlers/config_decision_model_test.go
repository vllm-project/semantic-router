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

func TestDecisionModelApplyPreservesOverridesAndReportsRestart(t *testing.T) {
	root := t.TempDir()
	path := createValidTestConfig(t, root)
	initial, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	initial = append(initial, []byte(`
global:
  model_catalog:
    system:
      decision_model: {deployment: primary}
      pii_classifier: models/Vela-1.0-Encoder-307M-PII
  router:
    strategy: priority
`)...)
	if writeErr := os.WriteFile(path, initial, 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	original := propagateConfig
	t.Cleanup(func() { propagateConfig = original })
	propagateConfig = func(string, string) error {
		return &restartNeededError{detail: "decision model deployment changed"}
	}
	patch := []byte(`{"model_catalog":{"deployments":{"selected":{"provider":"model_runtime","artifact":"vllm-sr/Vela-2.0-0.8B","device":"auto"}},"system":{"decision_model":{"deployment":"selected"}}}}`)
	response := httptest.NewRecorder()
	UpdateRouterDefaultsHandler(path, false, root)(response,
		httptest.NewRequest(http.MethodPost, "/api/router/config/global/update", bytes.NewReader(patch)))
	var result map[string]string
	if decodeErr := json.Unmarshal(response.Body.Bytes(), &result); decodeErr != nil {
		t.Fatalf("response %d: %s", response.Code, response.Body.String())
	}
	if response.Code != http.StatusAccepted || result["status"] != "restart_required" {
		t.Fatalf("response = %d %v; saved must not be reported as deployed", response.Code, result)
	}
	saved, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var document map[string]any
	if decodeErr := yaml.Unmarshal(saved, &document); decodeErr != nil {
		t.Fatal(decodeErr)
	}
	global := requireMapValue(t, document, "global")
	system := requireMapValue(t, requireMapValue(t, global, "model_catalog"), "system")
	if requireMapValue(t, system, "decision_model")["deployment"] != "selected" || system["pii_classifier"] != "models/Vela-1.0-Encoder-307M-PII" || len(system) != 2 {
		t.Fatalf("selection changed explicit or inherited bindings: %+v", system)
	}
	if requireMapValue(t, global, "router")["strategy"] != "priority" {
		t.Fatal("decision model apply discarded other global settings")
	}
	if !pendingActivationRecorded(path) {
		t.Fatal("decision model restart was not recorded for the CLI")
	}
}

func TestDecisionModelInvalidSelectionDoesNotPersist(t *testing.T) {
	root := t.TempDir()
	path := createValidTestConfig(t, root)
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	UpdateRouterDefaultsHandler(path, false, root)(response,
		httptest.NewRequest(http.MethodPost, "/api/router/config/global/update",
			bytes.NewBufferString(`{"model_catalog":{"system":{"decision_model":"not-a-model"}}}`)))
	if response.Code != http.StatusBadRequest {
		t.Fatalf("response = %d: %s", response.Code, response.Body.String())
	}
	after, err := os.ReadFile(path)
	if err != nil || !bytes.Equal(before, after) {
		t.Fatalf("invalid selection changed config: %v", err)
	}
}
