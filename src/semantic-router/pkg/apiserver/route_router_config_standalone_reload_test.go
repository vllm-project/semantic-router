package apiserver

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Standalone mode binds only its listeners at startup; its upstream layer
// rebuilds provider backends on reload. The hot-reload check says so, and
// never names Envoy there.
func TestStandaloneReloadNeedsARestartOnlyForListeners(t *testing.T) {
	current := minimalDeployTestConfig("route")

	backends := minimalDeployTestConfig("route")
	backends.VLLMEndpoints[0].Port++
	if err := validateParsedHotReloadCompatibilityInMode(config.GatewayStandalone, current, backends); err != nil {
		t.Fatalf("a provider backend change reloads in standalone mode: %v", err)
	}
	if err := validateParsedHotReloadCompatibilityInMode(config.GatewayExtProc, current, backends); err == nil ||
		!strings.Contains(err.Error(), "rendered into Envoy") {
		t.Fatalf("Envoy renders provider backends in extproc mode, got %v", err)
	}

	listeners := minimalDeployTestConfig("route")
	listeners.Listeners[0].Port = 8898
	err := validateParsedHotReloadCompatibilityInMode(config.GatewayStandalone, current, listeners)
	if err == nil || !strings.Contains(err.Error(), "standalone mode binds its listeners at startup") {
		t.Fatalf("a listener change needs a restart in standalone mode, got %v", err)
	}
	if strings.Contains(err.Error(), "Envoy") {
		t.Fatalf("the standalone message names Envoy: %v", err)
	}
}

// The validate endpoint returns the warnings the Router logs when it loads
// the document, so `vllm-sr config validate --endpoint` can print them.
func TestHandleConfigValidateReturnsLoadWarnings(t *testing.T) {
	document := strings.Join([]string{
		"version: v0.3",
		"listeners:",
		"  - name: http-8899",
		"    address: 0.0.0.0",
		"    port: 8899",
		"providers:",
		"  defaults:",
		"    model: m",
		"  models:",
		"    - name: m",
		"      backend_refs:",
		"        - endpoint: 127.0.0.1:8000",
		"routing:",
		"  modelCards:",
		"    - name: m",
		"  signals:",
		"    modality:",
		"      - name: DIFFUSION",
		"        description: Image generation requests.",
		"",
	}, "\n")
	body, err := json.Marshal(RouterConfigUpdateRequest{YAML: document})
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, apiConfigValidatePath, bytes.NewReader(body))
	(&ClassificationAPIServer{}).handleConfigValidate(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	var got RouterConfigValidateResponse
	if err := json.Unmarshal(response.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if !got.Valid || len(got.Warnings) != 1 || got.Warnings[0].Code != "modality_detector_disabled" {
		t.Fatalf("response = %+v, want one modality_detector_disabled warning", got)
	}
}
