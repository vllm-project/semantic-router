//go:build !windows && cgo

package apiserver

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestHandleConfigValidate(t *testing.T) {
	body := `{"yaml":"version: v0.3\nproviders:\n  defaults:\n    default_model: m1\n  models:\n    - name: m1\n      backend_refs:\n        - endpoint: 127.0.0.1:8000\nrouting:\n  modelCards:\n    - name: m1\n"}`
	request := httptest.NewRequest("POST", "/config/router/validate", strings.NewReader(body))
	response := httptest.NewRecorder()

	(&ClassificationAPIServer{}).handleConfigValidate(response, request)

	if response.Code != 200 {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	if !strings.Contains(response.Body.String(), `"valid":true`) {
		t.Fatalf("body = %s", response.Body.String())
	}
}

func TestConfigValidateRouteRequiresReadPermission(t *testing.T) {
	for _, route := range apiConfigRoutes() {
		if route.Path == "/config/router/validate" && route.Method == "POST" {
			if route.Permission != PermConfigRead {
				t.Fatalf("permission = %q, want %q", route.Permission, PermConfigRead)
			}
			return
		}
	}
	t.Fatal("config validation route not found")
}

func TestHandleConfigValidateDoesNotExpandEnvironmentSecrets(t *testing.T) {
	const canary = "validation-secret-canary"
	t.Setenv("VALIDATE_SECRET_CANARY", canary)
	yamlInput := `
version: v0.3
providers:
  defaults:
    default_model: m1
  models:
    - name: m1
      backend_refs:
        - endpoint: 127.0.0.1:8000
          api_key_env: VALIDATE_SECRET_CANARY
routing:
  modelCards:
    - name: m1
      description: ${VALIDATE_SECRET_CANARY}
`
	body, err := json.Marshal(RouterConfigUpdateRequest{YAML: yamlInput})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	request := httptest.NewRequest(
		"POST",
		"/config/router/validate",
		strings.NewReader(string(body)),
	)
	request = request.WithContext(context.WithValue(
		request.Context(),
		managementPrincipalContextKey,
		managementPrincipal{Role: "admin", AuthEnabled: true},
	))
	response := httptest.NewRecorder()

	(&ClassificationAPIServer{}).handleConfigValidate(response, request)

	if response.Code != 200 {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	if strings.Contains(response.Body.String(), canary) {
		t.Fatal("validation response expanded an environment secret")
	}
	if !strings.Contains(response.Body.String(), "${VALIDATE_SECRET_CANARY}") {
		t.Fatalf("validation response did not preserve the environment reference: %s", response.Body.String())
	}
	if !strings.Contains(
		response.Body.String(),
		"api_key_env: VALIDATE_SECRET_CANARY",
	) {
		t.Fatalf("validation response did not preserve api_key_env: %s", response.Body.String())
	}
}

func TestValidateHotReloadCompatibilityRejectsLocalClassifierChange(t *testing.T) {
	current := []byte(localClassifierReloadConfig("models/risk-v1"))
	next := []byte(localClassifierReloadConfig("models/risk-v2"))

	if err := validateHotReloadCompatibility(current, next); err == nil {
		t.Fatal("expected restart-required local classifier reload error")
	}
}

func localClassifierReloadConfig(modelPath string) string {
	return `
version: v0.3
providers:
  defaults:
    default_model: m1
  models:
    - name: m1
      backend_refs:
        - endpoint: 127.0.0.1:8000
routing:
  modelCards:
    - name: m1
  signals:
    classifiers:
      - name: risk
        type: local
        model_path: ` + modelPath + `
        labels: [SAFE, RISKY]
  decisions: []
`
}
