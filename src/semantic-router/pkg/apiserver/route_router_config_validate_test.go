//go:build !windows && cgo

package apiserver

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func TestHandleConfigValidate(t *testing.T) {
	body := `{"yaml":"version: v0.3\nproviders:\n  defaults:\n    model: m1\n  models:\n    - name: m1\n      backend_refs:\n        - endpoint: 127.0.0.1:8000\n          provider: vllm\nrouting:\n  modelCards:\n    - name: m1\n"}`
	request := httptest.NewRequest("POST", "/api/v1/config/validate", strings.NewReader(body))
	response := httptest.NewRecorder()

	(&ClassificationAPIServer{}).handleConfigValidate(response, request)

	if response.Code != 200 {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	decoded := decodeValidateResponse(t, response.Body.Bytes())
	if !decoded.Valid || decoded.ContractVersion != config.DiagnosticContractVersion {
		t.Fatalf("response = %+v", decoded)
	}
}

func TestHandleConfigValidateRejectsUnknownFields(t *testing.T) {
	body := `{"yaml":"version: v0.3\n","dsl":"ignored legacy payload"}`
	request := httptest.NewRequest("POST", "/api/v1/config/validate", strings.NewReader(body))
	response := httptest.NewRecorder()

	(&ClassificationAPIServer{}).handleConfigValidate(response, request)

	if response.Code != 400 || !strings.Contains(response.Body.String(), "unknown field") {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
}

func TestHandleConfigValidateRejectsUnknownYAMLFields(t *testing.T) {
	body := `{"yaml":"version: v0.3\nproviders:\n  defaults:\n    model: m1\nrouting:\n  modelCards:\n    - name: m1\n      descriptin: typo\n"}`
	request := httptest.NewRequest("POST", "/api/v1/config/validate", strings.NewReader(body))
	response := httptest.NewRecorder()

	(&ClassificationAPIServer{}).handleConfigValidate(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	decoded := decodeValidateResponse(t, response.Body.Bytes())
	if decoded.Valid {
		t.Fatal("expected valid=false for an unknown YAML field")
	}
	if len(decoded.Errors) == 0 || decoded.Errors[0].Code != config.DiagnosticUnknownField {
		t.Fatalf("errors = %+v", decoded.Errors)
	}
	if !strings.Contains(decoded.Errors[0].Field, "descriptin") {
		t.Fatalf("field = %q, want descriptin", decoded.Errors[0].Field)
	}
}

func TestConfigValidateRouteRequiresReadPermission(t *testing.T) {
	for _, route := range apiConfigRoutes() {
		if route.Path == "/api/v1/config/validate" && route.Method == "POST" {
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
    model: m1
  models:
    - name: m1
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
          api_key_env: VALIDATE_SECRET_CANARY
routing:
  modelCards:
    - name: m1
      description: ${VALIDATE_SECRET_CANARY}
`
	body, err := json.Marshal(RouterConfigValidateRequest{YAML: yamlInput})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	request := httptest.NewRequest(
		"POST",
		"/api/v1/config/validate",
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

func TestHandleConfigValidatePreservesCredentialEnvironmentReference(t *testing.T) {
	yamlInput := `
version: v0.3
providers:
  defaults:
    model: m1
  models:
    - name: m1
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
          api_key: ${MODEL_API_KEY}
routing:
  modelCards:
    - name: m1
`
	body, err := json.Marshal(RouterConfigValidateRequest{YAML: yamlInput})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	request := httptest.NewRequest(
		"POST",
		"/api/v1/config/validate",
		strings.NewReader(string(body)),
	)
	response := httptest.NewRecorder()

	(&ClassificationAPIServer{}).handleConfigValidate(response, request)

	if response.Code != 200 {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	if !strings.Contains(response.Body.String(), "api_key: ${MODEL_API_KEY}") {
		t.Fatalf("validation response did not preserve credential reference: %s", response.Body.String())
	}
}

func TestHandleConfigValidateReturnsStructuredInvalidResult(t *testing.T) {
	body, err := json.Marshal(RouterConfigValidateRequest{
		YAML: "version: v0.3\nproviders: [",
	})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	response := httptest.NewRecorder()
	(&ClassificationAPIServer{}).handleConfigValidate(
		response,
		httptest.NewRequest(http.MethodPost, "/api/v1/config/validate", strings.NewReader(string(body))),
	)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	decoded := decodeValidateResponse(t, response.Body.Bytes())
	if decoded.Valid {
		t.Fatal("expected valid=false")
	}
	if len(decoded.Errors) == 0 || decoded.Errors[0].Code != config.DiagnosticYAMLParseError {
		t.Fatalf("errors = %+v", decoded.Errors)
	}
}

func TestHandleConfigValidateCompareToActive(t *testing.T) {
	configPath := writeValidateTestConfig(t, validateTestYAML("active-card", "active-secret"))
	body, err := json.Marshal(RouterConfigValidateRequest{
		YAML:            validateTestYAML("candidate-card", "candidate-secret"),
		CompareToActive: true,
	})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	response := httptest.NewRecorder()
	(&ClassificationAPIServer{configPath: configPath}).handleConfigValidate(
		response,
		httptest.NewRequest(http.MethodPost, "/api/v1/config/validate", strings.NewReader(string(body))),
	)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	decoded := decodeValidateResponse(t, response.Body.Bytes())
	if !decoded.Valid {
		t.Fatalf("expected valid candidate, errors=%+v", decoded.Errors)
	}
	if decoded.Diff == nil {
		t.Fatal("expected diff")
	}
	if decoded.Diff.Truncated {
		t.Fatal("expected an untruncated small diff")
	}
	foundChange := false
	for _, entry := range decoded.Diff.Changed {
		if strings.Contains(entry.Field, "description") {
			foundChange = true
			break
		}
	}
	if !foundChange {
		t.Fatalf("expected a model-card description change, got %+v", decoded.Diff)
	}
	if strings.Contains(response.Body.String(), "candidate-secret") || strings.Contains(response.Body.String(), "active-secret") {
		t.Fatalf("diff leaked a secret: %s", response.Body.String())
	}
	if !strings.Contains(response.Body.String(), "[REDACTED]") {
		t.Fatal("expected redacted secret values in the validate response")
	}
}

func TestHandleConfigValidateIsSideEffectFree(t *testing.T) {
	configPath := writeValidateTestConfig(t, validateTestYAML("active-card", "active-secret"))
	backupDir := filepath.Join(filepath.Dir(configPath), ".vllm-sr", "config-backups")
	if err := os.MkdirAll(backupDir, 0o755); err != nil {
		t.Fatalf("mkdir backups: %v", err)
	}
	backupFile := filepath.Join(backupDir, "config.20240101-000000.yaml")
	if err := os.WriteFile(backupFile, []byte("seed"), 0o644); err != nil {
		t.Fatalf("write backup: %v", err)
	}
	activeCfg, err := config.ParseYAMLBytesWithoutEnvExpansion([]byte(validateTestYAML("active-card", "active-secret")))
	if err != nil {
		t.Fatalf("parse active: %v", err)
	}
	registry := routerruntime.NewRegistry(activeCfg)
	beforeSource := mustReadFile(t, configPath)
	beforeBackup := mustReadDirNames(t, backupDir)
	beforeHash := registry.CurrentConfig().DocumentHash

	server := &ClassificationAPIServer{configPath: configPath, runtimeRegistry: registry}
	payloads := []RouterConfigValidateRequest{
		{YAML: validateTestYAML("candidate-card", "candidate-secret")},
		{YAML: "version: v0.3\nproviders: ["},
		{YAML: validateTestYAML("candidate-card", "candidate-secret"), CompareToActive: true},
	}
	for _, payload := range payloads {
		body, err := json.Marshal(payload)
		if err != nil {
			t.Fatalf("marshal request: %v", err)
		}
		response := httptest.NewRecorder()
		server.handleConfigValidate(
			response,
			httptest.NewRequest(http.MethodPost, "/api/v1/config/validate", strings.NewReader(string(body))),
		)
		if response.Code != http.StatusOK {
			t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
		}
	}

	afterSource := mustReadFile(t, configPath)
	if string(afterSource) != string(beforeSource) {
		t.Fatal("validate mutated the source config")
	}
	afterBackup := mustReadDirNames(t, backupDir)
	if strings.Join(afterBackup, ",") != strings.Join(beforeBackup, ",") {
		t.Fatalf("validate mutated backups: before=%v after=%v", beforeBackup, afterBackup)
	}
	if registry.CurrentConfig().DocumentHash != beforeHash {
		t.Fatal("validate mutated the runtime snapshot")
	}
}

func TestHandleConfigValidateCompareToActiveUsesInMemorySnapshot(t *testing.T) {
	desiredPath := writeValidateTestConfig(t, validateTestYAML("desired-card", "desired-secret"))
	activeCfg, err := config.ParseYAMLBytesWithoutEnvExpansion([]byte(validateTestYAML("active-card", "active-secret")))
	if err != nil {
		t.Fatalf("parse active: %v", err)
	}
	body, err := json.Marshal(RouterConfigValidateRequest{
		YAML:            validateTestYAML("desired-card", "desired-secret"),
		CompareToActive: true,
	})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	response := httptest.NewRecorder()
	(&ClassificationAPIServer{
		configPath:      desiredPath,
		runtimeRegistry: routerruntime.NewRegistry(activeCfg),
	}).handleConfigValidate(
		response,
		httptest.NewRequest(http.MethodPost, "/api/v1/config/validate", strings.NewReader(string(body))),
	)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	decoded := decodeValidateResponse(t, response.Body.Bytes())
	if decoded.Diff == nil {
		t.Fatal("expected a diff against the in-memory active snapshot")
	}
	foundActive := false
	for _, entry := range decoded.Diff.Changed {
		if strings.Contains(entry.Field, "description") && fmt.Sprint(entry.Old) == "active-card" {
			foundActive = true
			break
		}
	}
	if !foundActive {
		t.Fatalf("compare_to_active used desired source instead of in-memory active state: %+v", decoded.Diff)
	}
	if strings.Contains(response.Body.String(), "active-secret") || strings.Contains(response.Body.String(), "desired-secret") {
		t.Fatalf("diff leaked a secret: %s", response.Body.String())
	}
}

func TestHandleConfigValidateCompareToActiveUsesGeneratedRuntime(t *testing.T) {
	tempDir := t.TempDir()
	sourcePath := filepath.Join(tempDir, "config.yaml")
	runtimePath := filepath.Join(tempDir, ".vllm-sr", "runtime-config.yaml")
	if err := os.MkdirAll(filepath.Dir(runtimePath), 0o755); err != nil {
		t.Fatalf("mkdir runtime dir: %v", err)
	}
	if err := os.WriteFile(sourcePath, []byte(validateTestYAML("source-card", "source-secret")), 0o644); err != nil {
		t.Fatalf("write source: %v", err)
	}
	if err := os.WriteFile(runtimePath, []byte(validateTestYAML("runtime-card", "runtime-secret")), 0o644); err != nil {
		t.Fatalf("write runtime: %v", err)
	}
	body, err := json.Marshal(RouterConfigValidateRequest{
		YAML:            validateTestYAML("candidate-card", "candidate-secret"),
		CompareToActive: true,
	})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	response := httptest.NewRecorder()
	(&ClassificationAPIServer{configPath: runtimePath}).handleConfigValidate(
		response,
		httptest.NewRequest(http.MethodPost, "/api/v1/config/validate", strings.NewReader(string(body))),
	)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	decoded := decodeValidateResponse(t, response.Body.Bytes())
	if decoded.Diff == nil {
		t.Fatal("expected a diff against generated runtime state")
	}
	foundRuntime := false
	for _, entry := range decoded.Diff.Changed {
		if strings.Contains(entry.Field, "description") && fmt.Sprint(entry.Old) == "runtime-card" {
			foundRuntime = true
			break
		}
	}
	if !foundRuntime {
		t.Fatalf("compare_to_active used desired source instead of generated runtime: %+v", decoded.Diff)
	}
}

func TestHandleConfigValidateRejectsEmptyYAML(t *testing.T) {
	body, err := json.Marshal(RouterConfigValidateRequest{YAML: "   "})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	response := httptest.NewRecorder()
	(&ClassificationAPIServer{}).handleConfigValidate(
		response,
		httptest.NewRequest(http.MethodPost, "/api/v1/config/validate", strings.NewReader(string(body))),
	)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", response.Code)
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
    model: m1
  models:
    - name: m1
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
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

func decodeValidateResponse(t *testing.T, body []byte) RouterConfigValidateResponse {
	t.Helper()
	var decoded RouterConfigValidateResponse
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("decode response: %v body=%s", err, body)
	}
	return decoded
}

func writeValidateTestConfig(t *testing.T, yamlInput string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(yamlInput), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	return path
}

func validateTestYAML(description, apiKey string) string {
	return `
version: v0.3
listeners: []
providers:
  defaults:
    model: m1
  models:
    - name: m1
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
          api_key: ` + apiKey + `
routing:
  modelCards:
    - name: m1
      description: ` + description + `
  decisions:
    - name: d1
      priority: 1
      rules: {operator: AND, conditions: []}
      modelRefs:
        - model: m1
          use_reasoning: false
`
}

func mustReadDirNames(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("readdir %s: %v", dir, err)
	}
	names := make([]string, 0, len(entries))
	for _, entry := range entries {
		names = append(names, entry.Name())
	}
	return names
}
