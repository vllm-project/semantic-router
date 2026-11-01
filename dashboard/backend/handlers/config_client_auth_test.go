package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
)

// The Router rejects entrypoints[].client_auth by its presence until it
// enforces the policy. A null value vanishes once decoded into the canonical
// types, so the Dashboard must check the raw document instead of writing a
// config without it.
const clientAuthNotAvailable = "entrypoints[0].client_auth: entrypoint client authentication is not available yet"

func nullClientAuthEntrypoints() []map[string]interface{} {
	return []map[string]interface{}{{"model_names": []string{"company/fast"}, "recipe": "default", "client_auth": nil}}
}

func TestConfigEditorRejectsNullClientAuth(t *testing.T) {
	isolateConfigMutationRuntime(t)
	configPath := createValidTestConfig(t, t.TempDir())
	before, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	body := canonicalConfigBody("127.0.0.1:8000")
	body["entrypoints"] = nullClientAuthEntrypoints()
	payload, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, "/api/router/config/update", bytes.NewReader(payload))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	UpdateConfigHandler(configPath, false, "")(response, request)
	if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), clientAuthNotAvailable) {
		t.Fatalf("save = %d %s, want the Router's client_auth rejection", response.Code, response.Body.String())
	}
	if after, _ := os.ReadFile(configPath); !bytes.Equal(after, before) {
		t.Fatal("a rejected save changed the config")
	}
}

func TestSetupRejectsNullClientAuth(t *testing.T) {
	isolateConfigMutationRuntime(t)
	dir := t.TempDir()
	configPath := createBootstrapSetupConfig(t, dir)
	resolver := setupmode.New(configPath, false)
	patch := createValidSetupPatch()
	patch["entrypoints"] = nullClientAuthEntrypoints()
	payload := mustJSONRaw(t, SetupConfigRequest{Config: mustJSONRaw(t, patch)})
	for path, handler := range map[string]http.HandlerFunc{
		"/api/setup/validate": SetupValidateHandler(configPath, resolver),
		"/api/setup/activate": SetupActivateHandler(configPath, false, dir, resolver),
	} {
		response := httptest.NewRecorder()
		handler(response, httptest.NewRequest(http.MethodPost, path, bytes.NewReader(payload)))
		if response.Code == http.StatusOK || !strings.Contains(response.Body.String(), clientAuthNotAvailable) {
			t.Fatalf("%s = %d %s, want the Router's client_auth rejection", path, response.Code, response.Body.String())
		}
	}
	if !resolver.Active() {
		t.Fatal("a rejected activation left setup mode")
	}
}
