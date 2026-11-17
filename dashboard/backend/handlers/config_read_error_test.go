package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
)

func missingConfigPath(t *testing.T) string {
	t.Helper()
	return filepath.Join(t.TempDir(), "missing", "config.yaml")
}

func assertFixedReadError(response *httptest.ResponseRecorder, message, configPath string, t *testing.T) {
	t.Helper()
	if response.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500; body: %s", response.Code, response.Body.String())
	}
	if want := message + "\n"; response.Body.String() != want {
		t.Fatalf("body = %q, want %q", response.Body.String(), want)
	}
	if strings.Contains(response.Body.String(), configPath) {
		t.Fatalf("body leaks the server path: %s", response.Body.String())
	}
}

func TestUpdateConfigHandlerReadErrorHidesPath(t *testing.T) {
	configPath := missingConfigPath(t)
	payload, err := json.Marshal(canonicalConfigBody("127.0.0.1:8000"))
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/update", bytes.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	UpdateConfigHandler(configPath, false, "")(response, req)
	assertFixedReadError(response, "Failed to read existing config", configPath, t)
}

func TestUpdateRouterDefaultsHandlerReadErrorHidesPath(t *testing.T) {
	configPath := missingConfigPath(t)
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/global/update", strings.NewReader("{}"))
	response := httptest.NewRecorder()
	UpdateRouterDefaultsHandler(configPath, false, "")(response, req)
	assertFixedReadError(response, "Failed to read config", configPath, t)
}

func TestGlobalConfigYAMLHandlerReadErrorHidesPath(t *testing.T) {
	configPath := missingConfigPath(t)
	req := httptest.NewRequest(http.MethodGet, "/api/router/config/global/raw", nil)
	response := httptest.NewRecorder()
	GlobalConfigYAMLHandler(configPath)(response, req)
	assertFixedReadError(response, "Failed to read global config", configPath, t)
}

func TestUpdateGlobalConfigYAMLHandlerReadErrorHidesPath(t *testing.T) {
	configPath := missingConfigPath(t)
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/global/raw/update", strings.NewReader("global: {}"))
	response := httptest.NewRecorder()
	UpdateGlobalConfigYAMLHandler(configPath, false, "")(response, req)
	assertFixedReadError(response, "Failed to read config", configPath, t)
}
