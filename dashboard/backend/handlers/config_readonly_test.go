package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestUpdateConfigHandler_ReadonlyMode(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	updateBody := map[string]interface{}{
		"test_key": "test_value",
	}

	bodyBytes, _ := json.Marshal(updateBody)
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/update", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	UpdateConfigHandler(configPath, true, "")(w, req)

	if w.Code != http.StatusForbidden {
		t.Errorf("Expected status 403, got %d. Response: %s", w.Code, w.Body.String())
	}
	if !contains(w.Body.String(), "read-only mode") {
		t.Errorf("Expected 'read-only mode' in error message, got: %s", w.Body.String())
	}
}

func TestUpdateRouterDefaultsHandler_ReadonlyMode(t *testing.T) {
	tempDir := t.TempDir()

	updateBody := map[string]interface{}{
		"test_key": "test_value",
	}

	bodyBytes, _ := json.Marshal(updateBody)
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/global/update", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	UpdateRouterDefaultsHandler(tempDir, true)(w, req)

	if w.Code != http.StatusForbidden {
		t.Errorf("Expected status 403, got %d. Response: %s", w.Code, w.Body.String())
	}
	if !contains(w.Body.String(), "read-only mode") {
		t.Errorf("Expected 'read-only mode' in error message, got: %s", w.Body.String())
	}

	defaultsPath := filepath.Join(tempDir, ".vllm-sr", "global-defaults.yaml")
	if _, err := os.Stat(defaultsPath); err == nil {
		t.Errorf("Expected global-defaults.yaml not to be created in read-only mode")
	} else if !os.IsNotExist(err) {
		t.Errorf("Unexpected error checking global-defaults.yaml: %v", err)
	}
}
