package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"gopkg.in/yaml.v3"
)

func createBootstrapSetupConfig(t *testing.T, dir string) string {
	t.Helper()

	configPath := filepath.Join(dir, "config.yaml")
	config := map[string]interface{}{
		"version": "v0.1",
		"listeners": []map[string]interface{}{
			{
				"name":    "http-8899",
				"address": "0.0.0.0",
				"port":    8899,
				"timeout": "300s",
			},
		},
		"setup": map[string]interface{}{
			"mode":  true,
			"state": "bootstrap",
		},
	}

	data, err := yaml.Marshal(config)
	if err != nil {
		t.Fatalf("failed to marshal bootstrap config: %v", err)
	}
	if err := os.WriteFile(configPath, data, 0o644); err != nil {
		t.Fatalf("failed to write bootstrap config: %v", err)
	}
	return configPath
}

func createValidSetupPatch() map[string]interface{} {
	return map[string]interface{}{
		"signals": map[string]interface{}{
			"domains": []map[string]interface{}{
				{
					"name":        "other",
					"description": "General requests",
				},
			},
			"keywords": []map[string]interface{}{
				{
					"name":           "test_keywords",
					"operator":       "OR",
					"keywords":       []string{"test"},
					"case_sensitive": false,
				},
			},
		},
		"decisions": []map[string]interface{}{
			{
				"name":        "default_route",
				"description": "Default setup route",
				"priority":    100,
				"rules": map[string]interface{}{
					"operator": "AND",
					"conditions": []map[string]interface{}{
						{
							"type": "domain",
							"name": "other",
						},
						{
							"type": "keyword",
							"name": "test_keywords",
						},
					},
				},
				"modelRefs": []map[string]interface{}{
					{
						"model":         "test-model",
						"use_reasoning": false,
					},
				},
			},
		},
		"providers": map[string]interface{}{
			"models": []map[string]interface{}{
				{
					"name": "test-model",
					"endpoints": []map[string]interface{}{
						{
							"name":     "test-endpoint",
							"weight":   1,
							"endpoint": "host.docker.internal:8000",
							"protocol": "http",
						},
					},
				},
			},
			"default_model": "test-model",
		},
	}
}

func TestSetupStateHandler(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createBootstrapSetupConfig(t, tempDir)

	req := httptest.NewRequest(http.MethodGet, "/api/setup/state", nil)
	w := httptest.NewRecorder()

	SetupStateHandler(configPath)(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var resp SetupStateResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if !resp.SetupMode {
		t.Fatalf("expected setupMode=true")
	}
	if resp.ListenerPort != 8899 {
		t.Fatalf("expected listenerPort=8899, got %d", resp.ListenerPort)
	}
	if resp.Models != 0 || resp.Decisions != 0 {
		t.Fatalf("expected empty bootstrap counts, got models=%d decisions=%d", resp.Models, resp.Decisions)
	}
}

func TestSetupValidateHandler(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createBootstrapSetupConfig(t, tempDir)

	body, err := json.Marshal(SetupConfigRequest{Config: createValidSetupPatch()})
	if err != nil {
		t.Fatalf("failed to marshal request: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/setup/validate", bytes.NewReader(body))
	w := httptest.NewRecorder()

	SetupValidateHandler(configPath)(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp SetupValidateResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if !resp.Valid {
		t.Fatalf("expected valid=true")
	}
	if !resp.CanActivate {
		t.Fatalf("expected canActivate=true")
	}
	if _, hasSetup := resp.Config["setup"]; hasSetup {
		t.Fatalf("validated config should not contain setup marker")
	}
}

func TestSetupActivateHandler(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createBootstrapSetupConfig(t, tempDir)

	body, err := json.Marshal(SetupConfigRequest{Config: createValidSetupPatch()})
	if err != nil {
		t.Fatalf("failed to marshal request: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/setup/activate", bytes.NewReader(body))
	w := httptest.NewRecorder()

	SetupActivateHandler(configPath, false, tempDir)(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	configData, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("failed to read activated config: %v", err)
	}

	var configMap map[string]interface{}
	if err := yaml.Unmarshal(configData, &configMap); err != nil {
		t.Fatalf("failed to parse activated config: %v", err)
	}

	if _, hasSetup := configMap["setup"]; hasSetup {
		t.Fatalf("setup marker should be removed after activation")
	}

	if info, err := os.Stat(filepath.Join(tempDir, ".vllm-sr")); err != nil || !info.IsDir() {
		t.Fatalf(".vllm-sr output directory should exist after activation: %v", err)
	}
}
