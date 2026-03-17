package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"gopkg.in/yaml.v3"
)

// createValidTestConfig creates a minimal canonical v0.3 config file for testing.
func createValidTestConfig(t *testing.T, dir string) string {
	configPath := filepath.Join(dir, "config.yaml")
	validConfig := `
version: v0.3
listeners:
  - name: public
    address: 0.0.0.0
    port: 8801
providers:
  defaults:
    default_model: test-model
    reasoning_families:
      qwen3:
        type: reasoning_effort
        parameter: reasoning_effort
  models:
    - name: test-model
      reasoning_family: qwen3
      provider_model_id: test-model
      backend_refs:
        - name: endpoint1
          endpoint: 127.0.0.1:8000
          protocol: http
          weight: 1
routing:
  modelCards:
    - name: test-model
  signals:
    domains:
      - name: business
        description: Business and management related queries
  decisions:
    - name: default-business
      description: Route business requests to the default model
      priority: 1
      rules:
        operator: OR
        conditions:
          - type: domain
            name: business
      modelRefs:
        - model: test-model
          use_reasoning: false
`
	if err := os.WriteFile(configPath, []byte(validConfig), 0o644); err != nil {
		t.Fatalf("Failed to create test config file: %v", err)
	}
	return configPath
}

func createLegacyTestConfig(t *testing.T, dir string) string {
	configPath := filepath.Join(dir, "config.yaml")
	legacyConfig := `
categories:
  - name: business
    description: Business and management related queries

vllm_endpoints:
  - name: endpoint1
    address: 127.0.0.1
    port: 8000
    weight: 1

default_model: test-model

model_config:
  test-model:
    reasoning_family: qwen3
`
	if err := os.WriteFile(configPath, []byte(legacyConfig), 0o644); err != nil {
		t.Fatalf("Failed to create legacy test config file: %v", err)
	}
	return configPath
}

func canonicalConfigBody(endpoint string) map[string]interface{} {
	return map[string]interface{}{
		"version": "v0.3",
		"listeners": []map[string]interface{}{
			{
				"name":    "public",
				"address": "0.0.0.0",
				"port":    8801,
			},
		},
		"providers": map[string]interface{}{
			"defaults": map[string]interface{}{
				"default_model": "test-model",
				"reasoning_families": map[string]interface{}{
					"qwen3": map[string]interface{}{
						"type":      "reasoning_effort",
						"parameter": "reasoning_effort",
					},
				},
			},
			"models": []map[string]interface{}{
				{
					"name":              "test-model",
					"reasoning_family":  "qwen3",
					"provider_model_id": "test-model",
					"backend_refs": []map[string]interface{}{
						{
							"name":     "endpoint1",
							"endpoint": endpoint,
							"protocol": "http",
							"weight":   1,
						},
					},
				},
			},
		},
		"routing": map[string]interface{}{
			"modelCards": []map[string]interface{}{
				{
					"name": "test-model",
				},
			},
			"signals": map[string]interface{}{
				"domains": []map[string]interface{}{
					{
						"name":        "business",
						"description": "Business and management related queries",
					},
				},
			},
			"decisions": []map[string]interface{}{
				{
					"name":        "default-business",
					"description": "Route business requests to the default model",
					"priority":    1,
					"rules": map[string]interface{}{
						"operator": "OR",
						"conditions": []map[string]interface{}{
							{
								"type": "domain",
								"name": "business",
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
		},
	}
}

func TestConfigHandler(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	tests := []struct {
		name           string
		method         string
		expectedStatus int
	}{
		{
			name:           "GET request should succeed",
			method:         http.MethodGet,
			expectedStatus: http.StatusOK,
		},
		{
			name:           "POST request should fail",
			method:         http.MethodPost,
			expectedStatus: http.StatusMethodNotAllowed,
		},
		{
			name:           "PUT request should fail",
			method:         http.MethodPut,
			expectedStatus: http.StatusMethodNotAllowed,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(tt.method, "/api/router/config/all", nil)
			w := httptest.NewRecorder()

			handler := ConfigHandler(configPath)
			handler(w, req)

			if w.Code != tt.expectedStatus {
				t.Errorf("Expected status %d, got %d", tt.expectedStatus, w.Code)
			}

			if tt.expectedStatus == http.StatusOK {
				// Verify response is valid JSON
				var result interface{}
				if err := json.NewDecoder(w.Body).Decode(&result); err != nil {
					t.Errorf("Response is not valid JSON: %v", err)
				}
			}
		})
	}
}

func TestUpdateConfigHandler(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	tests := []struct {
		name           string
		method         string
		requestBody    interface{}
		expectedStatus int
		expectedError  string
	}{
		{
			name:           "Valid canonical config update with valid IP address",
			method:         http.MethodPost,
			requestBody:    canonicalConfigBody("192.168.1.1:8000"),
			expectedStatus: http.StatusOK,
		},
		{
			name:           "Valid config - localhost (DNS name now allowed)",
			method:         http.MethodPost,
			requestBody:    canonicalConfigBody("localhost:8000"),
			expectedStatus: http.StatusOK,
		},
		{
			name:           "Valid config - domain name (DNS names now allowed)",
			method:         http.MethodPost,
			requestBody:    canonicalConfigBody("example.com:8000"),
			expectedStatus: http.StatusOK,
		},
		{
			name:           "Invalid config - protocol prefix in address",
			method:         http.MethodPost,
			requestBody:    canonicalConfigBody("http://127.0.0.1:8000"),
			expectedStatus: http.StatusBadRequest,
			expectedError:  "Config validation failed",
		},
		{
			name:           "Invalid config - path in backend ref endpoint",
			method:         http.MethodPost,
			requestBody:    canonicalConfigBody("127.0.0.1:8000/v1"),
			expectedStatus: http.StatusBadRequest,
			expectedError:  "Config validation failed",
		},
		{
			name:           "Invalid JSON body",
			method:         http.MethodPost,
			requestBody:    "invalid json",
			expectedStatus: http.StatusBadRequest,
			expectedError:  "Invalid request body",
		},
		{
			name:           "GET request should fail",
			method:         http.MethodGet,
			requestBody:    nil,
			expectedStatus: http.StatusMethodNotAllowed,
		},
		{
			name:           "PUT request should work",
			method:         http.MethodPut,
			requestBody:    canonicalConfigBody("10.0.0.1:8000"),
			expectedStatus: http.StatusOK,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Reset config file before each test
			createValidTestConfig(t, tempDir)

			var bodyBytes []byte
			var err error

			if tt.requestBody != nil {
				if str, ok := tt.requestBody.(string); ok {
					// For invalid JSON test
					bodyBytes = []byte(str)
				} else {
					bodyBytes, err = json.Marshal(tt.requestBody)
					if err != nil {
						t.Fatalf("Failed to marshal request body: %v", err)
					}
				}
			}

			req := httptest.NewRequest(tt.method, "/api/router/config/update", bytes.NewReader(bodyBytes))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			handler := UpdateConfigHandler(configPath, false, "")
			handler(w, req)

			if w.Code != tt.expectedStatus {
				t.Errorf("Expected status %d, got %d. Response body: %s", tt.expectedStatus, w.Code, w.Body.String())
			}

			if tt.expectedError != "" {
				body := w.Body.String()
				if !contains(body, tt.expectedError) {
					t.Errorf("Expected error message to contain '%s', got: %s", tt.expectedError, body)
				}
			}

			if tt.expectedStatus == http.StatusOK {
				// Verify response is valid JSON with success message
				var result map[string]string
				if err := json.NewDecoder(w.Body).Decode(&result); err != nil {
					t.Errorf("Response is not valid JSON: %v", err)
				}
				if result["status"] != "success" {
					t.Errorf("Expected status 'success', got '%s'", result["status"])
				}

				// Verify config file was actually updated
				data, err := os.ReadFile(configPath)
				if err != nil {
					t.Errorf("Failed to read updated config file: %v", err)
				}
				if len(data) == 0 {
					t.Error("Config file is empty after update")
				}
			}
		})
	}
}

// TestUpdateConfigHandler_FilePersistence verifies that config updates are actually written to disk
func TestUpdateConfigHandler_FilePersistence(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	// Test 1: canonical write should replace any legacy on-disk layout
	t.Run("Canonical update replaces legacy runtime fields", func(t *testing.T) {
		createLegacyTestConfig(t, tempDir)

		updateBody := canonicalConfigBody("192.168.1.1:8000")

		bodyBytes, _ := json.Marshal(updateBody)
		req := httptest.NewRequest(http.MethodPost, "/api/router/config/update", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		handler := UpdateConfigHandler(configPath, false, "")
		handler(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("Expected status 200, got %d. Response: %s", w.Code, w.Body.String())
		}

		// Verify file was updated
		updatedData, err := os.ReadFile(configPath)
		if err != nil {
			t.Fatalf("Failed to read updated config: %v", err)
		}

		// Parse updated config (YAML format)
		var updatedConfig map[string]interface{}
		if err := yaml.Unmarshal(updatedData, &updatedConfig); err != nil {
			t.Fatalf("Failed to parse updated config: %v", err)
		}

		if _, ok := updatedConfig["providers"]; !ok {
			t.Fatal("Expected canonical providers block to be present")
		}
		if _, ok := updatedConfig["routing"]; !ok {
			t.Fatal("Expected canonical routing block to be present")
		}
		if _, ok := updatedConfig["model_config"]; ok {
			t.Error("Legacy model_config should not be preserved after canonical save")
		}
		if _, ok := updatedConfig["vllm_endpoints"]; ok {
			t.Error("Legacy vllm_endpoints should not be preserved after canonical save")
		}
	})

	// Test 2: verify canonical payload is written as sent
	t.Run("Canonical update writes backend endpoint", func(t *testing.T) {
		createValidTestConfig(t, tempDir) // Reset

		updateBody := canonicalConfigBody("192.168.1.100:8000")

		bodyBytes, _ := json.Marshal(updateBody)
		req := httptest.NewRequest(http.MethodPost, "/api/router/config/update", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		handler := UpdateConfigHandler(configPath, false, "")
		handler(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("Expected status 200, got %d. Response: %s", w.Code, w.Body.String())
		}

		// Verify file was updated
		updatedData, err := os.ReadFile(configPath)
		if err != nil {
			t.Fatalf("Failed to read updated config: %v", err)
		}

		// Parse updated config (YAML format)
		var updatedConfig map[string]interface{}
		if err := yaml.Unmarshal(updatedData, &updatedConfig); err != nil {
			t.Fatalf("Failed to parse updated config: %v", err)
		}

		providers, ok := updatedConfig["providers"].(map[string]interface{})
		if !ok {
			t.Fatal("providers block not found in updated config")
		}
		models, ok := providers["models"].([]interface{})
		if !ok || len(models) == 0 {
			t.Fatal("providers.models not found or empty in updated config")
		}
		model, ok := models[0].(map[string]interface{})
		if !ok {
			t.Fatal("First provider model is not a map")
		}
		backendRefs, ok := model["backend_refs"].([]interface{})
		if !ok || len(backendRefs) == 0 {
			t.Fatal("backend_refs not found or empty in updated config")
		}
		backend, ok := backendRefs[0].(map[string]interface{})
		if !ok {
			t.Fatal("First backend ref is not a map")
		}
		if endpoint, ok := backend["endpoint"].(string); !ok || endpoint != "192.168.1.100:8000" {
			t.Errorf("Expected endpoint to be '192.168.1.100:8000', got '%v'", endpoint)
		}
	})

	// Test 3: Verify file modification timestamp changes
	t.Run("File modification timestamp changes", func(t *testing.T) {
		createValidTestConfig(t, tempDir) // Reset

		// Get original file info
		originalInfo, err := os.Stat(configPath)
		if err != nil {
			t.Fatalf("Failed to stat original config: %v", err)
		}
		originalModTime := originalInfo.ModTime()

		// Wait a bit to ensure timestamp difference
		time.Sleep(100 * time.Millisecond)

		updateBody := canonicalConfigBody("127.0.0.1:8001")

		bodyBytes, _ := json.Marshal(updateBody)
		req := httptest.NewRequest(http.MethodPost, "/api/router/config/update", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		handler := UpdateConfigHandler(configPath, false, "")
		handler(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("Expected status 200, got %d. Response: %s", w.Code, w.Body.String())
		}

		// Verify file modification time changed
		updatedInfo, err := os.Stat(configPath)
		if err != nil {
			t.Fatalf("Failed to stat updated config: %v", err)
		}

		if !updatedInfo.ModTime().After(originalModTime) {
			t.Error("Config file modification time did not change after update")
		}
	})
}

func TestUpdateConfigHandler_ValidationIntegration(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	// Test that validation prevents saving invalid config
	invalidConfig := canonicalConfigBody("http://127.0.0.1:8000")

	bodyBytes, _ := json.Marshal(invalidConfig)
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/update", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler := UpdateConfigHandler(configPath, false, "")
	handler(w, req)

	// Should return 400 Bad Request
	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400, got %d. Response: %s", w.Code, w.Body.String())
	}

	// Verify original config file was not modified
	originalData, _ := os.ReadFile(configPath)
	if len(originalData) == 0 {
		t.Error("Original config file should not be empty")
	}

	// Verify error message contains validation error
	body := w.Body.String()
	if !contains(body, "Config validation failed") {
		t.Errorf("Expected validation error message, got: %s", body)
	}
}

// TestUpdateConfigHandler_ReadonlyMode verifies that readonly mode blocks write operations
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

	// Enable readonly mode
	handler := UpdateConfigHandler(configPath, true, "")
	handler(w, req)

	// Should return 403 Forbidden
	if w.Code != http.StatusForbidden {
		t.Errorf("Expected status 403, got %d. Response: %s", w.Code, w.Body.String())
	}

	// Verify error message
	body := w.Body.String()
	if !contains(body, "read-only mode") {
		t.Errorf("Expected 'read-only mode' in error message, got: %s", body)
	}
}

// TestUpdateRouterDefaultsHandler_ReadonlyMode verifies that readonly mode blocks global runtime updates.
func TestUpdateRouterDefaultsHandler_ReadonlyMode(t *testing.T) {
	tempDir := t.TempDir()

	updateBody := map[string]interface{}{
		"test_key": "test_value",
	}

	bodyBytes, _ := json.Marshal(updateBody)
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/global/update", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler := UpdateRouterDefaultsHandler(tempDir, true)
	handler(w, req)

	// Should return 403 Forbidden
	if w.Code != http.StatusForbidden {
		t.Errorf("Expected status 403, got %d. Response: %s", w.Code, w.Body.String())
	}

	// Verify error message
	body := w.Body.String()
	if !contains(body, "read-only mode") {
		t.Errorf("Expected 'read-only mode' in error message, got: %s", body)
	}

	// Ensure no defaults reference file was created as a side effect.
	defaultsPath := filepath.Join(tempDir, ".vllm-sr", "global-defaults.yaml")
	if _, err := os.Stat(defaultsPath); err == nil {
		t.Errorf("Expected global-defaults.yaml not to be created in read-only mode")
	} else if !os.IsNotExist(err) {
		t.Errorf("Unexpected error checking global-defaults.yaml: %v", err)
	}
}

// Helper function to check if a string contains a substring
func contains(s, substr string) bool {
	if len(substr) == 0 {
		return true
	}
	if len(s) < len(substr) {
		return false
	}
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
