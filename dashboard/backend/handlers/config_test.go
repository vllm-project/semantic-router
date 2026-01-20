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

// createValidTestConfig creates a minimal valid config file for testing
func createValidTestConfig(t *testing.T, dir string) string {
	configPath := filepath.Join(dir, "config.yaml")
	validConfig := `
bert_model:
  model_id: models/all-MiniLM-L12-v2
  threshold: 0.6
  use_cpu: true

classifier:
  category_model:
    model_id: models/lora_intent_classifier_bert-base-uncased_model
    threshold: 0.6
    use_cpu: true
    category_mapping_path: models/lora_intent_classifier_bert-base-uncased_model/category_mapping.json
  pii_model:
    model_id: models/lora_pii_detector_bert-base-uncased_model
    threshold: 0.9
    use_cpu: true
    pii_mapping_path: models/pii_classifier_modernbert-base_presidio_token_model/pii_type_mapping.json

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
	if err := os.WriteFile(configPath, []byte(validConfig), 0o644); err != nil {
		t.Fatalf("Failed to create test config file: %v", err)
	}
	return configPath
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
			name:   "Valid config update with valid IP address",
			method: http.MethodPost,
			requestBody: map[string]interface{}{
				"vllm_endpoints": []map[string]interface{}{
					{
						"name":    "endpoint1",
						"address": "192.168.1.1",
						"port":    8000,
						"weight":  1,
					},
				},
				"default_model": "test-model",
				"model_config": map[string]interface{}{
					"test-model": map[string]interface{}{
						"reasoning_family": "qwen3",
					},
				},
			},
			expectedStatus: http.StatusOK,
		},
		{
			name:   "Valid config - localhost (DNS name now allowed)",
			method: http.MethodPost,
			requestBody: map[string]interface{}{
				"vllm_endpoints": []map[string]interface{}{
					{
						"name":    "test",
						"address": "localhost",
						"port":    8000,
					},
				},
			},
			expectedStatus: http.StatusOK,
		},
		{
			name:   "Valid config - domain name (DNS names now allowed)",
			method: http.MethodPost,
			requestBody: map[string]interface{}{
				"vllm_endpoints": []map[string]interface{}{
					{
						"name":    "test",
						"address": "example.com",
						"port":    8000,
					},
				},
			},
			expectedStatus: http.StatusOK,
		},
		{
			name:   "Invalid config - protocol prefix in address",
			method: http.MethodPost,
			requestBody: map[string]interface{}{
				"vllm_endpoints": []map[string]interface{}{
					{
						"name":    "test",
						"address": "http://127.0.0.1",
						"port":    8000,
					},
				},
			},
			expectedStatus: http.StatusBadRequest,
			expectedError:  "Config validation failed",
		},
		{
			name:   "Invalid config - port in address field",
			method: http.MethodPost,
			requestBody: map[string]interface{}{
				"vllm_endpoints": []map[string]interface{}{
					{
						"name":    "test",
						"address": "127.0.0.1:8000",
						"port":    8000,
					},
				},
			},
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
			name:   "PUT request should work",
			method: http.MethodPut,
			requestBody: map[string]interface{}{
				"vllm_endpoints": []map[string]interface{}{
					{
						"name":    "endpoint1",
						"address": "10.0.0.1",
						"port":    8000,
						"weight":  1,
					},
				},
				"default_model": "test-model",
				"model_config": map[string]interface{}{
					"test-model": map[string]interface{}{
						"reasoning_family": "qwen3",
					},
				},
			},
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

			handler := UpdateConfigHandler(configPath, false)
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

	// Test 1: Add new top-level key
	t.Run("Add new top-level key", func(t *testing.T) {
		createValidTestConfig(t, tempDir) // Reset

		updateBody := map[string]interface{}{
			"test_new_key": "test_new_value",
		}

		bodyBytes, _ := json.Marshal(updateBody)
		req := httptest.NewRequest(http.MethodPost, "/api/router/config/update", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		handler := UpdateConfigHandler(configPath, false)
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

		// Verify new key exists
		if val, ok := updatedConfig["test_new_key"]; !ok {
			t.Error("New key 'test_new_key' was not added to config file")
		} else if val != "test_new_value" {
			t.Errorf("Expected 'test_new_key' to be 'test_new_value', got '%v'", val)
		}

		// Verify original values are preserved
		if val, ok := updatedConfig["default_model"]; !ok || val != "test-model" {
			t.Errorf("Original 'default_model' was not preserved. Got: %v", val)
		}
	})

	// Test 2: Update existing nested value
	t.Run("Update existing nested value", func(t *testing.T) {
		createValidTestConfig(t, tempDir) // Reset

		updateBody := map[string]interface{}{
			"vllm_endpoints": []map[string]interface{}{
				{
					"name":    "endpoint1",
					"address": "192.168.1.100", // Changed from 127.0.0.1
					"port":    8000,
					"weight":  1,
				},
			},
		}

		bodyBytes, _ := json.Marshal(updateBody)
		req := httptest.NewRequest(http.MethodPost, "/api/router/config/update", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		handler := UpdateConfigHandler(configPath, false)
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

		// Verify endpoint was updated
		endpoints, ok := updatedConfig["vllm_endpoints"].([]interface{})
		if !ok || len(endpoints) == 0 {
			t.Fatal("vllm_endpoints not found or empty in updated config")
		}

		endpoint, ok := endpoints[0].(map[string]interface{})
		if !ok {
			t.Fatal("First endpoint is not a map")
		}

		if address, ok := endpoint["address"].(string); !ok || address != "192.168.1.100" {
			t.Errorf("Expected address to be '192.168.1.100', got '%v'", address)
		}

		// Verify other values are preserved
		if val, ok := updatedConfig["default_model"]; !ok || val != "test-model" {
			t.Errorf("Original 'default_model' was not preserved. Got: %v", val)
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

		updateBody := map[string]interface{}{
			"test_timestamp_key": "test_timestamp_value",
		}

		bodyBytes, _ := json.Marshal(updateBody)
		req := httptest.NewRequest(http.MethodPost, "/api/router/config/update", bytes.NewReader(bodyBytes))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		handler := UpdateConfigHandler(configPath, false)
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
	invalidConfig := map[string]interface{}{
		"vllm_endpoints": []map[string]interface{}{
			{
				"name":    "invalid-endpoint",
				"address": "http://127.0.0.1", // Invalid: protocol prefix not allowed
				"port":    8000,
			},
		},
	}

	bodyBytes, _ := json.Marshal(invalidConfig)
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/update", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler := UpdateConfigHandler(configPath, false)
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
	handler := UpdateConfigHandler(configPath, true)
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

// TestUpdateRouterDefaultsHandler_ReadonlyMode verifies that readonly mode blocks router defaults updates
func TestUpdateRouterDefaultsHandler_ReadonlyMode(t *testing.T) {
	tempDir := t.TempDir()

	updateBody := map[string]interface{}{
		"test_key": "test_value",
	}

	bodyBytes, _ := json.Marshal(updateBody)
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/defaults/update", bytes.NewReader(bodyBytes))
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

	// Ensure router-defaults file was not created
	routerDefaultsPath := filepath.Join(tempDir, ".vllm-sr", "router-defaults.yaml")
	if _, err := os.Stat(routerDefaultsPath); err == nil {
		t.Errorf("Expected router-defaults.yaml not to be created in read-only mode")
	} else if !os.IsNotExist(err) {
		t.Errorf("Unexpected error checking router-defaults.yaml: %v", err)
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
