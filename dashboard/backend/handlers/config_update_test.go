package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"gopkg.in/yaml.v3"
)

func TestUpdateConfigHandler_ValidUpdates(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	tests := []struct {
		name     string
		method   string
		endpoint string
	}{
		{
			name:     "Valid canonical config update with valid IP address",
			method:   http.MethodPost,
			endpoint: "192.168.1.1:8000",
		},
		{
			name:     "Valid config - localhost (DNS name now allowed)",
			method:   http.MethodPost,
			endpoint: "localhost:8000",
		},
		{
			name:     "Valid config - domain name (DNS names now allowed)",
			method:   http.MethodPost,
			endpoint: "example.com:8000",
		},
		{
			name:     "PUT request should work",
			method:   http.MethodPut,
			endpoint: "10.0.0.1:8000",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			createValidTestConfig(t, tempDir)
			bodyBytes, err := json.Marshal(canonicalConfigBody(tt.endpoint))
			if err != nil {
				t.Fatalf("Failed to marshal request body: %v", err)
			}

			req := httptest.NewRequest(tt.method, "/api/router/config/update", bytes.NewReader(bodyBytes))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			UpdateConfigHandler(configPath, false, "")(w, req)

			if w.Code != http.StatusOK {
				t.Errorf("Expected status 200, got %d. Response body: %s", w.Code, w.Body.String())
			}

			var result map[string]string
			if decodeErr := json.NewDecoder(w.Body).Decode(&result); decodeErr != nil {
				t.Errorf("Response is not valid JSON: %v", decodeErr)
			}
			if result["status"] != "success" {
				t.Errorf("Expected status 'success', got '%s'", result["status"])
			}

			data, err := os.ReadFile(configPath)
			if err != nil {
				t.Errorf("Failed to read updated config file: %v", err)
			}
			if len(data) == 0 {
				t.Error("Config file is empty after update")
			}
		})
	}
}

func TestUpdateConfigHandler_InvalidUpdates(t *testing.T) {
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			createValidTestConfig(t, tempDir)

			var bodyBytes []byte
			var err error
			if str, ok := tt.requestBody.(string); ok {
				bodyBytes = []byte(str)
			} else {
				bodyBytes, err = json.Marshal(tt.requestBody)
				if err != nil {
					t.Fatalf("Failed to marshal request body: %v", err)
				}
			}

			req := httptest.NewRequest(tt.method, "/api/router/config/update", bytes.NewReader(bodyBytes))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			UpdateConfigHandler(configPath, false, "")(w, req)

			if w.Code != tt.expectedStatus {
				t.Errorf("Expected status %d, got %d. Response body: %s", tt.expectedStatus, w.Code, w.Body.String())
			}
			if tt.expectedError != "" && !contains(w.Body.String(), tt.expectedError) {
				t.Errorf("Expected error message to contain '%s', got: %s", tt.expectedError, w.Body.String())
			}
		})
	}
}

func TestUpdateConfigHandler_MethodNotAllowed(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	req := httptest.NewRequest(http.MethodGet, "/api/router/config/update", nil)
	w := httptest.NewRecorder()
	UpdateConfigHandler(configPath, false, "")(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("Expected status 405, got %d", w.Code)
	}
}

func TestUpdateConfigHandler_ReplacesLegacyLayout(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)
	createLegacyTestConfig(t, tempDir)

	bodyBytes, _ := json.Marshal(canonicalConfigBody("192.168.1.1:8000"))
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/update", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	UpdateConfigHandler(configPath, false, "")(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected status 200, got %d. Response: %s", w.Code, w.Body.String())
	}

	updatedData, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("Failed to read updated config: %v", err)
	}

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
}

func TestUpdateConfigHandler_WritesBackendEndpoint(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	bodyBytes, _ := json.Marshal(canonicalConfigBody("192.168.1.100:8000"))
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/update", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	UpdateConfigHandler(configPath, false, "")(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected status 200, got %d. Response: %s", w.Code, w.Body.String())
	}

	updatedData, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("Failed to read updated config: %v", err)
	}

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
}

func TestUpdateConfigHandler_UpdatesModTime(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	originalInfo, err := os.Stat(configPath)
	if err != nil {
		t.Fatalf("Failed to stat original config: %v", err)
	}
	originalModTime := originalInfo.ModTime()
	time.Sleep(100 * time.Millisecond)

	bodyBytes, _ := json.Marshal(canonicalConfigBody("127.0.0.1:8001"))
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/update", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	UpdateConfigHandler(configPath, false, "")(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected status 200, got %d. Response: %s", w.Code, w.Body.String())
	}

	updatedInfo, err := os.Stat(configPath)
	if err != nil {
		t.Fatalf("Failed to stat updated config: %v", err)
	}
	if !updatedInfo.ModTime().After(originalModTime) {
		t.Error("Config file modification time did not change after update")
	}
}

func TestUpdateConfigHandler_ValidationIntegration(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	invalidConfig := canonicalConfigBody("http://127.0.0.1:8000")
	bodyBytes, _ := json.Marshal(invalidConfig)
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/update", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	UpdateConfigHandler(configPath, false, "")(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400, got %d. Response: %s", w.Code, w.Body.String())
	}

	originalData, _ := os.ReadFile(configPath)
	if len(originalData) == 0 {
		t.Error("Original config file should not be empty")
	}
	if !contains(w.Body.String(), "Config validation failed") {
		t.Errorf("Expected validation error message, got: %s", w.Body.String())
	}
}
