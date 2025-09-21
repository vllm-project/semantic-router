package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func TestHandleBatchClassification(t *testing.T) {
	// Create a test server with placeholder service
	apiServer := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config:            &config.RouterConfig{},
	}

	tests := []struct {
		name           string
		requestBody    string
		expectedStatus int
		expectedError  string
	}{
		{
			name: "Valid small batch",
			requestBody: `{
				"texts": ["What is machine learning?", "How to invest in stocks?"],
				"task_type": "intent"
			}`,
			expectedStatus: http.StatusServiceUnavailable,
			expectedError:  "Batch classification requires unified classifier. Please ensure models are available in ./models/ directory.",
		},
		{
			name: "Valid large batch",
			requestBody: func() string {
				texts := make([]string, 50)
				for i := range texts {
					texts[i] = fmt.Sprintf("Test text %d", i)
				}
				data := map[string]interface{}{
					"texts":     texts,
					"task_type": "intent",
				}
				b, _ := json.Marshal(data)
				return string(b)
			}(),
			expectedStatus: http.StatusServiceUnavailable,
			expectedError:  "Batch classification requires unified classifier. Please ensure models are available in ./models/ directory.",
		},
		{
			name: "Valid batch with options",
			requestBody: `{
				"texts": ["What is quantum physics?"],
				"task_type": "intent",
				"options": {
					"include_probabilities": true
				}
			}`,
			expectedStatus: http.StatusServiceUnavailable,
			expectedError:  "Batch classification requires unified classifier. Please ensure models are available in ./models/ directory.",
		},
		{
			name:           "Empty texts array",
			requestBody:    `{"texts": [], "task_type": "intent"}`,
			expectedStatus: http.StatusBadRequest,
			expectedError:  "texts array cannot be empty",
		},
		{
			name:           "Missing texts field",
			requestBody:    `{"task_type": "intent"}`,
			expectedStatus: http.StatusBadRequest,
			expectedError:  "texts field is required",
		},
		{
			name: "Batch too large",
			requestBody: func() string {
				texts := make([]string, 101)
				for i := range texts {
					texts[i] = fmt.Sprintf("Test text %d", i)
				}
				data := map[string]interface{}{"texts": texts}
				b, _ := json.Marshal(data)
				return string(b)
			}(),
			expectedStatus: http.StatusServiceUnavailable,
			expectedError:  "Batch classification requires unified classifier. Please ensure models are available in ./models/ directory.",
		},
		{
			name:           "Invalid JSON",
			requestBody:    `{"texts": [invalid json`,
			expectedStatus: http.StatusBadRequest,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest("POST", "/api/v1/classify/batch", bytes.NewBufferString(tt.requestBody))
			req.Header.Set("Content-Type", "application/json")

			rr := httptest.NewRecorder()

			apiServer.handleBatchClassification(rr, req)

			if rr.Code != tt.expectedStatus {
				t.Errorf("Expected status %d, got %d", tt.expectedStatus, rr.Code)
			}

			if tt.expectedStatus == http.StatusOK {
				// For successful requests, check response structure
				var response BatchClassificationResponse
				if err := json.Unmarshal(rr.Body.Bytes(), &response); err != nil {
					t.Errorf("Failed to unmarshal response: %v", err)
				}

				// Validate response structure
				if response.TotalCount == 0 {
					t.Error("Expected non-zero total count")
				}
				if len(response.Results) == 0 {
					t.Error("Expected non-empty results")
				}
				if response.ProcessingTimeMs < 0 {
					t.Error("Expected non-negative processing time")
				}

				// Check statistics
				if response.Statistics.AvgConfidence < 0 || response.Statistics.AvgConfidence > 1 {
					t.Error("Expected confidence between 0 and 1")
				}
			} else if tt.expectedError != "" {
				// For error responses, check error message
				var errorResponse map[string]interface{}
				if err := json.Unmarshal(rr.Body.Bytes(), &errorResponse); err != nil {
					t.Errorf("Failed to unmarshal error response: %v", err)
				}

				if errorData, ok := errorResponse["error"].(map[string]interface{}); ok {
					if message, ok := errorData["message"].(string); ok {
						if message != tt.expectedError {
							t.Errorf("Expected error message '%s', got '%s'", tt.expectedError, message)
						}
					}
				}
			}
		})
	}
}

func TestBatchClassificationConfiguration(t *testing.T) {
	tests := []struct {
		name           string
		config         *config.RouterConfig
		requestBody    string
		expectedStatus int
		expectedError  string
	}{
		{
			name: "Custom max batch size",
			config: &config.RouterConfig{
				API: config.APIConfig{
					BatchClassification: struct {
						Metrics config.BatchClassificationMetricsConfig `yaml:"metrics,omitempty"`
					}{
						Metrics: config.BatchClassificationMetricsConfig{
							Enabled: true,
						},
					},
				},
			},
			requestBody: `{
				"texts": ["text1", "text2", "text3", "text4"]
			}`,
			expectedStatus: http.StatusServiceUnavailable,
			expectedError:  "Batch classification requires unified classifier. Please ensure models are available in ./models/ directory.",
		},
		{
			name:   "Default config when config is nil",
			config: nil,
			requestBody: func() string {
				texts := make([]string, 101)
				for i := range texts {
					texts[i] = fmt.Sprintf("test query %d", i)
				}
				data := map[string]interface{}{"texts": texts}
				b, _ := json.Marshal(data)
				return string(b)
			}(),
			expectedStatus: http.StatusServiceUnavailable,
			expectedError:  "Batch classification requires unified classifier. Please ensure models are available in ./models/ directory.",
		},
		{
			name: "Valid request within custom limits",
			config: &config.RouterConfig{
				API: config.APIConfig{
					BatchClassification: struct {
						Metrics config.BatchClassificationMetricsConfig `yaml:"metrics,omitempty"`
					}{
						Metrics: config.BatchClassificationMetricsConfig{
							Enabled: true,
						},
					},
				},
			},
			requestBody: `{
				"texts": ["text1", "text2"]
			}`,
			expectedStatus: http.StatusServiceUnavailable,
			expectedError:  "Batch classification requires unified classifier. Please ensure models are available in ./models/ directory.",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			apiServer := &ClassificationAPIServer{
				classificationSvc: services.NewPlaceholderClassificationService(),
				config:            tt.config,
			}

			req := httptest.NewRequest("POST", "/api/v1/classify/batch", bytes.NewBufferString(tt.requestBody))
			req.Header.Set("Content-Type", "application/json")

			rr := httptest.NewRecorder()

			apiServer.handleBatchClassification(rr, req)

			if rr.Code != tt.expectedStatus {
				t.Errorf("Expected status %d, got %d", tt.expectedStatus, rr.Code)
			}

			if tt.expectedError != "" {
				var errorResponse map[string]interface{}
				if err := json.Unmarshal(rr.Body.Bytes(), &errorResponse); err != nil {
					t.Errorf("Failed to unmarshal error response: %v", err)
				}

				if errorData, ok := errorResponse["error"].(map[string]interface{}); ok {
					if message, ok := errorData["message"].(string); ok {
						if message != tt.expectedError {
							t.Errorf("Expected error message '%s', got '%s'", tt.expectedError, message)
						}
					}
				}
			}
		})
	}
}

func TestOpenAIModelsEndpoint(t *testing.T) {
	cfg := &config.RouterConfig{
		VLLMEndpoints: []config.VLLMEndpoint{
			{
				Name:    "primary",
				Address: "localhost",
				Port:    8000,
				Models:  []string{"gpt-4o-mini", "llama-3.1-8b-instruct"},
				Weight:  1,
			},
		},
	}

	apiServer := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config:            cfg,
	}

	req := httptest.NewRequest("GET", "/v1/models", nil)
	rr := httptest.NewRecorder()

	apiServer.handleOpenAIModels(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d", rr.Code)
	}

	var resp OpenAIModelList
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}

	if resp.Object != "list" {
		t.Errorf("expected object 'list', got %s", resp.Object)
	}

	// Build a set for easy lookup
	got := map[string]bool{}
	for _, m := range resp.Data {
		got[m.ID] = true
		if m.Object != "model" {
			t.Errorf("expected each item.object to be 'model', got %s", m.Object)
		}
		if m.Created == 0 {
			t.Errorf("expected created timestamp to be non-zero")
		}
	}

	// Must contain 'auto' and the configured models
	if !got["auto"] {
		t.Errorf("expected list to contain 'auto'")
	}
	if !got["gpt-4o-mini"] || !got["llama-3.1-8b-instruct"] {
		t.Errorf("expected configured models to be present, got=%v", got)
	}
}
