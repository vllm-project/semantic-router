package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
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
				"texts": ["solve math equation", "write business plan", "chemistry experiment"]
			}`,
			expectedStatus: http.StatusOK,
		},
		{
			name: "Valid large batch",
			requestBody: `{
				"texts": [
					"solve differential equation",
					"business strategy analysis",
					"chemistry reaction",
					"physics calculation",
					"market research",
					"mathematical modeling",
					"financial planning",
					"scientific experiment"
				]
			}`,
			expectedStatus: http.StatusOK,
		},
		{
			name: "Valid batch with options",
			requestBody: `{
				"texts": ["solve math equation", "write business plan"],
				"options": {"return_probabilities": true}
			}`,
			expectedStatus: http.StatusOK,
		},
		{
			name: "Empty texts array",
			requestBody: `{
				"texts": []
			}`,
			expectedStatus: http.StatusBadRequest,
			expectedError:  "texts array cannot be empty",
		},
		{
			name:           "Missing texts field",
			requestBody:    `{}`,
			expectedStatus: http.StatusBadRequest,
			expectedError:  "texts array cannot be empty",
		},
		{
			name: "Batch too large",
			requestBody: func() string {
				texts := make([]string, 101)
				for i := range texts {
					texts[i] = fmt.Sprintf("test query %d", i)
				}
				data := map[string]interface{}{"texts": texts}
				b, _ := json.Marshal(data)
				return string(b)
			}(),
			expectedStatus: http.StatusBadRequest,
			expectedError:  "batch size cannot exceed 100 texts",
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

func TestCalculateStatistics(t *testing.T) {
	apiServer := &ClassificationAPIServer{}

	tests := []struct {
		name     string
		results  []services.Classification
		expected CategoryClassificationStatistics
	}{
		{
			name: "Mixed categories",
			results: []services.Classification{
				{Category: "math", Confidence: 0.9},
				{Category: "math", Confidence: 0.8},
				{Category: "business", Confidence: 0.6},
				{Category: "science", Confidence: 0.5},
			},
			expected: CategoryClassificationStatistics{
				CategoryDistribution: map[string]int{
					"math":     2,
					"business": 1,
					"science":  1,
				},
				AvgConfidence:      0.7,
				LowConfidenceCount: 2, // 0.6 and 0.5 are below 0.7
			},
		},
		{
			name:    "Empty results",
			results: []services.Classification{},
			expected: CategoryClassificationStatistics{
				CategoryDistribution: map[string]int{},
				AvgConfidence:        0.0,
				LowConfidenceCount:   0,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stats := apiServer.calculateStatistics(tt.results)

			if math.Abs(stats.AvgConfidence-tt.expected.AvgConfidence) > 0.001 {
				t.Errorf("Expected avg confidence %.3f, got %.3f", tt.expected.AvgConfidence, stats.AvgConfidence)
			}

			if stats.LowConfidenceCount != tt.expected.LowConfidenceCount {
				t.Errorf("Expected low confidence count %d, got %d", tt.expected.LowConfidenceCount, stats.LowConfidenceCount)
			}

			for category, expectedCount := range tt.expected.CategoryDistribution {
				if actualCount, exists := stats.CategoryDistribution[category]; !exists || actualCount != expectedCount {
					t.Errorf("Expected category %s count %d, got %d", category, expectedCount, actualCount)
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
						MaxBatchSize         int                                     `yaml:"max_batch_size,omitempty"`
						ConcurrencyThreshold int                                     `yaml:"concurrency_threshold,omitempty"`
						MaxConcurrency       int                                     `yaml:"max_concurrency,omitempty"`
						Metrics              config.BatchClassificationMetricsConfig `yaml:"metrics,omitempty"`
					}{
						MaxBatchSize:         3, // Custom small limit
						ConcurrencyThreshold: 2,
						MaxConcurrency:       4,
						Metrics: config.BatchClassificationMetricsConfig{
							Enabled: true,
						},
					},
				},
			},
			requestBody: `{
				"texts": ["text1", "text2", "text3", "text4"]
			}`,
			expectedStatus: http.StatusBadRequest,
			expectedError:  "batch size cannot exceed 3 texts",
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
			expectedStatus: http.StatusBadRequest,
			expectedError:  "batch size cannot exceed 100 texts", // Default limit
		},
		{
			name: "Valid request within custom limits",
			config: &config.RouterConfig{
				API: config.APIConfig{
					BatchClassification: struct {
						MaxBatchSize         int                                     `yaml:"max_batch_size,omitempty"`
						ConcurrencyThreshold int                                     `yaml:"concurrency_threshold,omitempty"`
						MaxConcurrency       int                                     `yaml:"max_concurrency,omitempty"`
						Metrics              config.BatchClassificationMetricsConfig `yaml:"metrics,omitempty"`
					}{
						MaxBatchSize:         10,
						ConcurrencyThreshold: 3,
						MaxConcurrency:       2,
						Metrics: config.BatchClassificationMetricsConfig{
							Enabled: true,
						},
					},
				},
			},
			requestBody: `{
				"texts": ["text1", "text2"]
			}`,
			expectedStatus: http.StatusOK,
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
