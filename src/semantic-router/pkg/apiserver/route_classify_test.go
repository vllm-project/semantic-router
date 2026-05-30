//go:build !windows && cgo

package apiserver

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

const batchClassifierUnavailableMessage = "Batch classification requires unified classifier. Please ensure models are available in ./models/ directory."

type batchClassificationHTTPCase struct {
	name           string
	requestBody    string
	expectedStatus int
	expectedError  string
}

func TestHandleBatchClassificationRejectsInvalidRequests(t *testing.T) {
	apiServer := newBatchClassificationTestServer(nil)
	for _, tt := range invalidBatchClassificationCases() {
		runBatchClassificationHTTPCase(t, apiServer, tt)
	}
}

func TestHandleBatchClassificationReturnsUnavailableWithoutClassifier(t *testing.T) {
	apiServer := newBatchClassificationTestServer(nil)
	for _, tt := range unavailableBatchClassificationCases() {
		runBatchClassificationHTTPCase(t, apiServer, tt)
	}
}

func TestBatchClassificationConfiguration(t *testing.T) {
	for _, tt := range batchClassificationConfigCases() {
		apiServer := newBatchClassificationTestServer(tt.config)
		runBatchClassificationHTTPCase(t, apiServer, tt.batchClassificationHTTPCase)
	}
}

func invalidBatchClassificationCases() []batchClassificationHTTPCase {
	return []batchClassificationHTTPCase{
		{
			name:           "Invalid task_type - jailbreak",
			requestBody:    `{"texts": ["test text"], "task_type": "jailbreak"}`,
			expectedStatus: http.StatusBadRequest,
			expectedError:  "invalid task_type 'jailbreak'. Supported values: [intent pii security all]",
		},
		{
			name:           "Invalid task_type - random",
			requestBody:    `{"texts": ["test text"], "task_type": "invalid_type"}`,
			expectedStatus: http.StatusBadRequest,
			expectedError:  "invalid task_type 'invalid_type'. Supported values: [intent pii security all]",
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
			name:           "Invalid JSON",
			requestBody:    `{"texts": [invalid json`,
			expectedStatus: http.StatusBadRequest,
		},
	}
}

func unavailableBatchClassificationCases() []batchClassificationHTTPCase {
	return []batchClassificationHTTPCase{
		{
			name:           "Valid small batch",
			requestBody:    `{"texts": ["What is machine learning?", "How to invest in stocks?"], "task_type": "intent"}`,
			expectedStatus: http.StatusServiceUnavailable,
			expectedError:  batchClassifierUnavailableMessage,
		},
		{
			name:           "Valid task_type - pii",
			requestBody:    `{"texts": ["test text"], "task_type": "pii"}`,
			expectedStatus: http.StatusServiceUnavailable,
			expectedError:  batchClassifierUnavailableMessage,
		},
		{
			name:           "Valid task_type - security",
			requestBody:    `{"texts": ["test text"], "task_type": "security"}`,
			expectedStatus: http.StatusServiceUnavailable,
			expectedError:  batchClassifierUnavailableMessage,
		},
		{
			name:           "Valid task_type - all",
			requestBody:    `{"texts": ["test text"], "task_type": "all"}`,
			expectedStatus: http.StatusServiceUnavailable,
			expectedError:  batchClassifierUnavailableMessage,
		},
		{
			name:           "Empty task_type defaults to intent",
			requestBody:    `{"texts": ["test text"]}`,
			expectedStatus: http.StatusServiceUnavailable,
			expectedError:  batchClassifierUnavailableMessage,
		},
		{
			name:           "Valid large batch",
			requestBody:    marshalBatchTexts(50),
			expectedStatus: http.StatusServiceUnavailable,
			expectedError:  batchClassifierUnavailableMessage,
		},
		{
			name:           "Valid batch with options",
			requestBody:    `{"texts": ["What is quantum physics?"], "task_type": "intent", "options": {"include_probabilities": true}}`,
			expectedStatus: http.StatusServiceUnavailable,
			expectedError:  batchClassifierUnavailableMessage,
		},
		{
			name:           "Batch too large still checks classifier availability first",
			requestBody:    marshalBatchTexts(101),
			expectedStatus: http.StatusServiceUnavailable,
			expectedError:  batchClassifierUnavailableMessage,
		},
	}
}

type batchClassificationConfigCase struct {
	batchClassificationHTTPCase
	config *config.RouterConfig
}

func batchClassificationConfigCases() []batchClassificationConfigCase {
	return []batchClassificationConfigCase{
		{
			batchClassificationHTTPCase: batchClassificationHTTPCase{
				name:           "Custom max batch size",
				requestBody:    `{"texts": ["text1", "text2", "text3", "text4"]}`,
				expectedStatus: http.StatusServiceUnavailable,
				expectedError:  batchClassifierUnavailableMessage,
			},
			config: batchClassificationMetricsConfig(),
		},
		{
			batchClassificationHTTPCase: batchClassificationHTTPCase{
				name:           "Default config when config is nil",
				requestBody:    marshalBatchTexts(101),
				expectedStatus: http.StatusServiceUnavailable,
				expectedError:  batchClassifierUnavailableMessage,
			},
			config: nil,
		},
		{
			batchClassificationHTTPCase: batchClassificationHTTPCase{
				name:           "Valid request within custom limits",
				requestBody:    `{"texts": ["text1", "text2"]}`,
				expectedStatus: http.StatusServiceUnavailable,
				expectedError:  batchClassifierUnavailableMessage,
			},
			config: batchClassificationMetricsConfig(),
		},
	}
}

func batchClassificationMetricsConfig() *config.RouterConfig {
	return &config.RouterConfig{
		APIServer: config.APIServer{
			API: config.APIConfig{
				BatchClassification: config.BatchClassificationConfig{
					Metrics: config.BatchClassificationMetricsConfig{Enabled: true},
				},
			},
		},
	}
}

func marshalBatchTexts(count int) string {
	texts := make([]string, count)
	for i := range texts {
		texts[i] = fmt.Sprintf("Test text %d", i)
	}
	payload, _ := json.Marshal(map[string]interface{}{"texts": texts, "task_type": "intent"})
	return string(payload)
}

func newBatchClassificationTestServer(cfg *config.RouterConfig) *ClassificationAPIServer {
	return &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config:            cfg,
	}
}

func runBatchClassificationHTTPCase(t *testing.T, apiServer *ClassificationAPIServer, tt batchClassificationHTTPCase) {
	t.Helper()

	t.Run(tt.name, func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/api/v1/classify/batch", bytes.NewBufferString(tt.requestBody))
		req.Header.Set("Content-Type", "application/json")

		rr := httptest.NewRecorder()
		apiServer.handleBatchClassification(rr, req)

		if rr.Code != tt.expectedStatus {
			t.Fatalf("expected status %d, got %d: %s", tt.expectedStatus, rr.Code, rr.Body.String())
		}
		if tt.expectedError != "" {
			assertJSONErrorMessage(t, rr.Body.Bytes(), tt.expectedError)
		}
	})
}

func assertJSONErrorMessage(t *testing.T, body []byte, expected string) {
	t.Helper()

	var errorResponse map[string]interface{}
	if err := json.Unmarshal(body, &errorResponse); err != nil {
		t.Fatalf("failed to unmarshal error response: %v", err)
	}

	errorData, ok := errorResponse["error"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected error response object, got %+v", errorResponse)
	}

	message, ok := errorData["message"].(string)
	if !ok {
		t.Fatalf("expected error message string, got %+v", errorData)
	}
	if message != expected {
		t.Fatalf("expected error message %q, got %q", expected, message)
	}
}
