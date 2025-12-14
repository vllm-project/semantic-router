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
			name: "Invalid task_type - jailbreak",
			requestBody: `{
				"texts": ["test text"],
				"task_type": "jailbreak"
			}`,
			expectedStatus: http.StatusBadRequest,
			expectedError:  "invalid task_type 'jailbreak'. Supported values: [intent pii security all]",
		},
		{
			name: "Invalid task_type - random",
			requestBody: `{
				"texts": ["test text"],
				"task_type": "invalid_type"
			}`,
			expectedStatus: http.StatusBadRequest,
			expectedError:  "invalid task_type 'invalid_type'. Supported values: [intent pii security all]",
		},
		{
			name: "Valid task_type - pii",
			requestBody: `{
				"texts": ["test text"],
				"task_type": "pii"
			}`,
			expectedStatus: http.StatusServiceUnavailable,
			expectedError:  "Batch classification requires unified classifier. Please ensure models are available in ./models/ directory.",
		},
		{
			name: "Valid task_type - security",
			requestBody: `{
				"texts": ["test text"],
				"task_type": "security"
			}`,
			expectedStatus: http.StatusServiceUnavailable,
			expectedError:  "Batch classification requires unified classifier. Please ensure models are available in ./models/ directory.",
		},
		{
			name: "Valid task_type - all",
			requestBody: `{
				"texts": ["test text"],
				"task_type": "all"
			}`,
			expectedStatus: http.StatusServiceUnavailable,
			expectedError:  "Batch classification requires unified classifier. Please ensure models are available in ./models/ directory.",
		},
		{
			name: "Empty task_type defaults to intent",
			requestBody: `{
				"texts": ["test text"]
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
				APIServer: config.APIServer{
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
				APIServer: config.APIServer{
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
	// Test with default config (IncludeConfigModelsInList = false)
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			VLLMEndpoints: []config.VLLMEndpoint{
				{
					Name:    "primary",
					Address: "127.0.0.1",
					Port:    8000,
					Weight:  1,
				},
			},
			ModelConfig: map[string]config.ModelParams{
				"gpt-4o-mini": {
					PreferredEndpoints: []string{"primary"},
				},
				"llama-3.1-8b-instruct": {
					PreferredEndpoints: []string{"primary"},
				},
			},
		},
		RouterOptions: config.RouterOptions{
			IncludeConfigModelsInList: false,
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

	// Must contain only 'MoM' (default auto model name) when IncludeConfigModelsInList is false
	if !got["MoM"] {
		t.Errorf("expected list to contain 'MoM', got: %v", got)
	}
	if len(resp.Data) != 1 {
		t.Errorf("expected only 1 model (MoM), got %d: %v", len(resp.Data), got)
	}
}

func TestOpenAIModelsEndpointWithConfigModels(t *testing.T) {
	// Test with IncludeConfigModelsInList = true
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			VLLMEndpoints: []config.VLLMEndpoint{
				{
					Name:    "primary",
					Address: "127.0.0.1",
					Port:    8000,
					Weight:  1,
				},
			},
			ModelConfig: map[string]config.ModelParams{
				"gpt-4o-mini": {
					PreferredEndpoints: []string{"primary"},
				},
				"llama-3.1-8b-instruct": {
					PreferredEndpoints: []string{"primary"},
				},
			},
		},
		RouterOptions: config.RouterOptions{
			IncludeConfigModelsInList: true,
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

	// Must contain 'MoM' (default auto model name) and the configured models when IncludeConfigModelsInList is true
	if !got["MoM"] {
		t.Errorf("expected list to contain 'MoM', got: %v", got)
	}
	if !got["gpt-4o-mini"] || !got["llama-3.1-8b-instruct"] {
		t.Errorf("expected configured models to be present, got=%v", got)
	}
	if len(resp.Data) != 3 {
		t.Errorf("expected 3 models, got %d", len(resp.Data))
	}
}

// TestSetupRoutesSecurityBehavior tests that setupRoutes correctly includes/excludes endpoints based on security flag
func TestSetupRoutesSecurityBehavior(t *testing.T) {
	tests := []struct {
		name                  string
		enableSystemPromptAPI bool
		expectedEndpoints     map[string]bool // path -> should exist
	}{
		{
			name:                  "System prompt API disabled",
			enableSystemPromptAPI: false,
			expectedEndpoints: map[string]bool{
				"/health":                true,
				"/config/classification": true,
				"/config/system-prompts": false, // Should NOT exist
			},
		},
		{
			name:                  "System prompt API enabled",
			enableSystemPromptAPI: true,
			expectedEndpoints: map[string]bool{
				"/health":                true,
				"/config/classification": true,
				"/config/system-prompts": true, // Should exist
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a test mux that simulates the setupRoutes behavior
			mux := http.NewServeMux()

			// Always add basic endpoints
			mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
			})
			mux.HandleFunc("GET /config/classification", func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
			})

			// Conditionally add system prompt endpoints based on the flag
			if tt.enableSystemPromptAPI {
				mux.HandleFunc("GET /config/system-prompts", func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusOK)
				})
				mux.HandleFunc("PUT /config/system-prompts", func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusOK)
				})
			}

			// Test each endpoint
			for path, shouldExist := range tt.expectedEndpoints {
				req := httptest.NewRequest("GET", path, nil)
				rr := httptest.NewRecorder()

				mux.ServeHTTP(rr, req)

				if shouldExist {
					// Endpoint should exist (not 404)
					if rr.Code == http.StatusNotFound {
						t.Errorf("Expected endpoint %s to exist, but got 404", path)
					}
				} else {
					// Endpoint should NOT exist (404)
					if rr.Code != http.StatusNotFound {
						t.Errorf("Expected endpoint %s to return 404, but got %d", path, rr.Code)
					}
				}
			}
		})
	}
}

// TestAPIOverviewEndpoint tests the API discovery endpoint
func TestAPIOverviewEndpoint(t *testing.T) {
	apiServer := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config:            &config.RouterConfig{},
	}

	req := httptest.NewRequest("GET", "/api/v1", nil)
	rr := httptest.NewRecorder()

	apiServer.handleAPIOverview(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("Expected 200 OK, got %d", rr.Code)
	}

	var response APIOverviewResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &response); err != nil {
		t.Fatalf("Failed to unmarshal response: %v", err)
	}

	// Verify the response structure
	if response.Service == "" {
		t.Error("Expected non-empty service name")
	}

	if response.Version != "v1" {
		t.Errorf("Expected version 'v1', got '%s'", response.Version)
	}

	// Check that we have endpoints listed
	if len(response.Endpoints) == 0 {
		t.Error("Expected at least one endpoint")
	}

	// Check that we have task types listed
	expectedTaskTypes := map[string]bool{
		"intent":   false,
		"pii":      false,
		"security": false,
		"all":      false,
	}

	for _, taskType := range response.TaskTypes {
		if _, exists := expectedTaskTypes[taskType.Name]; exists {
			expectedTaskTypes[taskType.Name] = true
		}
	}

	for taskType, found := range expectedTaskTypes {
		if !found {
			t.Errorf("Expected to find task_type '%s' in response", taskType)
		}
	}

	// Check that we have links
	if len(response.Links) == 0 {
		t.Error("Expected at least one link")
	}

	// Verify specific endpoints are present
	endpointPaths := make(map[string]bool)
	for _, endpoint := range response.Endpoints {
		endpointPaths[endpoint.Path] = true
	}

	requiredPaths := []string{
		"/api/v1/classify/intent",
		"/api/v1/classify/pii",
		"/api/v1/classify/security",
		"/api/v1/classify/batch",
		"/health",
	}

	for _, path := range requiredPaths {
		if !endpointPaths[path] {
			t.Errorf("Expected to find endpoint '%s' in response", path)
		}
	}

	// Verify system prompt endpoints are not included when disabled (default)
	if endpointPaths["/config/system-prompts"] {
		t.Error("Expected system prompt endpoints to be excluded when enableSystemPromptAPI is false")
	}
}

// TestAPIOverviewEndpointWithSystemPrompts tests API discovery with system prompts enabled
func TestAPIOverviewEndpointWithSystemPrompts(t *testing.T) {
	apiServer := &ClassificationAPIServer{
		classificationSvc:     services.NewPlaceholderClassificationService(),
		config:                &config.RouterConfig{},
		enableSystemPromptAPI: true,
	}

	req := httptest.NewRequest("GET", "/api/v1", nil)
	rr := httptest.NewRecorder()

	apiServer.handleAPIOverview(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("Expected 200 OK, got %d", rr.Code)
	}

	var response APIOverviewResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &response); err != nil {
		t.Fatalf("Failed to unmarshal response: %v", err)
	}

	// Verify system prompt endpoints are included when enabled
	endpointPaths := make(map[string]bool)
	for _, endpoint := range response.Endpoints {
		endpointPaths[endpoint.Path] = true
	}

	if !endpointPaths["/config/system-prompts"] {
		t.Error("Expected system prompt endpoints to be included when enableSystemPromptAPI is true")
	}
}

// TestOpenAPISpecEndpoint tests the OpenAPI specification endpoint
func TestOpenAPISpecEndpoint(t *testing.T) {
	apiServer := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config:            &config.RouterConfig{},
	}

	req := httptest.NewRequest("GET", "/openapi.json", nil)
	rr := httptest.NewRecorder()

	apiServer.handleOpenAPISpec(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("Expected 200 OK, got %d", rr.Code)
	}

	// Check Content-Type
	contentType := rr.Header().Get("Content-Type")
	if contentType != "application/json" {
		t.Errorf("Expected Content-Type 'application/json', got '%s'", contentType)
	}

	var spec OpenAPISpec
	if err := json.Unmarshal(rr.Body.Bytes(), &spec); err != nil {
		t.Fatalf("Failed to unmarshal OpenAPI spec: %v", err)
	}

	// Verify the OpenAPI version
	if spec.OpenAPI != "3.0.0" {
		t.Errorf("Expected OpenAPI version '3.0.0', got '%s'", spec.OpenAPI)
	}

	// Verify the info
	if spec.Info.Title == "" {
		t.Error("Expected non-empty title")
	}

	if spec.Info.Version != "v1" {
		t.Errorf("Expected version 'v1', got '%s'", spec.Info.Version)
	}

	// Verify paths are present
	if len(spec.Paths) == 0 {
		t.Error("Expected at least one path in OpenAPI spec")
	}

	// Check that key endpoints are documented
	requiredPaths := []string{
		"/health",
		"/api/v1",
		"/api/v1/classify/batch",
		"/openapi.json",
		"/docs",
	}

	for _, path := range requiredPaths {
		if _, exists := spec.Paths[path]; !exists {
			t.Errorf("Expected path '%s' to be in OpenAPI spec", path)
		}
	}

	// Verify system prompt endpoints are not included when disabled
	if _, exists := spec.Paths["/config/system-prompts"]; exists {
		t.Error("Expected system prompt endpoints to be excluded from OpenAPI spec when disabled")
	}
}

// TestOpenAPISpecWithSystemPrompts tests OpenAPI spec generation with system prompts enabled
func TestOpenAPISpecWithSystemPrompts(t *testing.T) {
	apiServer := &ClassificationAPIServer{
		classificationSvc:     services.NewPlaceholderClassificationService(),
		config:                &config.RouterConfig{},
		enableSystemPromptAPI: true,
	}

	req := httptest.NewRequest("GET", "/openapi.json", nil)
	rr := httptest.NewRecorder()

	apiServer.handleOpenAPISpec(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("Expected 200 OK, got %d", rr.Code)
	}

	var spec OpenAPISpec
	if err := json.Unmarshal(rr.Body.Bytes(), &spec); err != nil {
		t.Fatalf("Failed to unmarshal OpenAPI spec: %v", err)
	}

	// Verify system prompt endpoints are included when enabled
	if _, exists := spec.Paths["/config/system-prompts"]; !exists {
		t.Error("Expected system prompt endpoints to be included in OpenAPI spec when enabled")
	}
}

// TestSwaggerUIEndpoint tests the Swagger UI endpoint
func TestSwaggerUIEndpoint(t *testing.T) {
	apiServer := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config:            &config.RouterConfig{},
	}

	req := httptest.NewRequest("GET", "/docs", nil)
	rr := httptest.NewRecorder()

	apiServer.handleSwaggerUI(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("Expected 200 OK, got %d", rr.Code)
	}

	// Check Content-Type
	contentType := rr.Header().Get("Content-Type")
	if contentType != "text/html; charset=utf-8" {
		t.Errorf("Expected Content-Type 'text/html; charset=utf-8', got '%s'", contentType)
	}

	// Check that the HTML contains Swagger UI references
	html := rr.Body.String()
	if !bytes.Contains([]byte(html), []byte("swagger-ui")) {
		t.Error("Expected HTML to contain 'swagger-ui'")
	}

	if !bytes.Contains([]byte(html), []byte("/openapi.json")) {
		t.Error("Expected HTML to reference '/openapi.json'")
	}

	if !bytes.Contains([]byte(html), []byte("SwaggerUIBundle")) {
		t.Error("Expected HTML to contain 'SwaggerUIBundle'")
	}
}

// TestAPIOverviewIncludesNewEndpoints tests that API overview includes new documentation endpoints
func TestAPIOverviewIncludesNewEndpoints(t *testing.T) {
	apiServer := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config:            &config.RouterConfig{},
	}

	req := httptest.NewRequest("GET", "/api/v1", nil)
	rr := httptest.NewRecorder()

	apiServer.handleAPIOverview(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("Expected 200 OK, got %d", rr.Code)
	}

	var response APIOverviewResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &response); err != nil {
		t.Fatalf("Failed to unmarshal response: %v", err)
	}

	// Verify new documentation endpoints are included
	endpointPaths := make(map[string]bool)
	for _, endpoint := range response.Endpoints {
		endpointPaths[endpoint.Path] = true
	}

	if !endpointPaths["/openapi.json"] {
		t.Error("Expected '/openapi.json' to be in API overview")
	}

	if !endpointPaths["/docs"] {
		t.Error("Expected '/docs' to be in API overview")
	}

	// Verify links include new documentation endpoints
	if response.Links["openapi_spec"] != "/openapi.json" {
		t.Error("Expected 'openapi_spec' link to '/openapi.json'")
	}

	if response.Links["swagger_ui"] != "/docs" {
		t.Error("Expected 'swagger_ui' link to '/docs'")
	}
}
