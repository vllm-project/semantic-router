package extproc

import (
	"encoding/json"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestHandleModelsRequest(t *testing.T) {
	// Create a test router with mock config
	cfg := &config.RouterConfig{
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
		IncludeConfigModelsInList: false, // Default: don't include configured models
	}

	cfgWithModels := &config.RouterConfig{
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
		IncludeConfigModelsInList: true, // Include configured models
	}

	tests := []struct {
		name           string
		config         *config.RouterConfig
		path           string
		expectedModels []string
		expectedCount  int
	}{
		{
			name:           "GET /v1/models - only auto model (default)",
			config:         cfg,
			path:           "/v1/models",
			expectedModels: []string{"MoM"},
			expectedCount:  1,
		},
		{
			name:           "GET /v1/models - with include_config_models_in_list enabled",
			config:         cfgWithModels,
			path:           "/v1/models",
			expectedModels: []string{"MoM", "gpt-4o-mini", "llama-3.1-8b-instruct"},
			expectedCount:  3,
		},
		{
			name:           "GET /v1/models?model=auto - only auto model (default)",
			config:         cfg,
			path:           "/v1/models?model=auto",
			expectedModels: []string{"MoM"},
			expectedCount:  1,
		},
		{
			name:           "GET /v1/models?model=auto - with include_config_models_in_list enabled",
			config:         cfgWithModels,
			path:           "/v1/models?model=auto",
			expectedModels: []string{"MoM", "gpt-4o-mini", "llama-3.1-8b-instruct"},
			expectedCount:  3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			router := &OpenAIRouter{
				Config: tt.config,
			}
			response, err := router.handleModelsRequest(tt.path)
			if err != nil {
				t.Fatalf("handleModelsRequest failed: %v", err)
			}

			// Verify it's an immediate response
			immediateResp := response.GetImmediateResponse()
			if immediateResp == nil {
				t.Fatal("Expected immediate response, got nil")
			}

			// Verify status code is 200 OK
			if immediateResp.Status.Code != typev3.StatusCode_OK {
				t.Errorf("Expected status code OK, got %v", immediateResp.Status.Code)
			}

			// Verify content-type header
			found := false
			for _, header := range immediateResp.Headers.SetHeaders {
				if header.Header.Key == "content-type" {
					if string(header.Header.RawValue) != "application/json" {
						t.Errorf("Expected content-type application/json, got %s", string(header.Header.RawValue))
					}
					found = true
					break
				}
			}
			if !found {
				t.Error("Expected content-type header not found")
			}

			// Parse response body
			var modelList OpenAIModelList
			if err := json.Unmarshal(immediateResp.Body, &modelList); err != nil {
				t.Fatalf("Failed to parse response body: %v", err)
			}

			// Verify response structure
			if modelList.Object != "list" {
				t.Errorf("Expected object 'list', got %s", modelList.Object)
			}

			if len(modelList.Data) != tt.expectedCount {
				t.Errorf("Expected %d models, got %d", tt.expectedCount, len(modelList.Data))
			}

			// Verify expected models are present
			modelMap := make(map[string]bool)
			for _, model := range modelList.Data {
				modelMap[model.ID] = true

				// Verify model structure
				if model.Object != "model" {
					t.Errorf("Expected model object 'model', got %s", model.Object)
				}
				if model.Created == 0 {
					t.Error("Expected non-zero created timestamp")
				}
				if model.OwnedBy != "vllm-semantic-router" {
					t.Errorf("Expected model owned_by 'vllm-semantic-router', got %s", model.OwnedBy)
				}
			}

			for _, expectedModel := range tt.expectedModels {
				if !modelMap[expectedModel] {
					t.Errorf("Expected model %s not found in response", expectedModel)
				}
			}
		})
	}
}

func TestHandleRequestHeadersWithModelsEndpoint(t *testing.T) {
	// Create a test router
	cfg := &config.RouterConfig{
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
		},
	}

	router := &OpenAIRouter{
		Config: cfg,
	}

	tests := []struct {
		name            string
		method          string
		path            string
		expectImmediate bool
	}{
		{
			name:            "GET /v1/models - should return immediate response",
			method:          "GET",
			path:            "/v1/models",
			expectImmediate: true,
		},
		{
			name:            "GET /v1/models?model=auto - should return immediate response",
			method:          "GET",
			path:            "/v1/models?model=auto",
			expectImmediate: true,
		},
		{
			name:            "POST /v1/chat/completions - should continue processing",
			method:          "POST",
			path:            "/v1/chat/completions",
			expectImmediate: false,
		},
		{
			name:            "POST /v1/models - should continue processing",
			method:          "POST",
			path:            "/v1/models",
			expectImmediate: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create request headers
			requestHeaders := &ext_proc.ProcessingRequest_RequestHeaders{
				RequestHeaders: &ext_proc.HttpHeaders{
					Headers: &core.HeaderMap{
						Headers: []*core.HeaderValue{
							{
								Key:   ":method",
								Value: tt.method,
							},
							{
								Key:   ":path",
								Value: tt.path,
							},
							{
								Key:   "content-type",
								Value: "application/json",
							},
						},
					},
				},
			}

			ctx := &RequestContext{
				Headers: make(map[string]string),
			}

			response, err := router.handleRequestHeaders(requestHeaders, ctx)
			if err != nil {
				t.Fatalf("handleRequestHeaders failed: %v", err)
			}

			if tt.expectImmediate {
				// Should return immediate response
				if response.GetImmediateResponse() == nil {
					t.Error("Expected immediate response for /v1/models endpoint")
				}
			} else {
				// Should return continue response
				if response.GetRequestHeaders() == nil {
					t.Error("Expected request headers response for non-models endpoint")
				}
				if response.GetRequestHeaders().Response.Status != ext_proc.CommonResponse_CONTINUE {
					t.Error("Expected CONTINUE status for non-models endpoint")
				}
			}
		})
	}
}
