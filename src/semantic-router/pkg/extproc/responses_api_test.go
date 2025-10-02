package extproc

import (
	"encoding/json"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/classification"
)

func TestHandleRequestHeaders_ResponsesAPI_POST(t *testing.T) {
	// Create a test router with mock config
	cfg := &config.RouterConfig{
		VLLMEndpoints: []config.VLLMEndpoint{
			{
				Name:    "primary",
				Address: "127.0.0.1",
				Port:    8000,
				Models:  []string{"gpt-4o", "o1"},
				Weight:  1,
			},
		},
	}

	router := &OpenAIRouter{
		Config: cfg,
	}

	// Test POST /v1/responses request
	ctx := &RequestContext{
		Headers: make(map[string]string),
	}

	headers := []*core.HeaderValue{
		{Key: ":method", Value: "POST"},
		{Key: ":path", Value: "/v1/responses"},
		{Key: "content-type", Value: "application/json"},
	}

	requestHeaders := &ext_proc.ProcessingRequest_RequestHeaders{
		RequestHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: headers,
			},
		},
	}

	response, err := router.handleRequestHeaders(requestHeaders, ctx)

	assert.NoError(t, err)
	assert.NotNil(t, response)
	assert.True(t, ctx.IsResponsesAPI, "Should detect Responses API request")
	assert.Equal(t, "POST", ctx.Headers[":method"])
	assert.Equal(t, "/v1/responses", ctx.Headers[":path"])
}

func TestHandleRequestHeaders_ResponsesAPI_GET(t *testing.T) {
	// Create a test router with mock config
	cfg := &config.RouterConfig{
		VLLMEndpoints: []config.VLLMEndpoint{
			{
				Name:    "primary",
				Address: "127.0.0.1",
				Port:    8000,
				Models:  []string{"gpt-4o", "o1"},
				Weight:  1,
			},
		},
	}

	router := &OpenAIRouter{
		Config: cfg,
	}

	// Test GET /v1/responses/{id} request - should pass through
	ctx := &RequestContext{
		Headers: make(map[string]string),
	}

	headers := []*core.HeaderValue{
		{Key: ":method", Value: "GET"},
		{Key: ":path", Value: "/v1/responses/resp_12345"},
		{Key: "content-type", Value: "application/json"},
	}

	requestHeaders := &ext_proc.ProcessingRequest_RequestHeaders{
		RequestHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: headers,
			},
		},
	}

	response, err := router.handleRequestHeaders(requestHeaders, ctx)

	assert.NoError(t, err)
	assert.NotNil(t, response)
	assert.True(t, ctx.IsResponsesAPI, "Should detect Responses API request")
	// GET request should return immediate CONTINUE without further processing
	assert.NotNil(t, response.GetRequestHeaders())
}

func TestParseOpenAIResponsesRequest(t *testing.T) {
	tests := []struct {
		name        string
		requestBody string
		expectError bool
		checkModel  string
	}{
		{
			name: "Valid Responses API request with string input",
			requestBody: `{
				"model": "gpt-4o",
				"input": "What is 2+2?"
			}`,
			expectError: false,
			checkModel:  "gpt-4o",
		},
		{
			name: "Valid Responses API request with message input",
			requestBody: `{
				"model": "o1",
				"input": [
					{"role": "user", "content": "Solve x^2 + 5x + 6 = 0"}
				]
			}`,
			expectError: false,
			checkModel:  "o1",
		},
		{
			name: "Valid Responses API request with previous_response_id",
			requestBody: `{
				"model": "gpt-4o",
				"input": "Continue from where we left off",
				"previous_response_id": "resp_12345"
			}`,
			expectError: false,
			checkModel:  "gpt-4o",
		},
		{
			name:        "Invalid JSON",
			requestBody: `{invalid json`,
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req, err := parseOpenAIResponsesRequest([]byte(tt.requestBody))

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, req)
				if tt.checkModel != "" {
					assert.Equal(t, tt.checkModel, string(req.Model))
				}
			}
		})
	}
}

func TestExtractContentFromResponsesInput_StringInput(t *testing.T) {
	requestBody := `{
		"model": "gpt-4o",
		"input": "What is the meaning of life?"
	}`

	req, err := parseOpenAIResponsesRequest([]byte(requestBody))
	assert.NoError(t, err)

	userContent, nonUserMessages := extractContentFromResponsesInput(req)

	assert.Equal(t, "What is the meaning of life?", userContent)
	assert.Empty(t, nonUserMessages)
}

func TestExtractContentFromResponsesInput_MessageArray(t *testing.T) {
	requestBody := `{
		"model": "gpt-4o",
		"input": [
			{"role": "system", "content": "You are a helpful assistant"},
			{"role": "user", "content": "What is 2+2?"},
			{"role": "assistant", "content": "2+2 equals 4"},
			{"role": "user", "content": "And what is 3+3?"}
		]
	}`

	req, err := parseOpenAIResponsesRequest([]byte(requestBody))
	assert.NoError(t, err)

	userContent, nonUserMessages := extractContentFromResponsesInput(req)

	// Should extract the last user message as userContent
	assert.Equal(t, "And what is 3+3?", userContent)
	// Should have system and assistant messages
	assert.Contains(t, nonUserMessages, "You are a helpful assistant")
	assert.Contains(t, nonUserMessages, "2+2 equals 4")
}

func TestExtractContentFromResponsesInput_ComplexContent(t *testing.T) {
	requestBody := `{
		"model": "gpt-4o",
		"input": [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": "What do you see in this image?"},
					{"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
				]
			}
		]
	}`

	req, err := parseOpenAIResponsesRequest([]byte(requestBody))
	assert.NoError(t, err)

	userContent, nonUserMessages := extractContentFromResponsesInput(req)

	// Should extract text from content array
	assert.Contains(t, userContent, "What do you see in this image?")
	assert.Empty(t, nonUserMessages)
}

func TestHandleResponsesAPIRequest_AutoModelSelection(t *testing.T) {
	// Create a more complete test router
	cfg := &config.RouterConfig{
		VLLMEndpoints: []config.VLLMEndpoint{
			{
				Name:    "primary",
				Address: "127.0.0.1",
				Port:    8000,
				Models:  []string{"gpt-4o-mini", "deepseek-v3"},
				Weight:  1,
			},
		},
		DefaultModel: "gpt-4o-mini",
		Categories: []config.Category{
			{
				Name:        "math",
				Description: "Mathematical calculations and equations",
				Models:      []string{"deepseek-v3"},
			},
			{
				Name:        "general",
				Description: "General conversation and questions",
				Models:      []string{"gpt-4o-mini"},
			},
		},
	}

	// Create a mock classifier
	classifier := &classification.Classifier{
		CategoryMapping: &classification.CategoryMapping{
			ID2Label: map[int]string{
				0: "math",
				1: "general",
			},
		},
	}

	// Create a minimal cache backend
	cacheBackend, _ := cache.NewCacheBackend(cache.CacheConfig{
		BackendType: cache.InMemoryCacheType,
		Enabled:     false,
	})

	router := &OpenAIRouter{
		Config:     cfg,
		Classifier: classifier,
		Cache:      cacheBackend,
	}

	// Test with auto model selection
	requestBody := []byte(`{
		"model": "auto",
		"input": "What is the derivative of x^2?"
	}`)

	ctx := &RequestContext{
		Headers:             make(map[string]string),
		IsResponsesAPI:      true,
		OriginalRequestBody: requestBody,
		RequestID:           "test-request-123",
	}

	requestBodyMsg := &ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{
			Body: requestBody,
		},
	}

	// Note: This test will work partially - full routing requires more setup
	// but we can at least verify parsing and basic flow
	response, err := router.handleResponsesAPIRequest(requestBodyMsg, ctx, false)

	// The test should not fail catastrophically
	assert.NotNil(t, response)
	// Error is expected due to incomplete classifier setup, but structure should be valid
	if err == nil {
		assert.NotNil(t, response.GetRequestBody())
	}
}

func TestSerializeOpenAIResponsesRequest(t *testing.T) {
	requestBody := `{
		"model": "gpt-4o",
		"input": "Test input",
		"temperature": 0.7
	}`

	req, err := parseOpenAIResponsesRequest([]byte(requestBody))
	assert.NoError(t, err)

	// Serialize back
	serialized, err := serializeOpenAIResponsesRequest(req)
	assert.NoError(t, err)
	assert.NotEmpty(t, serialized)

	// Verify it's valid JSON
	var result map[string]interface{}
	err = json.Unmarshal(serialized, &result)
	assert.NoError(t, err)
	assert.Equal(t, "gpt-4o", result["model"])
}

func TestSerializeOpenAIResponsesRequestWithStream(t *testing.T) {
	requestBody := `{
		"model": "gpt-4o",
		"input": "Test input"
	}`

	req, err := parseOpenAIResponsesRequest([]byte(requestBody))
	assert.NoError(t, err)

	// Serialize with stream parameter
	serialized, err := serializeOpenAIResponsesRequestWithStream(req, true)
	assert.NoError(t, err)
	assert.NotEmpty(t, serialized)

	// Verify stream parameter is present
	var result map[string]interface{}
	err = json.Unmarshal(serialized, &result)
	assert.NoError(t, err)
	assert.Equal(t, true, result["stream"])
}

func TestHandleRequestHeaders_ResponsesAPI_ExcludeInputItems(t *testing.T) {
	// Create a test router
	cfg := &config.RouterConfig{
		VLLMEndpoints: []config.VLLMEndpoint{
			{
				Name:    "primary",
				Address: "127.0.0.1",
				Port:    8000,
				Models:  []string{"gpt-4o"},
				Weight:  1,
			},
		},
	}

	router := &OpenAIRouter{
		Config: cfg,
	}

	// Test that input_items endpoints are not treated as Responses API
	tests := []struct {
		name            string
		path            string
		shouldBeRespAPI bool
	}{
		{
			name:            "POST /v1/responses - should be Responses API",
			path:            "/v1/responses",
			shouldBeRespAPI: true,
		},
		{
			name:            "GET /v1/responses/{id} - should be Responses API",
			path:            "/v1/responses/resp_123",
			shouldBeRespAPI: true,
		},
		{
			name:            "GET /v1/responses/{id}/input_items - should NOT be Responses API",
			path:            "/v1/responses/resp_123/input_items",
			shouldBeRespAPI: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := &RequestContext{
				Headers: make(map[string]string),
			}

			method := "POST"
			if tt.path != "/v1/responses" {
				method = "GET"
			}

			headers := []*core.HeaderValue{
				{Key: ":method", Value: method},
				{Key: ":path", Value: tt.path},
			}

			requestHeaders := &ext_proc.ProcessingRequest_RequestHeaders{
				RequestHeaders: &ext_proc.HttpHeaders{
					Headers: &core.HeaderMap{
						Headers: headers,
					},
				},
			}

			router.handleRequestHeaders(requestHeaders, ctx)

			assert.Equal(t, tt.shouldBeRespAPI, ctx.IsResponsesAPI)
		})
	}
}
