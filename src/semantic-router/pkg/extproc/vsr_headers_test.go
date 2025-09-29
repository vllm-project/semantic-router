package extproc

import (
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
)

func TestVSRHeadersAddedOnSuccessfulNonCachedResponse(t *testing.T) {
	// Create a mock router
	router := &OpenAIRouter{}

	// Create request context with VSR decision information
	ctx := &RequestContext{
		VSRSelectedCategory: "math",
		VSRReasoningMode:    "on",
		VSRSelectedModel:    "deepseek-v31",
		VSRCacheHit:         false, // Not a cache hit
	}

	// Create response headers with successful status (200)
	responseHeaders := &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":status", Value: "200"},
					{Key: "content-type", Value: "application/json"},
				},
			},
		},
	}

	// Call handleResponseHeaders
	response, err := router.handleResponseHeaders(responseHeaders, ctx)

	// Verify no error occurred
	assert.NoError(t, err)
	assert.NotNil(t, response)

	// Verify response structure
	assert.NotNil(t, response.GetResponseHeaders())
	assert.NotNil(t, response.GetResponseHeaders().GetResponse())

	// Verify VSR headers were added
	headerMutation := response.GetResponseHeaders().GetResponse().GetHeaderMutation()
	assert.NotNil(t, headerMutation, "HeaderMutation should not be nil for successful non-cached response")

	setHeaders := headerMutation.GetSetHeaders()
	assert.Len(t, setHeaders, 3, "Should have 3 VSR headers")

	// Verify each header
	headerMap := make(map[string]string)
	for _, header := range setHeaders {
		headerMap[header.Header.Key] = string(header.Header.RawValue)
	}

	assert.Equal(t, "math", headerMap["x-vsr-selected-category"])
	assert.Equal(t, "on", headerMap["x-vsr-selected-reasoning"])
	assert.Equal(t, "deepseek-v31", headerMap["x-vsr-selected-model"])
}

func TestVSRHeadersNotAddedOnCacheHit(t *testing.T) {
	// Create a mock router
	router := &OpenAIRouter{}

	// Create request context with cache hit
	ctx := &RequestContext{
		VSRSelectedCategory: "math",
		VSRReasoningMode:    "on",
		VSRSelectedModel:    "deepseek-v31",
		VSRCacheHit:         true, // Cache hit - headers should not be added
	}

	// Create response headers with successful status (200)
	responseHeaders := &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":status", Value: "200"},
					{Key: "content-type", Value: "application/json"},
				},
			},
		},
	}

	// Call handleResponseHeaders
	response, err := router.handleResponseHeaders(responseHeaders, ctx)

	// Verify no error occurred
	assert.NoError(t, err)
	assert.NotNil(t, response)

	// Verify VSR headers were NOT added due to cache hit
	headerMutation := response.GetResponseHeaders().GetResponse().GetHeaderMutation()
	assert.Nil(t, headerMutation, "HeaderMutation should be nil for cache hit")
}

func TestVSRHeadersNotAddedOnErrorResponse(t *testing.T) {
	// Create a mock router
	router := &OpenAIRouter{}

	// Create request context with VSR decision information
	ctx := &RequestContext{
		VSRSelectedCategory: "math",
		VSRReasoningMode:    "on",
		VSRSelectedModel:    "deepseek-v31",
		VSRCacheHit:         false, // Not a cache hit
	}

	// Create response headers with error status (500)
	responseHeaders := &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":status", Value: "500"},
					{Key: "content-type", Value: "application/json"},
				},
			},
		},
	}

	// Call handleResponseHeaders
	response, err := router.handleResponseHeaders(responseHeaders, ctx)

	// Verify no error occurred
	assert.NoError(t, err)
	assert.NotNil(t, response)

	// Verify VSR headers were NOT added due to error status
	headerMutation := response.GetResponseHeaders().GetResponse().GetHeaderMutation()
	assert.Nil(t, headerMutation, "HeaderMutation should be nil for error response")
}

func TestVSRHeadersPartialInformation(t *testing.T) {
	// Create a mock router
	router := &OpenAIRouter{}

	// Create request context with partial VSR information
	ctx := &RequestContext{
		VSRSelectedCategory: "math",
		VSRReasoningMode:    "", // Empty reasoning mode
		VSRSelectedModel:    "deepseek-v31",
		VSRCacheHit:         false,
	}

	// Create response headers with successful status (200)
	responseHeaders := &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{
					{Key: ":status", Value: "200"},
					{Key: "content-type", Value: "application/json"},
				},
			},
		},
	}

	// Call handleResponseHeaders
	response, err := router.handleResponseHeaders(responseHeaders, ctx)

	// Verify no error occurred
	assert.NoError(t, err)
	assert.NotNil(t, response)

	// Verify only non-empty headers were added
	headerMutation := response.GetResponseHeaders().GetResponse().GetHeaderMutation()
	assert.NotNil(t, headerMutation)

	setHeaders := headerMutation.GetSetHeaders()
	assert.Len(t, setHeaders, 2, "Should have 2 VSR headers (excluding empty reasoning mode)")

	// Verify each header
	headerMap := make(map[string]string)
	for _, header := range setHeaders {
		headerMap[header.Header.Key] = string(header.Header.RawValue)
	}

	assert.Equal(t, "math", headerMap["x-vsr-selected-category"])
	assert.Equal(t, "deepseek-v31", headerMap["x-vsr-selected-model"])
	assert.NotContains(t, headerMap, "x-vsr-selected-reasoning", "Empty reasoning mode should not be added")
}
