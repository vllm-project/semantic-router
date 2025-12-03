package http

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// CreatePIIViolationResponse creates an HTTP response for PII policy violations
func CreatePIIViolationResponse(model string, deniedPII []string, isStreaming bool, decisionName string, category string) *ext_proc.ProcessingResponse {
	// Record PII violation metrics
	metrics.RecordPIIViolations(model, deniedPII)

	// Join denied PII types for header
	deniedPIIStr := strings.Join(deniedPII, ",")

	// Create OpenAI-compatible response format for PII violations
	unixTimeStep := time.Now().Unix()
	var responseBody []byte
	var contentType string

	if isStreaming {
		// For streaming responses, use SSE format
		contentType = "text/event-stream"

		// Create streaming chunk with security violation message
		streamChunk := map[string]interface{}{
			"id":      fmt.Sprintf("chatcmpl-pii-violation-%d", unixTimeStep),
			"object":  "chat.completion.chunk",
			"created": unixTimeStep,
			"model":   model,
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]interface{}{
						"role":    "assistant",
						"content": fmt.Sprintf("I cannot process this request as it contains personally identifiable information (%v) that is not allowed for the '%s' model according to the configured privacy policy. Please remove any sensitive information and try again.", deniedPII, model),
					},
					"finish_reason": "content_filter",
				},
			},
		}

		chunkJSON, err := json.Marshal(streamChunk)
		if err != nil {
			logging.Errorf("Error marshaling streaming PII response: %v", err)
			responseBody = []byte("data: {\"error\": \"Failed to generate response\"}\n\ndata: [DONE]\n\n")
		} else {
			responseBody = []byte(fmt.Sprintf("data: %s\n\ndata: [DONE]\n\n", chunkJSON))
		}
	} else {
		// For non-streaming responses, use regular JSON format
		contentType = "application/json"

		openAIResponse := openai.ChatCompletion{
			ID:      fmt.Sprintf("chatcmpl-pii-violation-%d", unixTimeStep),
			Object:  "chat.completion",
			Created: unixTimeStep,
			Model:   model,
			Choices: []openai.ChatCompletionChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role:    "assistant",
						Content: fmt.Sprintf("I cannot process this request as it contains personally identifiable information (%v) that is not allowed for the '%s' model according to the configured privacy policy. Please remove any sensitive information and try again.", deniedPII, model),
					},
					FinishReason: "content_filter",
				},
			},
			Usage: openai.CompletionUsage{
				PromptTokens:     0,
				CompletionTokens: 0,
				TotalTokens:      0,
			},
		}

		var err error
		responseBody, err = json.Marshal(openAIResponse)
		if err != nil {
			// Log the error and return a fallback response
			logging.Errorf("Error marshaling OpenAI response: %v", err)
			responseBody = []byte(`{"error": "Failed to generate response"}`)
		}
	}

	immediateResponse := &ext_proc.ImmediateResponse{
		Status: &typev3.HttpStatus{
			Code: typev3.StatusCode_OK, // Return 200 OK to match OpenAI API behavior
		},
		Headers: &ext_proc.HeaderMutation{
			SetHeaders: []*core.HeaderValueOption{
				{
					Header: &core.HeaderValue{
						Key:      "content-type",
						RawValue: []byte(contentType),
					},
				},
				{
					Header: &core.HeaderValue{
						Key:      headers.VSRPIIViolation,
						RawValue: []byte("true"),
					},
				},
				{
					Header: &core.HeaderValue{
						Key:      headers.VSRPIITypes,
						RawValue: []byte(deniedPIIStr),
					},
				},
				{
					Header: &core.HeaderValue{
						Key:      headers.VSRSelectedDecision,
						RawValue: []byte(decisionName),
					},
				},
				{
					Header: &core.HeaderValue{
						Key:      headers.VSRSelectedCategory,
						RawValue: []byte(category),
					},
				},
			},
		},
		Body: responseBody,
	}

	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: immediateResponse,
		},
	}
}

// CreateJailbreakViolationResponse creates an HTTP response for jailbreak detection violations
func CreateJailbreakViolationResponse(jailbreakType string, confidence float32, isStreaming bool) *ext_proc.ProcessingResponse {
	// Create OpenAI-compatible response format for jailbreak violations
	unixTimeStep := time.Now().Unix()
	var responseBody []byte
	var contentType string

	if isStreaming {
		// For streaming responses, use SSE format
		contentType = "text/event-stream"

		// Create streaming chunk with security violation message
		streamChunk := map[string]interface{}{
			"id":      fmt.Sprintf("chatcmpl-jailbreak-blocked-%d", unixTimeStep),
			"object":  "chat.completion.chunk",
			"created": unixTimeStep,
			"model":   "security-filter",
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]interface{}{
						"role":    "assistant",
						"content": fmt.Sprintf("I cannot process this request as it appears to contain a potential jailbreak attempt (type: %s, confidence: %.3f). Please rephrase your request in a way that complies with our usage policies.", jailbreakType, confidence),
					},
					"finish_reason": "content_filter",
				},
			},
		}

		chunkJSON, err := json.Marshal(streamChunk)
		if err != nil {
			logging.Errorf("Error marshaling streaming jailbreak response: %v", err)
			responseBody = []byte("data: {\"error\": \"Failed to generate response\"}\n\ndata: [DONE]\n\n")
		} else {
			responseBody = []byte(fmt.Sprintf("data: %s\n\ndata: [DONE]\n\n", chunkJSON))
		}
	} else {
		// For non-streaming responses, use regular JSON format
		contentType = "application/json"

		openAIResponse := openai.ChatCompletion{
			ID:      fmt.Sprintf("chatcmpl-jailbreak-blocked-%d", unixTimeStep),
			Object:  "chat.completion",
			Created: unixTimeStep,
			Model:   "security-filter",
			Choices: []openai.ChatCompletionChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role:    "assistant",
						Content: fmt.Sprintf("I cannot process this request as it appears to contain a potential jailbreak attempt (type: %s, confidence: %.3f). Please rephrase your request in a way that complies with our usage policies.", jailbreakType, confidence),
					},
					FinishReason: "content_filter",
				},
			},
			Usage: openai.CompletionUsage{
				PromptTokens:     0,
				CompletionTokens: 0,
				TotalTokens:      0,
			},
		}

		var err error
		responseBody, err = json.Marshal(openAIResponse)
		if err != nil {
			// Log the error and return a fallback response
			logging.Errorf("Error marshaling jailbreak response: %v", err)
			responseBody = []byte(`{"error": "Failed to generate response"}`)
		}
	}

	immediateResponse := &ext_proc.ImmediateResponse{
		Status: &typev3.HttpStatus{
			Code: typev3.StatusCode_OK, // Return 200 OK to match OpenAI API behavior
		},
		Headers: &ext_proc.HeaderMutation{
			SetHeaders: []*core.HeaderValueOption{
				{
					Header: &core.HeaderValue{
						Key:      "content-type",
						RawValue: []byte(contentType),
					},
				},
				{
					Header: &core.HeaderValue{
						Key:      headers.VSRJailbreakBlocked,
						RawValue: []byte("true"),
					},
				},
				{
					Header: &core.HeaderValue{
						Key:      headers.VSRJailbreakType,
						RawValue: []byte(jailbreakType),
					},
				},
				{
					Header: &core.HeaderValue{
						Key:      headers.VSRJailbreakConfidence,
						RawValue: []byte(fmt.Sprintf("%.3f", confidence)),
					},
				},
			},
		},
		Body: responseBody,
	}

	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: immediateResponse,
		},
	}
}

// CreateCacheHitResponse creates an immediate response from cache
func CreateCacheHitResponse(cachedResponse []byte, isStreaming bool, category string, decisionName string) *ext_proc.ProcessingResponse {
	var responseBody []byte
	var contentType string

	if isStreaming {
		// For streaming responses, convert cached JSON to SSE format
		contentType = "text/event-stream"

		// Parse the cached JSON response
		var cachedCompletion openai.ChatCompletion
		if err := json.Unmarshal(cachedResponse, &cachedCompletion); err != nil {
			logging.Errorf("Error parsing cached response for streaming conversion: %v", err)
			responseBody = []byte("data: {\"error\": \"Failed to convert cached response\"}\n\ndata: [DONE]\n\n")
		} else {
			// Convert chat.completion to chat.completion.chunk format
			streamChunk := map[string]interface{}{
				"id":      cachedCompletion.ID,
				"object":  "chat.completion.chunk",
				"created": cachedCompletion.Created,
				"model":   cachedCompletion.Model,
				"choices": []map[string]interface{}{},
			}

			// Convert choices from message format to delta format
			for _, choice := range cachedCompletion.Choices {
				streamChoice := map[string]interface{}{
					"index": choice.Index,
					"delta": map[string]interface{}{
						"role":    choice.Message.Role,
						"content": choice.Message.Content,
					},
					"finish_reason": choice.FinishReason,
				}
				streamChunk["choices"] = append(streamChunk["choices"].([]map[string]interface{}), streamChoice)
			}

			chunkJSON, err := json.Marshal(streamChunk)
			if err != nil {
				logging.Errorf("Error marshaling streaming cache response: %v", err)
				responseBody = []byte("data: {\"error\": \"Failed to generate response\"}\n\ndata: [DONE]\n\n")
			} else {
				responseBody = []byte(fmt.Sprintf("data: %s\n\ndata: [DONE]\n\n", chunkJSON))
			}
		}
	} else {
		// For non-streaming responses, use cached JSON directly
		contentType = "application/json"
		responseBody = cachedResponse
	}

	immediateResponse := &ext_proc.ImmediateResponse{
		Status: &typev3.HttpStatus{
			Code: typev3.StatusCode_OK,
		},
		Headers: &ext_proc.HeaderMutation{
			SetHeaders: []*core.HeaderValueOption{
				{
					Header: &core.HeaderValue{
						Key:      "content-type",
						RawValue: []byte(contentType),
					},
				},
				{
					Header: &core.HeaderValue{
						Key:      headers.VSRCacheHit,
						RawValue: []byte("true"),
					},
				},
				{
					Header: &core.HeaderValue{
						Key:      headers.VSRSelectedCategory,
						RawValue: []byte(category),
					},
				},
				{
					Header: &core.HeaderValue{
						Key:      headers.VSRSelectedDecision,
						RawValue: []byte(decisionName),
					},
				},
			},
		},
		Body: responseBody,
	}

	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: immediateResponse,
		},
	}
}
