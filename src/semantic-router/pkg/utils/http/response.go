package http

import (
	"encoding/json"
	"fmt"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/openai/openai-go"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
)

// CreatePIIViolationResponse creates an HTTP response for PII policy violations
func CreatePIIViolationResponse(model string, deniedPII []string) *ext_proc.ProcessingResponse {
	// Record PII violation metrics
	metrics.RecordPIIViolations(model, deniedPII)

	// Create OpenAI-compatible response format for PII violations
	unixTimeStep := time.Now().Unix()
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

	responseBody, err := json.Marshal(openAIResponse)
	if err != nil {
		// Log the error and return a fallback response
		observability.Errorf("Error marshaling OpenAI response: %v", err)
		responseBody = []byte(`{"error": "Failed to generate response"}`)
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
						RawValue: []byte("application/json"),
					},
				},
				{
					Header: &core.HeaderValue{
						Key:      "x-pii-violation",
						RawValue: []byte("true"),
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
func CreateJailbreakViolationResponse(jailbreakType string, confidence float32) *ext_proc.ProcessingResponse {
	// Create OpenAI-compatible response format for jailbreak violations
	openAIResponse := openai.ChatCompletion{
		ID:      fmt.Sprintf("chatcmpl-jailbreak-blocked-%d", time.Now().Unix()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
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

	responseBody, err := json.Marshal(openAIResponse)
	if err != nil {
		// Log the error and return a fallback response
		observability.Errorf("Error marshaling jailbreak response: %v", err)
		responseBody = []byte(`{"error": "Failed to generate response"}`)
	}

	immediateResponse := &ext_proc.ImmediateResponse{
		Status: &typev3.HttpStatus{
			Code: typev3.StatusCode_OK, // Return 200 OK to match OpenAI API behavior
		},
		Headers: &ext_proc.HeaderMutation{
			SetHeaders: []*core.HeaderValueOption{
				{
					Header: &core.HeaderValue{
						Key:   "content-type",
						Value: "application/json",
					},
				},
				{
					Header: &core.HeaderValue{
						Key:   "x-jailbreak-blocked",
						Value: "true",
					},
				},
				{
					Header: &core.HeaderValue{
						Key:   "x-jailbreak-type",
						Value: jailbreakType,
					},
				},
				{
					Header: &core.HeaderValue{
						Key:   "x-jailbreak-confidence",
						Value: fmt.Sprintf("%.3f", confidence),
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
func CreateCacheHitResponse(cachedResponse []byte) *ext_proc.ProcessingResponse {
	immediateResponse := &ext_proc.ImmediateResponse{
		Status: &typev3.HttpStatus{
			Code: typev3.StatusCode_OK,
		},
		Headers: &ext_proc.HeaderMutation{
			SetHeaders: []*core.HeaderValueOption{
				{
					Header: &core.HeaderValue{
						Key:      "content-type",
						RawValue: []byte("application/json"),
					},
				},
				{
					Header: &core.HeaderValue{
						Key:      "x-cache-hit",
						RawValue: []byte("true"),
					},
				},
			},
		},
		Body: cachedResponse,
	}

	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: immediateResponse,
		},
	}
}
