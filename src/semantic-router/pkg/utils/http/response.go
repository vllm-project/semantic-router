package http

import (
	"encoding/json"
	"fmt"
	"log"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/metrics"
)

// CreatePIIViolationResponse creates an HTTP response for PII policy violations
func CreatePIIViolationResponse(model string, deniedPII []string) *ext_proc.ProcessingResponse {
	// Record PII violation metrics
	metrics.RecordPIIViolations(model, deniedPII)

	// Create OpenAI-compatible response format for PII violations
	openAIResponse := map[string]interface{}{
		"id":                 fmt.Sprintf("chatcmpl-pii-violation-%d", time.Now().Unix()),
		"object":             "chat.completion",
		"created":            time.Now().Unix(),
		"model":              model,
		"system_fingerprint": "router_pii_policy",
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": fmt.Sprintf("I cannot process this request as it contains personally identifiable information (%v) that is not allowed for the '%s' model according to the configured privacy policy. Please remove any sensitive information and try again.", deniedPII, model),
				},
				"finish_reason": "content_filter",
			},
		},
		"usage": map[string]interface{}{
			"prompt_tokens":     0,
			"completion_tokens": 0,
			"total_tokens":      0,
		},
	}

	responseBody, err := json.Marshal(openAIResponse)
	if err != nil {
		// Log the error and return a fallback response
		log.Printf("Error marshaling OpenAI response: %v", err)
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
	openAIResponse := map[string]interface{}{
		"id":                 fmt.Sprintf("chatcmpl-jailbreak-blocked-%d", time.Now().Unix()),
		"object":             "chat.completion",
		"created":            time.Now().Unix(),
		"model":              "security-filter",
		"system_fingerprint": "router_prompt_guard",
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": fmt.Sprintf("I cannot process this request as it appears to contain a potential jailbreak attempt (type: %s, confidence: %.3f). Please rephrase your request in a way that complies with our usage policies.", jailbreakType, confidence),
				},
				"finish_reason": "content_filter",
			},
		},
		"usage": map[string]interface{}{
			"prompt_tokens":     0,
			"completion_tokens": 0,
			"total_tokens":      0,
		},
	}

	responseBody, err := json.Marshal(openAIResponse)
	if err != nil {
		// Log the error and return a fallback response
		log.Printf("Error marshaling jailbreak response: %v", err)
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
