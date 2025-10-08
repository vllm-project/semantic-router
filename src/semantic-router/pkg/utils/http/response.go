// Copyright 2025 The vLLM Semantic Router Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
func CreatePIIViolationResponse(model string, deniedPII []string, isStreaming bool) *ext_proc.ProcessingResponse {
	// Record PII violation metrics
	metrics.RecordPIIViolations(model, deniedPII)

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
			observability.Errorf("Error marshaling streaming PII response: %v", err)
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
			observability.Errorf("Error marshaling OpenAI response: %v", err)
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
			observability.Errorf("Error marshaling streaming jailbreak response: %v", err)
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
			observability.Errorf("Error marshaling jailbreak response: %v", err)
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
						Key:      "x-jailbreak-blocked",
						RawValue: []byte("true"),
					},
				},
				{
					Header: &core.HeaderValue{
						Key:      "x-jailbreak-type",
						RawValue: []byte(jailbreakType),
					},
				},
				{
					Header: &core.HeaderValue{
						Key:      "x-jailbreak-confidence",
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
						Key:      "x-vsr-cache-hit",
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
