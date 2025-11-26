package http

import (
	"bytes"
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
func CreatePIIViolationResponse(model string, deniedPII []string, isStreaming bool, decisionName string, category string, matchedKeywords []string) *ext_proc.ProcessingResponse {
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

	// Add matched keywords header if provided
	if len(matchedKeywords) > 0 {
		immediateResponse.Headers.SetHeaders = append(immediateResponse.Headers.SetHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.VSRMatchedKeywords,
				RawValue: []byte(strings.Join(matchedKeywords, ",")),
			},
		})
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

// isErrorResponse checks if a JSON response is an error response
func isErrorResponse(responseBytes []byte) bool {
	var responseMap map[string]interface{}
	if err := json.Unmarshal(responseBytes, &responseMap); err != nil {
		return false
	}
	// Check for common error response structures
	_, hasError := responseMap["error"]
	_, hasDetail := responseMap["detail"]
	// If it has "error" or "detail" but no "choices", it's likely an error response
	_, hasChoices := responseMap["choices"]
	return (hasError || hasDetail) && !hasChoices
}

// extractErrorMessage extracts error message from error response
func extractErrorMessage(responseBytes []byte) string {
	var responseMap map[string]interface{}
	if err := json.Unmarshal(responseBytes, &responseMap); err != nil {
		return "Failed to parse error response"
	}

	// Try to extract error message from various formats
	if errorObj, ok := responseMap["error"].(map[string]interface{}); ok {
		if msg, ok := errorObj["message"].(string); ok {
			return msg
		}
	}
	if detail, ok := responseMap["detail"].(string); ok {
		return detail
	}
	return "Error response from cache"
}

// splitContentIntoChunks splits content into word-by-word chunks for streaming
func splitContentIntoChunks(content string) []string {
	if content == "" {
		return []string{}
	}

	// Split by words (preserving spaces)
	words := strings.Fields(content)
	if len(words) == 0 {
		return []string{content}
	}

	chunks := make([]string, 0, len(words))
	for i, word := range words {
		if i < len(words)-1 {
			// Add space after word (except last word)
			chunks = append(chunks, word+" ")
		} else {
			// Last word without trailing space
			chunks = append(chunks, word)
		}
	}
	return chunks
}

// CreateCacheHitResponse creates an immediate response from cache
func CreateCacheHitResponse(cachedResponse []byte, isStreaming bool, category string, decisionName string, matchedKeywords []string) *ext_proc.ProcessingResponse {
	var responseBody []byte
	var contentType string

	if isStreaming {
		// For streaming responses, convert cached JSON to SSE format
		contentType = "text/event-stream"

		// Check if cached response is an error response BEFORE parsing
		if isErrorResponse(cachedResponse) {
			errorMsg := extractErrorMessage(cachedResponse)
			logging.Errorf("Cached response is an error response, cannot convert to streaming: %s", errorMsg)

			// Return error in SSE format
			now := time.Now().Unix()
			errorChunk := map[string]interface{}{
				"id":      fmt.Sprintf("chatcmpl-cache-error-%d", now),
				"object":  "chat.completion.chunk",
				"created": now,
				"model":   "cache",
				"choices": []map[string]interface{}{
					{
						"index": 0,
						"delta": map[string]interface{}{
							"role":    "assistant",
							"content": fmt.Sprintf("Error: %s", errorMsg),
						},
						"finish_reason": "error",
					},
				},
			}
			chunkJSON, err := json.Marshal(errorChunk)
			if err != nil {
				responseBody = []byte("data: {\"error\": \"Failed to convert cached error response\"}\n\ndata: [DONE]\n\n")
			} else {
				responseBody = []byte(fmt.Sprintf("data: %s\n\ndata: [DONE]\n\n", chunkJSON))
			}
		} else {
			// Parse the cached JSON response as ChatCompletion
			var cachedCompletion openai.ChatCompletion
			if err := json.Unmarshal(cachedResponse, &cachedCompletion); err != nil {
				logging.Errorf("Error parsing cached response for streaming conversion: %v", err)
				responseBody = []byte("data: {\"error\": \"Failed to convert cached response\"}\n\ndata: [DONE]\n\n")
			} else {
				// Validate that we have valid choices with content
				if len(cachedCompletion.Choices) == 0 || cachedCompletion.Choices[0].Message.Content == "" {
					logging.Errorf("Cached response has no valid choices or content")
					responseBody = []byte("data: {\"error\": \"Cached response has no content\"}\n\ndata: [DONE]\n\n")
				} else {
					// Generate new ID and timestamp for this cache hit (each request is a distinct event)
					unixTimeStep := time.Now().Unix()
					newID := fmt.Sprintf("chatcmpl-cache-%d", unixTimeStep)

					// Extract content and split into chunks
					content := cachedCompletion.Choices[0].Message.Content
					chunks := splitContentIntoChunks(content)

					if len(chunks) == 0 {
						// Fallback: if splitting failed, use original content as single chunk
						chunks = []string{content}
					}

					// Build SSE response with multiple chunks
					var sseChunks []string

					// Send incremental content chunks
					for i, chunkContent := range chunks {
						streamChunk := map[string]interface{}{
							"id":      newID,
							"object":  "chat.completion.chunk",
							"created": unixTimeStep,
							"model":   cachedCompletion.Model,
							"choices": []map[string]interface{}{
								{
									"index": cachedCompletion.Choices[0].Index,
									"delta": map[string]interface{}{
										"content": chunkContent,
									},
									"finish_reason": nil,
								},
							},
						}

						chunkJSON, err := json.Marshal(streamChunk)
						if err != nil {
							logging.Errorf("Error marshaling streaming chunk %d: %v", i, err)
							// Add error chunk instead of silently skipping
							errorChunk := fmt.Sprintf("data: {\"error\": \"Failed to marshal chunk %d\"}\n\n", i)
							sseChunks = append(sseChunks, errorChunk)
							continue
						}
						sseChunks = append(sseChunks, fmt.Sprintf("data: %s\n\n", chunkJSON))
					}

					// Add final chunk with finish_reason
					finalChunk := map[string]interface{}{
						"id":      newID,
						"object":  "chat.completion.chunk",
						"created": unixTimeStep,
						"model":   cachedCompletion.Model,
						"choices": []map[string]interface{}{
							{
								"index":         cachedCompletion.Choices[0].Index,
								"delta":         map[string]interface{}{},
								"finish_reason": cachedCompletion.Choices[0].FinishReason,
							},
						},
					}

					finalChunkJSON, err := json.Marshal(finalChunk)
					if err != nil {
						logging.Errorf("Error marshaling final streaming chunk: %v", err)
						// Still add a basic final chunk to ensure proper SSE termination
						sseChunks = append(sseChunks, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
					} else {
						sseChunks = append(sseChunks, fmt.Sprintf("data: %s\n\n", finalChunkJSON))
					}

					// Add [DONE] marker (always, even if final chunk failed)
					sseChunks = append(sseChunks, "data: [DONE]\n\n")

					// Use bytes.Buffer for more efficient string building
					var buf bytes.Buffer
					for _, chunk := range sseChunks {
						buf.WriteString(chunk)
					}
					responseBody = buf.Bytes()
				}
			}
		}
	} else {
		// For non-streaming responses, parse and regenerate ID/timestamp
		contentType = "application/json"

		// Check if cached response is an error response
		if isErrorResponse(cachedResponse) {
			// For error responses, use as-is (they already have unique IDs)
			responseBody = cachedResponse
		} else {
			// Parse cached response to regenerate ID and timestamp
			var cachedCompletion openai.ChatCompletion
			if err := json.Unmarshal(cachedResponse, &cachedCompletion); err != nil {
				logging.Errorf("Error parsing cached response for ID regeneration: %v", err)
				// Fallback: use cached response as-is if parsing fails
				responseBody = cachedResponse
			} else {
				// Generate new ID and timestamp for this cache hit
				unixTimeStep := time.Now().Unix()
				cachedCompletion.ID = fmt.Sprintf("chatcmpl-cache-%d", unixTimeStep)
				cachedCompletion.Created = unixTimeStep

				// Marshal back to JSON
				marshaledBody, marshalErr := json.Marshal(cachedCompletion)
				if marshalErr != nil {
					logging.Errorf("Error marshaling regenerated cache response: %v", marshalErr)
					// Fallback: use cached response as-is if marshaling fails
					responseBody = cachedResponse
				} else {
					responseBody = marshaledBody
				}
			}
		}
	}

	// Build headers including VSR decision headers for cache hits
	setHeaders := []*core.HeaderValueOption{
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
	}

	// Add matched keywords header if provided
	if len(matchedKeywords) > 0 {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.VSRMatchedKeywords,
				RawValue: []byte(strings.Join(matchedKeywords, ",")),
			},
		})
	}

	immediateResponse := &ext_proc.ImmediateResponse{
		Status: &typev3.HttpStatus{
			Code: typev3.StatusCode_OK,
		},
		Headers: &ext_proc.HeaderMutation{
			SetHeaders: setHeaders,
		},
		Body: responseBody,
	}

	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: immediateResponse,
		},
	}
}
