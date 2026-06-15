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
)

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

// buildStreamingCacheErrorBody returns an SSE-formatted error body from a cached error response.
func buildStreamingCacheErrorBody(cachedResponse []byte) []byte {
	errorMsg := extractErrorMessage(cachedResponse)
	logging.Errorf("Cached response is an error response, cannot convert to streaming: %s", errorMsg)

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
		return []byte("data: {\"error\": \"Failed to convert cached error response\"}\n\ndata: [DONE]\n\n")
	}
	return []byte(fmt.Sprintf("data: %s\n\ndata: [DONE]\n\n", chunkJSON))
}

// buildStreamingCacheBody converts a cached ChatCompletion into SSE chunks.
func buildStreamingCacheBody(cachedResponse []byte) []byte {
	var cachedCompletion openai.ChatCompletion
	if err := json.Unmarshal(cachedResponse, &cachedCompletion); err != nil {
		logging.Errorf("Error parsing cached response for streaming conversion: %v", err)
		return []byte("data: {\"error\": \"Failed to convert cached response\"}\n\ndata: [DONE]\n\n")
	}

	if len(cachedCompletion.Choices) == 0 || cachedCompletion.Choices[0].Message.Content == "" {
		logging.Errorf("Cached response has no valid choices or content")
		return []byte("data: {\"error\": \"Cached response has no content\"}\n\ndata: [DONE]\n\n")
	}

	unixTimeStep := time.Now().Unix()
	newID := fmt.Sprintf("chatcmpl-cache-%d", unixTimeStep)
	content := cachedCompletion.Choices[0].Message.Content
	chunks := splitContentIntoChunks(content)
	if len(chunks) == 0 {
		chunks = []string{content}
	}

	var sseChunks []string
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
			sseChunks = append(sseChunks, fmt.Sprintf("data: {\"error\": \"Failed to marshal chunk %d\"}\n\n", i))
			continue
		}
		sseChunks = append(sseChunks, fmt.Sprintf("data: %s\n\n", chunkJSON))
	}

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
		sseChunks = append(sseChunks, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
	} else {
		sseChunks = append(sseChunks, fmt.Sprintf("data: %s\n\n", finalChunkJSON))
	}
	sseChunks = append(sseChunks, "data: [DONE]\n\n")

	var buf bytes.Buffer
	for _, chunk := range sseChunks {
		buf.WriteString(chunk)
	}
	return buf.Bytes()
}

// buildNonStreamingCacheBody regenerates ID/timestamp on a cached response.
func buildNonStreamingCacheBody(cachedResponse []byte) []byte {
	if isErrorResponse(cachedResponse) {
		return cachedResponse
	}

	var cachedCompletion openai.ChatCompletion
	if err := json.Unmarshal(cachedResponse, &cachedCompletion); err != nil {
		logging.Errorf("Error parsing cached response for ID regeneration: %v", err)
		return cachedResponse
	}

	unixTimeStep := time.Now().Unix()
	cachedCompletion.ID = fmt.Sprintf("chatcmpl-cache-%d", unixTimeStep)
	cachedCompletion.Created = unixTimeStep

	marshaledBody, err := json.Marshal(cachedCompletion)
	if err != nil {
		logging.Errorf("Error marshaling regenerated cache response: %v", err)
		return cachedResponse
	}
	return marshaledBody
}

// CreateCacheHitResponse creates an immediate response from cache
func CreateCacheHitResponse(cachedResponse []byte, isStreaming bool, category string, decisionName string, matchedKeywords []string, similarity ...float32) *ext_proc.ProcessingResponse {
	var responseBody []byte
	var contentType string

	if isStreaming {
		contentType = "text/event-stream"
		if isErrorResponse(cachedResponse) {
			responseBody = buildStreamingCacheErrorBody(cachedResponse)
		} else {
			responseBody = buildStreamingCacheBody(cachedResponse)
		}
	} else {
		contentType = "application/json"
		responseBody = buildNonStreamingCacheBody(cachedResponse)
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

	// Add cache similarity header
	if len(similarity) > 0 && similarity[0] > 0 {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      "x-vsr-cache-similarity",
				RawValue: []byte(fmt.Sprintf("%.4f", similarity[0])),
			},
		})
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

// buildStreamingFastBody converts a message into SSE chunks for the fast_response plugin.
func buildStreamingFastBody(message string, unixTimeStep int64) []byte {
	chunks := splitContentIntoChunks(message)
	if len(chunks) == 0 {
		chunks = []string{message}
	}

	var sseChunks []string
	newID := fmt.Sprintf("chatcmpl-fast-%d", unixTimeStep)

	for i, chunkContent := range chunks {
		streamChunk := map[string]interface{}{
			"id":      newID,
			"object":  "chat.completion.chunk",
			"created": unixTimeStep,
			"model":   "router",
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]interface{}{
						"content": chunkContent,
					},
					"finish_reason": nil,
				},
			},
		}
		if i == 0 {
			streamChunk["choices"].([]map[string]interface{})[0]["delta"].(map[string]interface{})["role"] = "assistant"
		}

		chunkJSON, err := json.Marshal(streamChunk)
		if err != nil {
			continue
		}
		sseChunks = append(sseChunks, fmt.Sprintf("data: %s\n\n", chunkJSON))
	}

	finalChunk := map[string]interface{}{
		"id":      newID,
		"object":  "chat.completion.chunk",
		"created": unixTimeStep,
		"model":   "router",
		"choices": []map[string]interface{}{
			{
				"index":         0,
				"delta":         map[string]interface{}{},
				"finish_reason": "stop",
			},
		},
	}
	finalJSON, _ := json.Marshal(finalChunk)
	sseChunks = append(sseChunks, fmt.Sprintf("data: %s\n\n", finalJSON))
	sseChunks = append(sseChunks, "data: [DONE]\n\n")

	return []byte(strings.Join(sseChunks, ""))
}

// buildNonStreamingFastBody creates a JSON ChatCompletion body for the fast_response plugin.
func buildNonStreamingFastBody(message string, unixTimeStep int64) []byte {
	openAIResponse := openai.ChatCompletion{
		ID:      fmt.Sprintf("chatcmpl-fast-%d", unixTimeStep),
		Object:  "chat.completion",
		Created: unixTimeStep,
		Model:   "router",
		Choices: []openai.ChatCompletionChoice{
			{
				Index: 0,
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: message,
				},
				FinishReason: "stop",
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
		logging.Errorf("Error marshaling fast_response: %v", err)
		return []byte(`{"error": "Failed to generate response"}`)
	}
	return responseBody
}

// CreateFastResponse creates an OpenAI-compatible immediate response for the
// fast_response plugin. It supports both streaming (SSE) and non-streaming (JSON)
// formats based on the original request's "stream" flag.
// The response is returned as 200 OK with finish_reason "stop".
func CreateFastResponse(message string, isStreaming bool, decisionName string) *ext_proc.ProcessingResponse {
	unixTimeStep := time.Now().Unix()
	var responseBody []byte
	var contentType string

	if isStreaming {
		contentType = "text/event-stream"
		responseBody = buildStreamingFastBody(message, unixTimeStep)
	} else {
		contentType = "application/json"
		responseBody = buildNonStreamingFastBody(message, unixTimeStep)
	}

	setHeaders := []*core.HeaderValueOption{
		{
			Header: &core.HeaderValue{
				Key:      "content-type",
				RawValue: []byte(contentType),
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
				Key:      headers.VSRFastResponse,
				RawValue: []byte("true"),
			},
		},
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
