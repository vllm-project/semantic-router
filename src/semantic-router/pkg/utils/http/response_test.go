package http

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/openai/openai-go"
)

func TestCreateCacheHitResponse_NonStreaming(t *testing.T) {
	// Create a sample cached response
	cachedCompletion := openai.ChatCompletion{
		ID:      "chatcmpl-test-123",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "test-model",
		Choices: []openai.ChatCompletionChoice{
			{
				Index: 0,
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: "This is a cached response.",
				},
				FinishReason: "stop",
			},
		},
		Usage: openai.CompletionUsage{
			PromptTokens:     10,
			CompletionTokens: 5,
			TotalTokens:      15,
		},
	}

	cachedResponse, err := json.Marshal(cachedCompletion)
	if err != nil {
		t.Fatalf("Failed to marshal cached response: %v", err)
	}

	// Test non-streaming response
	response := CreateCacheHitResponse(cachedResponse, false, "math", "math_decision")

	// Verify response structure
	if response == nil {
		t.Fatal("Response is nil")
	}

	immediateResp := response.GetImmediateResponse()
	if immediateResp == nil {
		t.Fatal("ImmediateResponse is nil")
	}

	// Verify status code
	if immediateResp.Status.Code.String() != "OK" {
		t.Errorf("Expected status OK, got %s", immediateResp.Status.Code.String())
	}

	// Verify content-type header
	var contentType string
	var cacheHit string
	for _, header := range immediateResp.Headers.SetHeaders {
		if header.Header.Key == "content-type" {
			contentType = string(header.Header.RawValue)
		}
		if header.Header.Key == "x-vsr-cache-hit" {
			cacheHit = string(header.Header.RawValue)
		}
	}

	if contentType != "application/json" {
		t.Errorf("Expected content-type application/json, got %s", contentType)
	}

	if cacheHit != "true" {
		t.Errorf("Expected x-vsr-cache-hit true, got %s", cacheHit)
	}

	// Verify body is unchanged
	if string(immediateResp.Body) != string(cachedResponse) {
		t.Error("Body was modified for non-streaming response")
	}

	// Verify body can be parsed as JSON
	var parsedResponse openai.ChatCompletion
	if err := json.Unmarshal(immediateResp.Body, &parsedResponse); err != nil {
		t.Errorf("Failed to parse response body as JSON: %v", err)
	}

	if parsedResponse.Object != "chat.completion" {
		t.Errorf("Expected object chat.completion, got %s", parsedResponse.Object)
	}
}

func TestCreateCacheHitResponse_Streaming(t *testing.T) {
	// Create a sample cached response
	cachedCompletion := openai.ChatCompletion{
		ID:      "chatcmpl-test-456",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "test-model",
		Choices: []openai.ChatCompletionChoice{
			{
				Index: 0,
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: "This is a cached streaming response.",
				},
				FinishReason: "stop",
			},
		},
		Usage: openai.CompletionUsage{
			PromptTokens:     10,
			CompletionTokens: 5,
			TotalTokens:      15,
		},
	}

	cachedResponse, err := json.Marshal(cachedCompletion)
	if err != nil {
		t.Fatalf("Failed to marshal cached response: %v", err)
	}

	// Test streaming response
	response := CreateCacheHitResponse(cachedResponse, true, "math", "math_decision")

	// Verify response structure
	if response == nil {
		t.Fatal("Response is nil")
	}

	immediateResp := response.GetImmediateResponse()
	if immediateResp == nil {
		t.Fatal("ImmediateResponse is nil")
	}

	// Verify status code
	if immediateResp.Status.Code.String() != "OK" {
		t.Errorf("Expected status OK, got %s", immediateResp.Status.Code.String())
	}

	// Verify content-type header
	var contentType string
	var cacheHit string
	for _, header := range immediateResp.Headers.SetHeaders {
		if header.Header.Key == "content-type" {
			contentType = string(header.Header.RawValue)
		}
		if header.Header.Key == "x-vsr-cache-hit" {
			cacheHit = string(header.Header.RawValue)
		}
	}

	if contentType != "text/event-stream" {
		t.Errorf("Expected content-type text/event-stream, got %s", contentType)
	}

	if cacheHit != "true" {
		t.Errorf("Expected x-vsr-cache-hit true, got %s", cacheHit)
	}

	// Verify body is in SSE format
	bodyStr := string(immediateResp.Body)
	if !strings.HasPrefix(bodyStr, "data: ") {
		t.Error("Body does not start with 'data: ' prefix")
	}

	if !strings.Contains(bodyStr, "data: [DONE]") {
		t.Error("Body does not contain 'data: [DONE]' terminator")
	}

	// Parse the SSE data
	lines := strings.Split(bodyStr, "\n")
	var dataLine string
	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") && !strings.Contains(line, "[DONE]") {
			dataLine = strings.TrimPrefix(line, "data: ")
			break
		}
	}

	if dataLine == "" {
		t.Fatal("No data line found in SSE response")
	}

	// Parse the JSON chunk
	var chunk map[string]interface{}
	if err := json.Unmarshal([]byte(dataLine), &chunk); err != nil {
		t.Fatalf("Failed to parse SSE data as JSON: %v", err)
	}

	// Verify chunk structure
	if chunk["object"] != "chat.completion.chunk" {
		t.Errorf("Expected object chat.completion.chunk, got %v", chunk["object"])
	}

	if chunk["id"] != "chatcmpl-test-456" {
		t.Errorf("Expected id chatcmpl-test-456, got %v", chunk["id"])
	}

	// Verify choices structure
	choices, ok := chunk["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		t.Fatal("Choices not found or empty")
	}

	choice := choices[0].(map[string]interface{})
	delta, ok := choice["delta"].(map[string]interface{})
	if !ok {
		t.Fatal("Delta not found in choice")
	}

	if delta["role"] != "assistant" {
		t.Errorf("Expected role assistant, got %v", delta["role"])
	}

	if delta["content"] != "This is a cached streaming response." {
		t.Errorf("Expected content 'This is a cached streaming response.', got %v", delta["content"])
	}

	if choice["finish_reason"] != "stop" {
		t.Errorf("Expected finish_reason stop, got %v", choice["finish_reason"])
	}
}

func TestCreateCacheHitResponse_StreamingWithInvalidJSON(t *testing.T) {
	// Test with invalid JSON
	invalidJSON := []byte("invalid json")

	response := CreateCacheHitResponse(invalidJSON, true, "other", "other_decision")

	// Verify response structure
	if response == nil {
		t.Fatal("Response is nil")
	}

	immediateResp := response.GetImmediateResponse()
	if immediateResp == nil {
		t.Fatal("ImmediateResponse is nil")
	}

	// Verify error response
	bodyStr := string(immediateResp.Body)
	if !strings.Contains(bodyStr, "error") {
		t.Error("Expected error message in response body")
	}

	if !strings.Contains(bodyStr, "data: [DONE]") {
		t.Error("Expected SSE terminator even in error case")
	}
}
