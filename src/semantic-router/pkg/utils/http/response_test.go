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
	response := CreateCacheHitResponse(cachedResponse, false, "math", "math_decision", nil)

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

	// Verify body can be parsed as JSON
	var parsedResponse openai.ChatCompletion
	if err := json.Unmarshal(immediateResp.Body, &parsedResponse); err != nil {
		t.Errorf("Failed to parse response body as JSON: %v", err)
	}

	if parsedResponse.Object != "chat.completion" {
		t.Errorf("Expected object chat.completion, got %s", parsedResponse.Object)
	}

	// Verify ID is regenerated (should have "chatcmpl-cache-" prefix)
	if !strings.HasPrefix(parsedResponse.ID, "chatcmpl-cache-") {
		t.Errorf("Expected ID to start with 'chatcmpl-cache-', got %s", parsedResponse.ID)
	}

	// Verify ID is different from cached ID
	if parsedResponse.ID == "chatcmpl-test-123" {
		t.Error("ID was not regenerated - still using cached ID")
	}

	// Verify created timestamp is updated (should be recent, not the old timestamp)
	if parsedResponse.Created == 1234567890 {
		t.Error("Created timestamp was not updated - still using cached timestamp")
	}

	// Verify other fields are preserved
	if parsedResponse.Model != "test-model" {
		t.Errorf("Expected model test-model, got %s", parsedResponse.Model)
	}
	if len(parsedResponse.Choices) == 0 || parsedResponse.Choices[0].Message.Content != "This is a cached response." {
		t.Error("Content was not preserved correctly")
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
	response := CreateCacheHitResponse(cachedResponse, true, "math", "math_decision", nil)

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

	// Parse all SSE chunks (content is now split word-by-word)
	lines := strings.Split(bodyStr, "\n")
	var dataChunks []string
	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") && !strings.Contains(line, "[DONE]") {
			dataChunks = append(dataChunks, strings.TrimPrefix(line, "data: "))
		}
	}

	if len(dataChunks) == 0 {
		t.Fatal("No data chunks found in SSE response")
	}

	// Verify we have multiple chunks (content is split word-by-word)
	// "This is a cached streaming response." should be split into 5 words
	expectedMinChunks := 5 // "This", "is", "a", "cached", "streaming", "response."
	if len(dataChunks) < expectedMinChunks {
		t.Errorf("Expected at least %d chunks (word-by-word), got %d", expectedMinChunks, len(dataChunks))
	}

	// Parse the first chunk
	var firstChunk map[string]interface{}
	if err := json.Unmarshal([]byte(dataChunks[0]), &firstChunk); err != nil {
		t.Fatalf("Failed to parse first SSE chunk as JSON: %v", err)
	}

	// Verify chunk structure
	if firstChunk["object"] != "chat.completion.chunk" {
		t.Errorf("Expected object chat.completion.chunk, got %v", firstChunk["object"])
	}

	// Verify ID is regenerated (should have "chatcmpl-cache-" prefix)
	chunkID, ok := firstChunk["id"].(string)
	if !ok {
		t.Fatal("ID is not a string")
	}
	if !strings.HasPrefix(chunkID, "chatcmpl-cache-") {
		t.Errorf("Expected ID to start with 'chatcmpl-cache-', got %s", chunkID)
	}

	// Verify ID is different from cached ID
	if chunkID == "chatcmpl-test-456" {
		t.Error("ID was not regenerated - still using cached ID")
	}

	// Verify created timestamp is updated
	chunkCreated, ok := firstChunk["created"].(float64)
	if !ok {
		t.Fatal("Created is not a number")
	}
	if int64(chunkCreated) == 1234567890 {
		t.Error("Created timestamp was not updated - still using cached timestamp")
	}

	// Verify choices structure in first chunk
	choices, ok := firstChunk["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		t.Fatal("Choices not found or empty in first chunk")
	}

	choice := choices[0].(map[string]interface{})
	delta, ok := choice["delta"].(map[string]interface{})
	if !ok {
		t.Fatal("Delta not found in choice")
	}

	// First chunk should contain only the first word (with space)
	expectedFirstContent := "This "
	if delta["content"] != expectedFirstContent {
		t.Errorf("Expected first chunk content '%s', got %v", expectedFirstContent, delta["content"])
	}

	// First chunk should not have finish_reason (nil)
	if choice["finish_reason"] != nil {
		t.Errorf("Expected finish_reason nil in content chunks, got %v", choice["finish_reason"])
	}

	// Verify final chunk has finish_reason
	var finalChunk map[string]interface{}
	if err := json.Unmarshal([]byte(dataChunks[len(dataChunks)-1]), &finalChunk); err != nil {
		t.Fatalf("Failed to parse final chunk as JSON: %v", err)
	}

	finalChoices, ok := finalChunk["choices"].([]interface{})
	if !ok || len(finalChoices) == 0 {
		t.Fatal("Choices not found in final chunk")
	}

	finalChoice := finalChoices[0].(map[string]interface{})
	if finalChoice["finish_reason"] != "stop" {
		t.Errorf("Expected finish_reason 'stop' in final chunk, got %v", finalChoice["finish_reason"])
	}

	// Verify all chunks combined reconstruct the original content
	var reconstructedContent strings.Builder
	for i := 0; i < len(dataChunks)-1; i++ { // Exclude final chunk
		var chunk map[string]interface{}
		if err := json.Unmarshal([]byte(dataChunks[i]), &chunk); err != nil {
			continue
		}
		if choices, ok := chunk["choices"].([]interface{}); ok && len(choices) > 0 {
			if choice, ok := choices[0].(map[string]interface{}); ok {
				if delta, ok := choice["delta"].(map[string]interface{}); ok {
					if content, ok := delta["content"].(string); ok {
						reconstructedContent.WriteString(content)
					}
				}
			}
		}
	}

	expectedContent := "This is a cached streaming response."
	if reconstructedContent.String() != expectedContent {
		t.Errorf("Reconstructed content mismatch. Expected '%s', got '%s'", expectedContent, reconstructedContent.String())
	}
}

func TestCreateCacheHitResponse_StreamingWithInvalidJSON(t *testing.T) {
	// Test with invalid JSON
	invalidJSON := []byte("invalid json")

	response := CreateCacheHitResponse(invalidJSON, true, "other", "other_decision", nil)

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

func TestCreateCacheHitResponse_StreamingWithErrorResponse(t *testing.T) {
	// Test with cached error response (e.g., from upstream LLM error)
	errorResponse := []byte(`{"error": {"message": "temperature must be > 0"}, "detail": "Validation error"}`)

	response := CreateCacheHitResponse(errorResponse, true, "math", "math_decision", nil)

	if response == nil {
		t.Fatal("Response is nil")
	}

	immediateResp := response.GetImmediateResponse()
	if immediateResp == nil {
		t.Fatal("ImmediateResponse is nil")
	}

	bodyStr := string(immediateResp.Body)

	// Should contain error message
	if !strings.Contains(bodyStr, "Error:") {
		t.Error("Expected error message in SSE response")
	}

	// Should be in SSE format
	if !strings.Contains(bodyStr, "data: [DONE]") {
		t.Error("Expected SSE terminator")
	}

	// Should have chat.completion.chunk format
	if !strings.Contains(bodyStr, "chat.completion.chunk") {
		t.Error("Expected chat.completion.chunk object type")
	}
}

func TestCreateCacheHitResponse_StreamingWithEmptyContent(t *testing.T) {
	// Test with empty content
	cachedCompletion := openai.ChatCompletion{
		ID:      "chatcmpl-test-empty",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "test-model",
		Choices: []openai.ChatCompletionChoice{
			{
				Index: 0,
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: "", // Empty content
				},
				FinishReason: "stop",
			},
		},
	}

	cachedResponse, err := json.Marshal(cachedCompletion)
	if err != nil {
		t.Fatalf("Failed to marshal cached response: %v", err)
	}

	response := CreateCacheHitResponse(cachedResponse, true, "math", "math_decision", nil)

	immediateResp := response.GetImmediateResponse()
	bodyStr := string(immediateResp.Body)

	// Should still have [DONE] marker
	if !strings.Contains(bodyStr, "data: [DONE]") {
		t.Error("Expected SSE terminator even with empty content")
	}

	// Should contain error about no content
	if !strings.Contains(bodyStr, "no content") {
		t.Error("Expected error message about missing content")
	}
}

func TestCreateCacheHitResponse_StreamingWithEmptyChoices(t *testing.T) {
	// Test with empty choices array
	cachedCompletion := openai.ChatCompletion{
		ID:      "chatcmpl-test-empty-choices",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "test-model",
		Choices: []openai.ChatCompletionChoice{}, // Empty choices
	}

	cachedResponse, err := json.Marshal(cachedCompletion)
	if err != nil {
		t.Fatalf("Failed to marshal cached response: %v", err)
	}

	response := CreateCacheHitResponse(cachedResponse, true, "math", "math_decision", nil)

	immediateResp := response.GetImmediateResponse()
	bodyStr := string(immediateResp.Body)

	// Should contain error about no valid choices (check for actual error message)
	hasError := strings.Contains(bodyStr, "no content") ||
		strings.Contains(bodyStr, "no valid choices") ||
		strings.Contains(bodyStr, "Cached response has no")
	if !hasError {
		preview := bodyStr
		if len(preview) > 200 {
			preview = preview[:200]
		}
		t.Errorf("Expected error message about missing choices, got: %s", preview)
	}

	if !strings.Contains(bodyStr, "data: [DONE]") {
		t.Error("Expected SSE terminator")
	}
}

func TestCreateCacheHitResponse_StreamingWithWhitespaceContent(t *testing.T) {
	// Test with whitespace-only content
	cachedCompletion := openai.ChatCompletion{
		ID:      "chatcmpl-test-whitespace",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "test-model",
		Choices: []openai.ChatCompletionChoice{
			{
				Index: 0,
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: "   ", // Only whitespace
				},
				FinishReason: "stop",
			},
		},
	}

	cachedResponse, err := json.Marshal(cachedCompletion)
	if err != nil {
		t.Fatalf("Failed to marshal cached response: %v", err)
	}

	response := CreateCacheHitResponse(cachedResponse, true, "math", "math_decision", nil)

	immediateResp := response.GetImmediateResponse()
	bodyStr := string(immediateResp.Body)

	// Should still produce valid SSE (whitespace is preserved as single chunk)
	if !strings.Contains(bodyStr, "data: [DONE]") {
		t.Error("Expected SSE terminator")
	}
}

func TestCreateCacheHitResponse_StreamingWithLongContent(t *testing.T) {
	// Test with long content to verify multiple chunks
	longContent := "This is a very long response that should be split into many word-by-word chunks for proper streaming behavior."

	cachedCompletion := openai.ChatCompletion{
		ID:      "chatcmpl-test-long",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "test-model",
		Choices: []openai.ChatCompletionChoice{
			{
				Index: 0,
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: longContent,
				},
				FinishReason: "stop",
			},
		},
	}

	cachedResponse, err := json.Marshal(cachedCompletion)
	if err != nil {
		t.Fatalf("Failed to marshal cached response: %v", err)
	}

	response := CreateCacheHitResponse(cachedResponse, true, "math", "math_decision", nil)

	immediateResp := response.GetImmediateResponse()
	bodyStr := string(immediateResp.Body)

	// Count chunks (excluding [DONE])
	lines := strings.Split(bodyStr, "\n")
	chunkCount := 0
	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") && !strings.Contains(line, "[DONE]") {
			chunkCount++
		}
	}

	// Should have multiple chunks (one per word + final chunk)
	expectedMinChunks := 15 // Number of words in longContent
	if chunkCount < expectedMinChunks {
		t.Errorf("Expected at least %d chunks for long content, got %d", expectedMinChunks, chunkCount)
	}

	// Verify [DONE] marker
	if !strings.Contains(bodyStr, "data: [DONE]") {
		t.Error("Expected SSE terminator")
	}
}

func TestSplitContentIntoChunks(t *testing.T) {
	tests := []struct {
		name     string
		content  string
		expected []string
	}{
		{
			name:     "empty string",
			content:  "",
			expected: []string{},
		},
		{
			name:     "single word",
			content:  "Hello",
			expected: []string{"Hello"},
		},
		{
			name:     "multiple words",
			content:  "Hello world",
			expected: []string{"Hello ", "world"},
		},
		{
			name:     "three words",
			content:  "This is test",
			expected: []string{"This ", "is ", "test"},
		},
		{
			name:     "with punctuation",
			content:  "Hello, world!",
			expected: []string{"Hello, ", "world!"},
		},
		{
			name:     "whitespace only",
			content:  "   ",
			expected: []string{"   "}, // Preserved as single chunk
		},
		{
			name:     "multiple spaces",
			content:  "word1  word2",
			expected: []string{"word1 ", "word2"}, // Extra spaces collapsed by strings.Fields
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := splitContentIntoChunks(tt.content)

			if len(result) != len(tt.expected) {
				t.Errorf("Expected %d chunks, got %d", len(tt.expected), len(result))
				return
			}

			for i, expected := range tt.expected {
				if i < len(result) && result[i] != expected {
					t.Errorf("Chunk %d: expected '%s', got '%s'", i, expected, result[i])
				}
			}
		})
	}
}
