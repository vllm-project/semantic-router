package http

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/openai/openai-go"
)

func TestBuildStreamingModalityBody_TextOnly(t *testing.T) {
	completion := openai.ChatCompletion{
		ID:      "chatcmpl-test-001",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "mock-diffusion",
		Choices: []openai.ChatCompletionChoice{
			{
				Index: 0,
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: "Here is the generated image based on your request.",
				},
				FinishReason: "stop",
			},
		},
	}
	body, _ := json.Marshal(completion)

	result := BuildStreamingModalityBody(body)
	resultStr := string(result)

	t.Logf("SSE output:\n%s", resultStr[:min(len(resultStr), 400)])

	// Must start with data:
	if !bytes.HasPrefix(result, []byte("data: ")) {
		t.Fatal("BUG: SSE output does not start with 'data: '")
	}

	// Must have [DONE] termination
	if !strings.Contains(resultStr, "data: [DONE]") {
		t.Fatal("BUG: SSE missing termination 'data: [DONE]'")
	}

	// Must have chat.completion.chunk object type
	if !strings.Contains(resultStr, "chat.completion.chunk") {
		t.Fatal("BUG: SSE missing 'chat.completion.chunk' object")
	}

	// Must have finish_reason in final chunk
	if !strings.Contains(resultStr, `"finish_reason":"stop"`) {
		t.Error("BUG: SSE missing finish_reason in final chunk")
	}

	t.Log("PASS: text-only modality response → valid SSE")
}

func TestBuildStreamingModalityBody_Multimodal(t *testing.T) {
	// Build multimodal JSON where content is an actual JSON array, not a string.
	// We can't use openai.ChatCompletion directly because Content is typed string.
	multimodal := map[string]interface{}{
		"id":      "chatcmpl-test-002",
		"object":  "chat.completion",
		"created": 1234567890,
		"model":   "mock-omni",
		"choices": []interface{}{
			map[string]interface{}{
				"index": 0,
				"message": map[string]interface{}{
					"role": "assistant",
					"content": []interface{}{
						map[string]interface{}{"type": "text", "text": "Photosynthesis is a process..."},
						map[string]interface{}{"type": "image_url", "image_url": map[string]string{"url": "data:image/png;base64,FAKE"}},
					},
				},
				"finish_reason": "stop",
			},
		},
	}
	body, _ := json.Marshal(multimodal)

	result := BuildStreamingModalityBody(body)
	resultStr := string(result)

	t.Logf("SSE output:\n%s", resultStr[:min(len(resultStr), 500)])

	// Must have [DONE]
	if !strings.Contains(resultStr, "data: [DONE]") {
		t.Fatal("BUG: multimodal SSE missing 'data: [DONE]'")
	}

	// Multimodal content should be in a single delta chunk
	if !strings.Contains(resultStr, "image_url") {
		t.Error("BUG: multimodal content lost image_url in SSE conversion")
	}

	t.Log("PASS: multimodal modality response → valid SSE with image_url preserved")
}

func TestBuildStreamingModalityBody_InvalidJSON(t *testing.T) {
	result := BuildStreamingModalityBody([]byte("not-json"))

	if !bytes.Contains(result, []byte("data: [DONE]")) {
		t.Fatal("BUG: even on invalid input, [DONE] must be present")
	}
	t.Log("PASS: invalid JSON → graceful error SSE")
}

func TestIsMultimodalContent(t *testing.T) {
	if isMultimodalContent("plain string") {
		t.Error("BUG: plain string detected as multimodal")
	}
	if !isMultimodalContent([]interface{}{}) {
		t.Error("BUG: array not detected as multimodal")
	}
	t.Log("PASS: multimodal content detection")
}

func TestExtractStringContent(t *testing.T) {
	if extractStringContent("hello") != "hello" {
		t.Error("BUG: failed to extract string content")
	}
	if extractStringContent(42) != "" {
		t.Error("BUG: non-string should return empty")
	}
	t.Log("PASS: string content extraction")
}
