package extproc

import (
	"encoding/json"
	"testing"

	"github.com/openai/openai-go"
)

// A reasoning model streams its thinking under delta.reasoning_content. The
// streaming accumulator must capture it, and the reconstructed response that is
// cached must carry it — otherwise a later cache hit returns a response missing
// the reasoning the original live stream delivered.

func TestExtractStreamingContentAccumulatesReasoning(t *testing.T) {
	ctx := &RequestContext{StreamingMetadata: map[string]interface{}{}}

	chunks := []map[string]interface{}{
		{"choices": []interface{}{map[string]interface{}{
			"delta": map[string]interface{}{"reasoning_content": "Let me "},
		}}},
		{"choices": []interface{}{map[string]interface{}{
			"delta": map[string]interface{}{"reasoning_content": "think. "},
		}}},
		{"choices": []interface{}{map[string]interface{}{
			"delta": map[string]interface{}{"content": "Answer."},
		}}},
	}
	for _, c := range chunks {
		extractStreamingContent(ctx, c)
	}

	if ctx.StreamingReasoning != "Let me think. " {
		t.Fatalf("reasoning not accumulated: %q", ctx.StreamingReasoning)
	}
	if ctx.StreamingContent != "Answer." {
		t.Fatalf("content not accumulated: %q", ctx.StreamingContent)
	}
}

func TestReconstructedStreamingResponseIncludesReasoning(t *testing.T) {
	ctx := &RequestContext{
		StreamingContent:   "The answer is 4.",
		StreamingReasoning: "2+2 equals 4.",
		StreamingMetadata: map[string]interface{}{
			"id":      "chatcmpl-reasoning",
			"model":   "qwen3",
			"created": int64(1),
		},
	}

	body, err := buildReconstructedStreamingResponse(ctx, openai.CompletionUsage{}, false)
	if err != nil {
		t.Fatalf("reconstruct error: %v", err)
	}

	var parsed struct {
		Choices []struct {
			Message struct {
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(parsed.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(parsed.Choices))
	}
	if parsed.Choices[0].Message.ReasoningContent != "2+2 equals 4." {
		t.Fatalf("reconstructed response dropped reasoning_content: %q", parsed.Choices[0].Message.ReasoningContent)
	}
	if parsed.Choices[0].Message.Content != "The answer is 4." {
		t.Fatalf("content mismatch: %q", parsed.Choices[0].Message.Content)
	}
}

// A response with no reasoning must not gain an empty reasoning_content field.
func TestReconstructedStreamingResponseOmitsEmptyReasoning(t *testing.T) {
	ctx := &RequestContext{
		StreamingContent: "hello",
		StreamingMetadata: map[string]interface{}{
			"id":      "chatcmpl-plain",
			"model":   "qwen3",
			"created": int64(1),
		},
	}
	body, err := buildReconstructedStreamingResponse(ctx, openai.CompletionUsage{}, false)
	if err != nil {
		t.Fatalf("reconstruct error: %v", err)
	}
	var raw map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	choice := raw["choices"].([]interface{})[0].(map[string]interface{})
	msg := choice["message"].(map[string]interface{})
	if _, present := msg["reasoning_content"]; present {
		t.Fatal("reasoning_content must be absent when no reasoning was streamed")
	}
}
