package protocolcodec

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A prior assistant turn is an output message in the Responses input array and
// must carry output_text content; user turns stay input_text. Strict Responses
// parsers reject an output message whose content is input_text with "unknown
// type for output message content: input_text". History arriving from Chat or
// Anthropic has no provider ids, so no id may be synthesized either.
func TestResponsesAssistantHistoryUsesOutputText(t *testing.T) {
	body := []byte(`{"model":"m","max_tokens":64,"messages":[` +
		`{"role":"user","content":"hi"},` +
		`{"role":"assistant","content":"hello there"},` +
		`{"role":"user","content":"continue"}]}`)
	engine := NewBuiltinEngine()
	for _, source := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1} {
		translated, err := engine.TranslateRequest(source, llmprotocol.OpenAIResponsesV1, body, nil)
		if err != nil {
			t.Fatalf("%s: translation rejected: %v", source, err)
		}
		var req struct {
			Input []struct {
				Type    string `json:"type"`
				Role    string `json:"role"`
				ID      string `json:"id"`
				Content []struct {
					Type string `json:"type"`
				} `json:"content"`
			} `json:"input"`
		}
		if err := json.Unmarshal(translated.Body, &req); err != nil {
			t.Fatalf("%s: decode failed: %v\n%s", source, err, translated.Body)
		}
		want := []struct{ role, content string }{
			{"user", "input_text"},
			{"assistant", "output_text"},
			{"user", "input_text"},
		}
		if len(req.Input) != len(want) {
			t.Fatalf("%s: got %d input items, want %d: %s", source, len(req.Input), len(want), translated.Body)
		}
		for i, item := range req.Input {
			if item.Type != "message" || item.Role != want[i].role || len(item.Content) != 1 || item.Content[0].Type != want[i].content {
				t.Fatalf("%s: input[%d] = %s/%s %+v, want %s message with %s: %s",
					source, i, item.Type, item.Role, item.Content, want[i].role, want[i].content, translated.Body)
			}
			if item.ID != "" {
				t.Fatalf("%s: input[%d] carries synthesized id %q: %s", source, i, item.ID, translated.Body)
			}
		}
	}
}

// A Responses output message cannot carry an image, so an Anthropic assistant
// turn with an image block fails in the router with a typed error instead of
// being sent to a provider that rejects it.
func TestResponsesAssistantHistoryRejectsImages(t *testing.T) {
	body := []byte(`{"model":"m","max_tokens":64,"messages":[` +
		`{"role":"user","content":"hi"},` +
		`{"role":"assistant","content":[{"type":"text","text":"here"},` +
		`{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgo="}}]},` +
		`{"role":"user","content":"continue"}]}`)
	_, err := NewBuiltinEngine().TranslateRequest(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1, body, nil)
	var protocolErr *llmprotocol.ProtocolError
	if !errors.As(err, &protocolErr) || protocolErr.Code != "image_content_position" {
		t.Fatalf("translation error = %v, want image_content_position", err)
	}
}
