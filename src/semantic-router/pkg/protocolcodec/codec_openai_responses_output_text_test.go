package protocolcodec

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

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

func TestResponsesAssistantMediaHistorySurvivesMutation(t *testing.T) {
	for _, tc := range []struct {
		name, parts string
		want        []string
	}{
		{"image", `{"type":"input_image","image_url":"https://example.com/a.png"}`, []string{"input_image"}},
		{"file", `{"type":"input_file","file_id":"file-abc"}`, []string{"input_file"}},
		{"text-and-image", `{"type":"input_text","text":"see"},{"type":"input_image","image_url":"https://example.com/a.png"}`, []string{"input_text", "input_image"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := []byte(`{"model":"alias","input":[` +
				`{"role":"user","content":[{"type":"input_text","text":"hi"}]},` +
				`{"role":"assistant","content":[` + tc.parts + `]},` +
				`{"role":"user","content":[{"type":"input_text","text":"continue"}]}]}`)
			engine := NewBuiltinEngine()
			request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, body)
			if err != nil {
				t.Fatal(err)
			}
			request.Model, request.Generation = "backend", request.Generation+1
			encoded, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, request, envelope)
			if err != nil {
				t.Fatalf("re-encoding rejected assistant %s history: %v", tc.name, err)
			}
			var req struct {
				Input []struct {
					Role    string `json:"role"`
					Content []struct {
						Type string `json:"type"`
					} `json:"content"`
				} `json:"input"`
			}
			if err := json.Unmarshal(encoded.Body, &req); err != nil {
				t.Fatalf("decode failed: %v\n%s", err, encoded.Body)
			}
			if len(req.Input) != 3 || req.Input[1].Role != "assistant" || len(req.Input[1].Content) != len(tc.want) {
				t.Fatalf("assistant history changed shape: %s", encoded.Body)
			}
			for i, part := range req.Input[1].Content {
				if part.Type != tc.want[i] {
					t.Fatalf("assistant content[%d] = %q, want %q: %s", i, part.Type, tc.want[i], encoded.Body)
				}
			}
		})
	}
}
