package protocolcodec

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const anthropicPerMessageEffortRequest = `{
	"model":"m","max_tokens":64,"messages":[
		{"role":"user","content":"first turn"},
		{"role":"system","content":[{"type":"text","text":"Today is 2026-01-01."}],"output_config":{"effort":"medium"}},
		{"role":"assistant","content":"first reply"},
		{"role":"system","content":[],"output_config":{"effort":"low"}},
		{"role":"user","content":"second turn"}
	]}`

func TestAnthropicPerMessageEffortSurvivesDispatchAndProjectsToOtherFormats(t *testing.T) {
	engine := NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.AnthropicMessagesV1, []byte(anthropicPerMessageEffortRequest))
	if err != nil {
		t.Fatal(err)
	}
	if len(request.Messages) != 5 || request.Messages[1].ReasoningEffort != "medium" ||
		request.Messages[3].ReasoningEffort != "low" || len(request.Messages[3].Content) != 0 {
		t.Fatalf("per-message effort or empty system message was lost: %+v", request.Messages)
	}
	request.Model = "routed-model"
	request.Generation++
	encoded, err := engine.EncodeRequest(llmprotocol.AnthropicMessagesV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	var same struct {
		Messages []anthropicMessageWire `json:"messages"`
	}
	if err := json.Unmarshal(encoded.Body, &same); err != nil {
		t.Fatal(err)
	}
	if len(same.Messages) != 5 || same.Messages[1].OutputConfig == nil || same.Messages[1].OutputConfig.Effort != "medium" ||
		same.Messages[3].OutputConfig == nil || same.Messages[3].OutputConfig.Effort != "low" || string(same.Messages[3].Content) != "[]" {
		t.Fatalf("Anthropic dispatch lost per-message effort or placement: %s", encoded.Body)
	}

	for _, format := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		t.Run(string(format), func(t *testing.T) {
			projected, encodeErr := engine.EncodeRequest(format, request, envelope)
			if encodeErr != nil {
				t.Fatal(encodeErr)
			}
			if !hasDiagnostic(projected.Diagnostics, "messages[].output_config.effort", llmprotocol.DiagnosticDropped) {
				t.Fatalf("missing per-message effort diagnostic: %+v", projected.Diagnostics)
			}
			var wire map[string]json.RawMessage
			if err := json.Unmarshal(projected.Body, &wire); err != nil {
				t.Fatal(err)
			}
			key := "messages"
			if format == llmprotocol.OpenAIResponsesV1 {
				key = "input"
			}
			var messages []map[string]json.RawMessage
			if err := json.Unmarshal(wire[key], &messages); err != nil {
				t.Fatal(err)
			}
			if len(messages) != 4 {
				t.Fatalf("empty effort-only system message was forwarded to %s: %s", format, projected.Body)
			}
			for _, message := range messages {
				if _, leaked := message["output_config"]; leaked {
					t.Fatalf("per-message effort leaked to %s: %s", format, projected.Body)
				}
			}
		})
	}
}

func TestAnthropicPerMessageEffortRejectsInvalidVariants(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, test := range []struct {
		name, message string
	}{
		{"non-system role", `{"role":"user","content":"hello","output_config":{"effort":"low"}}`},
		{"unknown effort", `{"role":"system","content":[],"output_config":{"effort":"turbo"}}`},
		{"message format", `{"role":"system","content":[],"output_config":{"effort":"low","format":{"type":"json_schema","schema":{}}}}`},
		{"empty without effort", `{"role":"system","content":[]}`},
	} {
		t.Run(test.name, func(t *testing.T) {
			body := []byte(`{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hello"},` + test.message + `]}`)
			if _, _, _, err := engine.DecodeRequestForMutation(llmprotocol.AnthropicMessagesV1, body); err == nil {
				t.Fatalf("invalid per-message effort variant was accepted: %s", body)
			}
		})
	}
}
