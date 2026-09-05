package protocolcodec

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// When the upstream Anthropic API honors a context_management request
// directive, the Messages response reports the result in a top-level
// context_management object. The strict provider decode must accept it, or the
// router turns a successful backend response into an upstream failure on
// exactly the requests whose directive it just forwarded. Same-format replay
// returns the field to the client verbatim; a re-encoded response cannot
// represent it and records an explicit omission diagnostic, matching the
// container and stop_details handling.

// appliedContextManagement is the response-side field under test: the applied
// edit report the upstream emits after trimming the billed prompt.
const appliedContextManagement = `"context_management":{"applied_edits":[{"type":"clear_thinking_20251015","cleared_input_tokens":4096,"cleared_tool_uses":0}]}`

func anthropicResponseBodyWithContextManagement() []byte {
	return []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"provider-model",` +
		`"content":[{"type":"text","text":"done"}],"stop_reason":"end_turn","stop_sequence":null,` +
		appliedContextManagement + `,` +
		`"usage":{"input_tokens":10,"output_tokens":6}}`)
}

func TestAnthropicResponseContextManagementDecodesAndReplays(t *testing.T) {
	engine := NewBuiltinEngine()
	response, envelope, diagnostics, err := engine.DecodeResponse(
		llmprotocol.AnthropicMessagesV1, anthropicResponseBodyWithContextManagement(),
	)
	if err != nil {
		t.Fatalf("response carrying context_management was rejected: %v", err)
	}
	assertDiagnosticFields(t, diagnostics, "context_management")

	// The generation is unchanged, so the same-format encode replays the
	// captured provider bytes and the client still receives the applied edits.
	replayed, err := engine.EncodeResponse(llmprotocol.AnthropicMessagesV1, response, envelope)
	if err != nil {
		t.Fatalf("replaying the response failed: %v", err)
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(replayed.Body, &object); err != nil {
		t.Fatalf("replayed body is not a JSON object: %v\n%s", err, replayed.Body)
	}
	if _, present := object["context_management"]; !present {
		t.Fatalf("replay dropped context_management: %s", replayed.Body)
	}
}

func TestAnthropicResponseContextManagementReEncodeOmitsExplicitly(t *testing.T) {
	engine := NewBuiltinEngine()
	response, envelope, _, err := engine.DecodeResponse(
		llmprotocol.AnthropicMessagesV1, anthropicResponseBodyWithContextManagement(),
	)
	if err != nil {
		t.Fatalf("response carrying context_management was rejected: %v", err)
	}
	// Bumping the generation forces the encode to rebuild the wire from the
	// neutral response, which has no representation for the applied-edit report.
	response.Generation++
	reEncoded, err := engine.EncodeResponse(llmprotocol.AnthropicMessagesV1, response, envelope)
	if err != nil {
		t.Fatalf("re-encoding the response failed: %v", err)
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(reEncoded.Body, &object); err != nil {
		t.Fatalf("re-encoded body is not a JSON object: %v\n%s", err, reEncoded.Body)
	}
	if _, present := object["context_management"]; present {
		t.Fatalf("re-encode invented context_management without a source: %s", reEncoded.Body)
	}
}

func TestAnthropicStreamMessageStartAcceptsContextManagement(t *testing.T) {
	decoder := AnthropicMessagesCodec{}.NewDecoder(
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
		llmprotocol.DefaultPolicy(),
	)
	start, err := encodeSSE("message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": "msg_1", "type": "message", "role": "assistant", "model": "provider-model",
			"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
			"context_management": map[string]any{"applied_edits": []any{
				map[string]any{"type": "clear_thinking_20251015", "cleared_input_tokens": 4096, "cleared_tool_uses": 0},
			}},
			"usage": map[string]any{"input_tokens": 3, "output_tokens": 0},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	events, _, err := decoder.Push(start)
	if err != nil {
		t.Fatalf("streamed message_start carrying context_management was rejected: %v", err)
	}
	if len(events) != 1 || events[0].Type != llmprotocol.EventResponseStarted {
		t.Fatalf("message_start did not produce a response-started event: %+v", events)
	}
}
