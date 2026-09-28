package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestAnthropicContextManagementSurvivesRoutedRequest(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"model":"source","max_tokens":128,"messages":[{"role":"user","content":"hello"}],"context_management":{"edits":[{"type":"clear_thinking_20251015","keep":"last"}]}}`)
	result, err := engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, body,
		func(request *llmprotocol.Request) error { request.Model = "selected"; return nil })
	if err != nil {
		t.Fatal(err)
	}
	var wire anthropicRequestWire
	if unmarshalErr := json.Unmarshal(result.Body, &wire); unmarshalErr != nil {
		t.Fatal(unmarshalErr)
	}
	if wire.Model != "selected" || !bytes.Equal(wire.ContextManagement, []byte(`{"edits":[{"type":"clear_thinking_20251015","keep":"last"}]}`)) {
		t.Fatalf("routed request lost context edits: %s", result.Body)
	}
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		_, translationErr := engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, target, body, nil)
		assertProtocolError(t, translationErr, llmprotocol.ErrorUnsupportedFeature, "lossy_translation")
	}
	policy := llmprotocol.DefaultPolicy()
	policy.LossyFeatures = llmprotocol.LossyAllowWithDiagnostic
	permissive, err := NewEngine(NewBuiltinRegistry(), policy)
	if err != nil {
		t.Fatal(err)
	}
	translated, err := permissive.TranslateRequest(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, body, nil)
	if err != nil || !hasDiagnostic(translated.Diagnostics, "context_management", llmprotocol.DiagnosticApproximated) {
		t.Fatalf("cross-format context edit loss was silent: result=%+v error=%v", translated, err)
	}
	if bytes.Contains(translated.Body, []byte("context_management")) {
		t.Fatalf("Chat target received an Anthropic-only field: %s", translated.Body)
	}
	invalid := bytes.Replace(body, []byte(`"context_management":{"edits":[{"type":"clear_thinking_20251015","keep":"last"}]}`),
		[]byte(`"context_management":[]`), 1)
	_, _, _, err = engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, invalid)
	assertProtocolError(t, err, llmprotocol.ErrorInvalidRequest, "invalid_context_management")
}

func TestAnthropicTelemetryDecodesWithoutWeakeningUnknownFieldChecks(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"id":"msg1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","stop_sequence":null,"context_management":{"applied_edits":[{"type":"clear_thinking_20251015","cleared_input_tokens":10}]},"usage":{"input_tokens":2,"output_tokens":1,"iterations":[{"type":"message","input_tokens":2,"output_tokens":1}]}}`)
	result, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, body, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(result.Body, body) || !hasDiagnostic(result.Diagnostics, "context_management", llmprotocol.DiagnosticDropped) ||
		!hasDiagnostic(result.Diagnostics, "usage.iterations", llmprotocol.DiagnosticDropped) {
		t.Fatalf("same-format response lost telemetry or omission evidence: body=%s diagnostics=%+v", result.Body, result.Diagnostics)
	}
	unknown := bytes.Replace(body, []byte(`"iterations":`), []byte(`"unmodeled_usage":0,"iterations":`), 1)
	_, err = engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, unknown, nil)
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json")
}

func TestAnthropicStreamAcceptsContextAndThinkingTelemetry(t *testing.T) {
	decoder := AnthropicMessagesCodec{}.NewDecoder(llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy())
	frames := []string{
		`{"type":"message_start","message":{"id":"msg1","type":"message","role":"assistant","model":"m","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":2,"output_tokens":0,"iterations":[{"type":"message","input_tokens":2,"output_tokens":0}]}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"thought","estimated_tokens":8}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":2,"output_tokens":1,"iterations":[{"type":"message","input_tokens":2,"output_tokens":1}]},"context_management":{"applied_edits":[]}}`,
		`{"type":"message_stop"}`,
	}
	var diagnostics llmprotocol.Diagnostics
	for _, frame := range frames {
		var event struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal([]byte(frame), &event); err != nil {
			t.Fatal(err)
		}
		_, observed, err := decoder.Push([]byte("event: " + event.Type + "\ndata: " + frame + "\n\n"))
		if err != nil {
			t.Fatalf("%s: %v", event.Type, err)
		}
		diagnostics = append(diagnostics, observed...)
	}
	if _, _, err := decoder.Finalize(nil); err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"stream.message.usage.iterations", "stream.usage.iterations", "stream.context_management"} {
		if !hasDiagnostic(diagnostics, field, llmprotocol.DiagnosticDropped) {
			t.Fatalf("missing %s diagnostic: %+v", field, diagnostics)
		}
	}
}
