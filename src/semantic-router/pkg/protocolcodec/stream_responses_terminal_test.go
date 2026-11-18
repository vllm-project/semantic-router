package protocolcodec

import (
	"bytes"
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestResponsesToolStreamInfersToolCallTerminalReason(t *testing.T) {
	engine := NewBuiltinEngine()
	targets := []struct {
		name     string
		format   llmprotocol.WireFormat
		expected []byte
	}{
		{name: "chat", format: llmprotocol.OpenAIChatV1, expected: []byte(`"finish_reason":"tool_calls"`)},
		{name: "anthropic", format: llmprotocol.AnthropicMessagesV1, expected: []byte(`"stop_reason":"tool_use"`)},
	}

	for _, target := range targets {
		t.Run(target.name, func(t *testing.T) {
			assertResponsesToolStreamTerminal(t, engine, target.format, target.expected)
		})
	}
}

func assertResponsesToolStreamTerminal(
	t *testing.T,
	engine *Engine,
	target llmprotocol.WireFormat,
	expected []byte,
) {
	t.Helper()
	stream, err := engine.NewStream(
		llmprotocol.OpenAIResponsesV1,
		target,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
	)
	if err != nil {
		t.Fatal(err)
	}
	frames, events, _, err := stream.Push(toolStreamFixture(llmprotocol.OpenAIResponsesV1))
	if err != nil {
		t.Fatal(err)
	}
	finalFrames, finalEvents, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	frames = append(frames, finalFrames...)
	events = append(events, finalEvents...)
	if output := bytes.Join(frames, nil); !bytes.Contains(output, expected) {
		t.Fatalf("translated terminal did not preserve tool-call semantics: %s", output)
	}
	assertToolCallTerminalEvent(t, events)
}

func assertToolCallTerminalEvent(t *testing.T, events []llmprotocol.Event) {
	t.Helper()
	for _, event := range events {
		if event.Type != llmprotocol.EventResponseCompleted {
			continue
		}
		if event.StopReason != llmprotocol.StopToolCall {
			t.Fatalf("terminal stop reason = %q, want %q", event.StopReason, llmprotocol.StopToolCall)
		}
		return
	}
	t.Fatal("translated stream did not emit a terminal event")
}

func TestResponsesToolResponseInfersToolCallStopReason(t *testing.T) {
	body := []byte(`{
		"id":"response_tool","object":"response","model":"provider-model","status":"completed",
		"output":[{"type":"function_call","id":"item_1","call_id":"call_1","name":"lookup","arguments":"{}","status":"completed"}]
	}`)
	response, _, _, err := (OpenAIResponsesCodec{}).DecodeResponse(body, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	if response.StopReason != llmprotocol.StopToolCall {
		t.Fatalf("stop reason = %q, want %q", response.StopReason, llmprotocol.StopToolCall)
	}
}

func TestResponsesReasoningTerminalCiphertext(t *testing.T) {
	tests := []struct {
		name       string
		mutate     func(completed, terminal map[string]any)
		incomplete bool
		wantError  bool
	}{
		{name: "different ciphertext"},
		{name: "incomplete response", incomplete: true},
		{name: "missing terminal ciphertext", wantError: true, mutate: func(_, terminal map[string]any) { delete(terminal, "encrypted_content") }},
		{name: "missing completed ciphertext", wantError: true, mutate: func(completed, _ map[string]any) { delete(completed, "encrypted_content") }},
		{name: "null ciphertext", wantError: true, mutate: func(_, terminal map[string]any) { terminal["encrypted_content"] = nil }},
		{name: "empty ciphertext", wantError: true, mutate: func(completed, _ map[string]any) { completed["encrypted_content"] = "" }},
		{name: "non-string ciphertext", wantError: true, mutate: func(_, terminal map[string]any) { terminal["encrypted_content"] = 1 }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			frames := responsesReasoningFrames(t)
			response := frames[17]["response"].(map[string]any)
			if test.mutate != nil {
				test.mutate(frames[10]["item"].(map[string]any), response["output"].([]any)[0].(map[string]any))
			}
			wantReason := llmprotocol.StopEndTurn
			if test.incomplete {
				frames[17]["type"] = "response.incomplete"
				response["status"] = "incomplete"
				response["incomplete_details"] = map[string]any{"reason": "max_output_tokens"}
				wantReason = llmprotocol.StopMaxTokens
			}
			events, err := decodeResponsesReasoningFrames(t, frames)
			if test.wantError {
				assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "stream_terminal_output_mismatch")
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			terminal := events[len(events)-1]
			if terminal.Type != llmprotocol.EventResponseCompleted || terminal.StopReason != wantReason {
				t.Fatalf("unexpected terminal event: %+v", terminal)
			}
		})
	}
}

func TestResponsesOpaqueEncryptedReasoning(t *testing.T) {
	frames := responsesReasoningFrames(t)
	completed := frames[10]["item"].(map[string]any)
	terminal := frames[17]["response"].(map[string]any)["output"].([]any)[0].(map[string]any)
	for _, item := range []map[string]any{completed, terminal} {
		item["summary"] = []any{}
		delete(item, "content")
	}
	frames = append(frames[:2], frames[10:]...)
	events, err := decodeResponsesReasoningFrames(t, frames)
	if err != nil {
		t.Fatal(err)
	}
	if events[len(events)-1].Type != llmprotocol.EventResponseCompleted {
		t.Fatal("opaque reasoning prevented stream completion")
	}
}
