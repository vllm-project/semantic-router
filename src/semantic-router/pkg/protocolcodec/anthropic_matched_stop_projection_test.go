package protocolcodec

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const anthropicMatchedStopBuffered = `{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"ALPHA BRAVO "}],` + `"stop_reason":"stop_sequence","stop_sequence":"CHARLIE","usage":{"input_tokens":3,"output_tokens":2}}`

const anthropicMatchedStopEnded = `{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"ALPHA BRAVO "}],` + `"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":2}}`

const anthropicMatchedStopStream = "event: message_start\ndata: " + `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}` + "\n\n" +
	"event: content_block_start\ndata: " + `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n" +
	"event: content_block_delta\ndata: " + `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ALPHA BRAVO "}}` + "\n\n" +
	"event: content_block_stop\ndata: " + `{"type":"content_block_stop","index":0}` + "\n\n" +
	"event: message_delta\ndata: " + `{"type":"message_delta","delta":{"stop_reason":"stop_sequence","stop_sequence":"CHARLIE"},"usage":{"output_tokens":1}}` + "\n\n" +
	"event: message_stop\ndata: " + `{"type":"message_stop"}` + "\n\n"

func assertMatchedStopDroppedDiagnostic(t *testing.T, diagnostics llmprotocol.Diagnostics, field string) {
	t.Helper()
	for _, diagnostic := range diagnostics {
		if diagnostic.Field == field && diagnostic.Action == llmprotocol.DiagnosticDropped &&
			strings.Contains(diagnostic.Reason, "matched stop sequence") {
			return
		}
	}
	t.Fatalf("expected a %q dropped diagnostic, got %+v", field, diagnostics)
}

// An Anthropic reply that ends on a stop sequence must reach clients whose
// format cannot carry the matched string instead of failing the whole response:
// the terminal projects to the generic stop with a diagnostic, mirroring how
// #4391 only reports a Chat source's matched sequence when the target can
// carry it.
func TestAnthropicMatchedStopSequenceProjectsToGenericStop(t *testing.T) {
	engine := NewBuiltinEngine()

	chat, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, []byte(anthropicMatchedStopBuffered), nil)
	if err != nil {
		t.Fatalf("chat target failed: %v", err)
	}
	if !bytes.Contains(chat.Body, []byte(`"finish_reason":"stop"`)) {
		t.Fatalf("chat terminal = %s", chat.Body)
	}
	if bytes.Contains(chat.Body, []byte("CHARLIE")) {
		t.Fatalf("chat body must not carry the matched string: %s", chat.Body)
	}
	assertMatchedStopDroppedDiagnostic(t, chat.Diagnostics, "matched_stop_sequence")

	responses, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1, []byte(anthropicMatchedStopBuffered), nil)
	if err != nil {
		t.Fatalf("responses target failed: %v", err)
	}
	if !strings.Contains(string(responses.Body), `"status":"completed"`) {
		t.Fatalf("responses terminal = %s", responses.Body)
	}
	assertMatchedStopDroppedDiagnostic(t, responses.Diagnostics, "matched_stop_sequence")

	// The end-turn control must keep translating without a projection.
	ended, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, []byte(anthropicMatchedStopEnded), nil)
	if err != nil {
		t.Fatalf("end_turn control failed: %v", err)
	}
	if !bytes.Contains(ended.Body, []byte(`"finish_reason":"stop"`)) {
		t.Fatalf("end_turn terminal = %s", ended.Body)
	}
	for _, diagnostic := range ended.Diagnostics {
		if diagnostic.Field == "matched_stop_sequence" {
			t.Fatalf("end_turn must not project a matched stop: %+v", diagnostic)
		}
	}

	// Same-format translation keeps the exact terminal reason.
	sameFormat, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, []byte(anthropicMatchedStopBuffered), nil)
	if err != nil {
		t.Fatalf("same-format target failed: %v", err)
	}
	if !bytes.Contains(sameFormat.Body, []byte(`"stop_reason":"stop_sequence"`)) ||
		!bytes.Contains(sameFormat.Body, []byte(`"stop_sequence":"CHARLIE"`)) {
		t.Fatalf("same-format terminal lost the matched stop sequence: %s", sameFormat.Body)
	}
}

// A stream that ends on a stop sequence must finish with a success terminal,
// not an error frame after the content.
func TestAnthropicMatchedStopSequenceStreamProjectsToGenericStop(t *testing.T) {
	engine := NewBuiltinEngine()
	stream, err := engine.NewStreamWithMutation(
		llmprotocol.AnthropicMessagesV1,
		llmprotocol.OpenAIChatV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "m"},
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	frames, _, _, err := stream.Push([]byte(anthropicMatchedStopStream))
	if err != nil {
		t.Fatal(err)
	}
	finalFrames, _, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	output := bytes.Join(append(frames, finalFrames...), nil)
	if !bytes.Contains(output, []byte(`"finish_reason":"stop"`)) {
		t.Fatalf("stream terminal = %s", output)
	}
	if bytes.Contains(output, []byte("CHARLIE")) {
		t.Fatalf("stream must not carry the matched string: %s", output)
	}
	if bytes.Contains(output, []byte(`"error"`)) {
		t.Fatalf("stream must not end on an error frame: %s", output)
	}

	// The same stream into an Anthropic client keeps the exact terminal.
	anthropicStream, err := engine.NewStreamWithMutation(
		llmprotocol.AnthropicMessagesV1,
		llmprotocol.AnthropicMessagesV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "m"},
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	anthropicFrames, _, anthropicDiagnostics, err := anthropicStream.Push([]byte(anthropicMatchedStopStream))
	if err != nil {
		t.Fatal(err)
	}
	anthropicFinal, _, _, err := anthropicStream.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	anthropicOutput := bytes.Join(append(anthropicFrames, anthropicFinal...), nil)
	if !bytes.Contains(anthropicOutput, []byte(`"stop_reason":"stop_sequence"`)) ||
		!bytes.Contains(anthropicOutput, []byte(`"stop_sequence":"CHARLIE"`)) {
		t.Fatalf("anthropic stream lost the matched stop sequence: %s", anthropicOutput)
	}
	for _, diagnostic := range anthropicDiagnostics {
		if diagnostic.Field == "matched_stop_sequence" {
			t.Fatalf("same-format stream must not project: %+v", diagnostic)
		}
	}
}
