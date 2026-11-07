package protocolcodec

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Captured 2026-10-21 from vllm serve 0.29.1rc1 with Qwen/Qwen3.5-35B-A3B-FP8 on
// its /v1/messages route: one forced get_weather call, buffered and streamed,
// and the envelope for an unknown model (HTTP 404). The bytes are unedited.
func vllmMessagesCapture(t *testing.T, name string) []byte {
	t.Helper()
	body, err := os.ReadFile(filepath.Join("testdata", "providers", name))
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func TestVLLMMessagesToolUseResponseDecodes(t *testing.T) {
	response, _, _, err := AnthropicMessagesCodec{}.DecodeResponse(
		vllmMessagesCapture(t, "vllm-messages-tool-use-out.json"), llmprotocol.DefaultPolicy(),
	)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if response.StopReason != llmprotocol.StopToolCall || len(response.Output) != 1 {
		t.Fatalf("response = %+v, want one output ending in a tool call", response)
	}
	kinds := []string{}
	var call *llmprotocol.ToolCall
	for _, content := range response.Output[0].Content {
		kinds = append(kinds, string(content.Kind))
		if content.Kind == llmprotocol.ContentToolCall {
			call = content.ToolCall
		}
	}
	if strings.Join(kinds, ",") != "reasoning,tool_call" {
		t.Fatalf("content kinds = %v, want the thinking block then the tool call", kinds)
	}
	if call == nil || call.Name != "get_weather" || call.Arguments != `{"city":"Paris"}` {
		t.Fatalf("tool call = %+v", call)
	}
}

func TestVLLMMessagesToolUseStreamPassesThrough(t *testing.T) {
	out := translateMatchedStopStream(t, llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1,
		string(vllmMessagesCapture(t, "vllm-messages-tool-use-stream.sse")))
	for _, want := range []string{`"name":"get_weather"`, `"partial_json":"{\"city\": \"Paris\"}"`, `"stop_reason":"tool_use"`, "event: message_stop"} {
		if !strings.Contains(out, want) {
			t.Fatalf("translated stream lacks %s:\n%s", want, out)
		}
	}
}

// vLLM 0.11 through 0.18 write "data: [DONE]" after message_stop (message_stream_converter
// in vllm/entrypoints/anthropic), so the capture gets that line appended.
func TestVLLMMessagesStreamAcceptsDoneOnlyAfterMessageStop(t *testing.T) {
	capture := string(vllmMessagesCapture(t, "vllm-messages-tool-use-stream.sse"))
	out := translateMatchedStopStream(t, llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1,
		capture+"data: [DONE]\n\n")
	if !strings.HasSuffix(out, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n") || strings.Contains(out, "[DONE]") {
		t.Fatalf("stream should end at message_stop without the sentinel:\n%s", out)
	}

	stream, err := NewBuiltinEngine().NewStreamWithMutation(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1,
		llmprotocol.StreamContext{Source: llmprotocol.AnthropicMessagesV1, Target: llmprotocol.AnthropicMessagesV1, PublicModel: "m"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	_, _, _, err = stream.Push([]byte(capture + "event: ping\ndata: {\"type\":\"ping\"}\n\n"))
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "stream_event_after_terminal")
}

func TestVLLMMessagesNotFoundEnvelopeDecodes(t *testing.T) {
	transportError, _, err := AnthropicMessagesCodec{}.DecodeTransportError(
		vllmMessagesCapture(t, "vllm-messages-not-found-error-out.json"), llmprotocol.DefaultPolicy(),
	)
	if err != nil {
		t.Fatalf("DecodeTransportError() error = %v", err)
	}
	if transportError.Error == nil || !strings.Contains(transportError.Error.Message, "no-such-model") {
		t.Fatalf("transport error = %+v, want vLLM's message kept", transportError.Error)
	}
}
