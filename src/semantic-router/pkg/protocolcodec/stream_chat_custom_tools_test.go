package protocolcodec

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const (
	chatCustomCallStart   = `{"index":0,"id":"call_1","type":"custom","custom":{"name":"apply_patch","input":"abc"}}`
	chatFunctionCallStart = `{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}`
)

func TestChatStreamRejectsToolKindSwitch(t *testing.T) {
	for _, test := range []struct {
		name  string
		calls []string
	}{
		{"custom_then_explicit_function", []string{chatCustomCallStart, `{"index":0,"type":"function","function":{"arguments":"{\"x\":1}"}}`}},
		{"custom_then_untyped_function", []string{chatCustomCallStart, `{"index":0,"function":{"arguments":"{\"x\":1}"}}`}},
		{"function_then_custom", []string{chatFunctionCallStart, `{"index":0,"custom":{"input":"abc"}}`}},
	} {
		t.Run(test.name, func(t *testing.T) {
			frames, _, err := pushChatToolCallStream(t, test.calls...)
			if err == nil {
				t.Fatalf("kind switch was accepted and re-emitted:\n%s", bytes.Join(frames, nil))
			}
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "stream_tool_identity_mismatch")
		})
	}
}

func TestChatStreamAccumulatesUntypedCustomDeltas(t *testing.T) {
	_, events, err := pushChatToolCallStream(t,
		chatCustomCallStart,
		`{"index":0,"custom":{"input":"def"}}`,
		`{"index":0}`,
		`{"index":0,"custom":{"input":"ghi"}}`,
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, event := range events {
		if event.Type != llmprotocol.EventOutputItemCompleted || event.ToolCall == nil {
			continue
		}
		if event.ToolCall.Kind != llmprotocol.ToolKindCustom || event.ToolCall.Arguments != "abcdefghi" {
			t.Fatalf("completed call = %+v, want custom input abcdefghi", *event.ToolCall)
		}
		return
	}
	t.Fatal("stream did not complete the custom call")
}

func pushChatToolCallStream(t *testing.T, calls ...string) ([][]byte, []llmprotocol.Event, error) {
	t.Helper()
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model", ProviderModel: "provider-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	var payload strings.Builder
	for index, call := range calls {
		finish := "null"
		if index == len(calls)-1 {
			finish = `"tool_calls"`
		}
		fmt.Fprintf(&payload, `data: {"id":"chatcmpl_kind","object":"chat.completion.chunk","model":"provider-model",`+
			`"choices":[{"index":0,"delta":{"tool_calls":[%s]},"finish_reason":%s}]}`+"\n\n", call, finish)
	}
	frames, events, _, err := stream.Push([]byte(payload.String()))
	return frames, events, err
}
