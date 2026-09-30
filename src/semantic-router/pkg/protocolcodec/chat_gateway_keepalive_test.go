package protocolcodec

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestGatewayKeepaliveDoesNotEstablishChatStreamIdentity(t *testing.T) {
	decoder := OpenAIChatCodec{}.NewDecoder(llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy())
	frames := []string{
		`{"id":"chatcmpl-keepalive","object":"chat.completion.chunk","created":0,"model":"keepalive","choices":[{"index":0,"delta":{},"finish_reason":null}]}`,
		`{"id":"chatcmpl-real","object":"chat.completion.chunk","created":1,"model":"real","choices":[{"index":0,"delta":{"role":"assistant","content":"hello"},"finish_reason":null}],"agent":{"pool":"a"}}`,
		`{"id":"chatcmpl-real","object":"chat.completion.chunk","created":1,"model":"real","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	}
	var sawAgentDiagnostic bool
	var sawOutput bool
	for _, frame := range frames {
		events, diagnostics, err := decoder.Push([]byte("data: " + frame + "\n\n"))
		if err != nil {
			t.Fatal(err)
		}
		for _, event := range events {
			if event.Type == llmprotocol.EventOutputTextDelta && event.Delta == "hello" {
				sawOutput = true
			}
		}
		sawAgentDiagnostic = sawAgentDiagnostic || hasDiagnostic(diagnostics, "stream.agent", llmprotocol.DiagnosticDropped)
	}
	if _, _, err := decoder.Push([]byte("data: [DONE]\n\n")); err != nil {
		t.Fatal(err)
	}
	if !sawOutput || !sawAgentDiagnostic {
		t.Fatalf("gateway stream missing output or agent diagnostic: output=%t diagnostic=%t", sawOutput, sawAgentDiagnostic)
	}
}

func TestGatewayKeepaliveRequiresExactEmptyShape(t *testing.T) {
	decoder := OpenAIChatCodec{}.NewDecoder(llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy())
	first := `{"id":"chatcmpl-keepalive","object":"chat.completion.chunk","created":0,"model":"keepalive","choices":[{"index":0,"delta":{"content":"real"},"finish_reason":null}]}`
	if _, _, err := decoder.Push([]byte("data: " + first + "\n\n")); err != nil {
		t.Fatal(err)
	}
	second := `{"id":"chatcmpl-real","object":"chat.completion.chunk","created":1,"model":"real","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`
	_, _, err := decoder.Push([]byte("data: " + second + "\n\n"))
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "stream_response_id_mismatch")
}
