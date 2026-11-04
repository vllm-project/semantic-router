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

func TestGatewayKeepaliveAcceptsEmptyChoiceChunksWithOwnIdentity(t *testing.T) {
	decoder := OpenAIChatCodec{}.NewDecoder(llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy())
	frames := []string{
		`{"id":"gw-heartbeat","object":"chat.completion.chunk","created":1770000000,"model":"gw-internal","choices":[]}`,
		`{"id":"chatcmpl-real","object":"chat.completion.chunk","created":1,"model":"real","choices":[{"index":0,"delta":{"role":"assistant","content":"hello"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-real","object":"chat.completion.chunk","created":1,"model":"real","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	}
	var sawOutput bool
	for _, frame := range frames {
		events, _, err := decoder.Push([]byte("data: " + frame + "\n\n"))
		if err != nil {
			t.Fatalf("empty-choice heartbeat must not fail the stream: %v", err)
		}
		for _, event := range events {
			if event.Type == llmprotocol.EventOutputTextDelta && event.Delta == "hello" {
				sawOutput = true
			}
		}
	}
	completed, _, err := decoder.Push([]byte("data: [DONE]\n\n"))
	if err != nil {
		t.Fatal(err)
	}
	if !sawOutput {
		t.Fatal("real chunk after an empty-choice heartbeat lost its output")
	}
	if len(completed) == 0 || completed[len(completed)-1].Type != llmprotocol.EventResponseCompleted {
		t.Fatalf("stream must complete after the heartbeat, got %+v", completed)
	}
}

func TestGatewayEmptyChoiceChunkWithUsageIsNotKeepalive(t *testing.T) {
	decoder := OpenAIChatCodec{}.NewDecoder(llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy())
	frames := []string{
		`{"id":"chatcmpl-real","object":"chat.completion.chunk","created":1,"model":"real","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-real","object":"chat.completion.chunk","created":1,"model":"real","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		`{"id":"chatcmpl-real","object":"chat.completion.chunk","created":1,"model":"real","choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`,
	}
	for _, frame := range frames {
		if _, _, err := decoder.Push([]byte("data: " + frame + "\n\n")); err != nil {
			t.Fatal(err)
		}
	}
	events, _, err := decoder.Push([]byte("data: [DONE]\n\n"))
	if err != nil {
		t.Fatal(err)
	}
	var completed *llmprotocol.Event
	for i := range events {
		if events[i].Type == llmprotocol.EventResponseCompleted {
			completed = &events[i]
		}
	}
	if completed == nil || completed.Usage == nil || completed.Usage.Total.Value == nil || *completed.Usage.Total.Value != 3 {
		t.Fatalf("terminal event must carry the usage reported on the empty-choices chunk, got %+v", completed)
	}
}

// Once a real response has started, only the exact keepalive sentinel stays
// exempt: a post-start empty-choice chunk carrying a different response ID
// must still fail closed through identity observation instead of being
// swallowed as a heartbeat.
func TestGatewayPostStartEmptyChoiceChunkWithForeignIDFailsClosed(t *testing.T) {
	decoder := OpenAIChatCodec{}.NewDecoder(llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy())
	first := `{"id":"chatcmpl-real","object":"chat.completion.chunk","created":1,"model":"real","choices":[{"index":0,"delta":{"role":"assistant","content":"hello"},"finish_reason":null}]}`
	if _, _, err := decoder.Push([]byte("data: " + first + "\n\n")); err != nil {
		t.Fatal(err)
	}
	postStart := `{"id":"chatcmpl-other","object":"chat.completion.chunk","created":1770000000,"model":"gw-internal","choices":[]}`
	_, _, err := decoder.Push([]byte("data: " + postStart + "\n\n"))
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "stream_response_id_mismatch")
}

func TestGatewayPostStartEmptyChoiceChunkAfterGeneratedIDFailsClosed(t *testing.T) {
	decoder := OpenAIChatCodec{}.NewDecoder(llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy())
	first := `{"object":"chat.completion.chunk","created":1,"model":"real","choices":[{"index":0,"delta":{"role":"assistant","content":"hello"},"finish_reason":null}]}`
	if _, _, err := decoder.Push([]byte("data: " + first + "\n\n")); err != nil {
		t.Fatal(err)
	}
	postStart := `{"id":"chatcmpl-other","object":"chat.completion.chunk","created":1770000000,"model":"foreign","choices":[]}`
	_, _, err := decoder.Push([]byte("data: " + postStart + "\n\n"))
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "stream_model_mismatch")
}

// A post-start empty-choice chunk that carries the pinned response ID stays
// benign: it establishes nothing new and the stream completes.
func TestGatewayPostStartEmptyChoiceChunkWithPinnedIDStaysBenign(t *testing.T) {
	decoder := OpenAIChatCodec{}.NewDecoder(llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy())
	frames := []string{
		`{"id":"chatcmpl-real","object":"chat.completion.chunk","created":1,"model":"real","choices":[{"index":0,"delta":{"role":"assistant","content":"hello"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-real","object":"chat.completion.chunk","created":1,"model":"real","choices":[]}`,
		`{"id":"chatcmpl-real","object":"chat.completion.chunk","created":1,"model":"real","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	}
	for _, frame := range frames {
		if _, _, err := decoder.Push([]byte("data: " + frame + "\n\n")); err != nil {
			t.Fatalf("post-start empty-choice chunk with the pinned ID must not fail the stream: %v", err)
		}
	}
	events, _, err := decoder.Push([]byte("data: [DONE]\n\n"))
	if err != nil {
		t.Fatal(err)
	}
	if len(events) == 0 || events[len(events)-1].Type != llmprotocol.EventResponseCompleted {
		t.Fatalf("stream must complete after the post-start empty chunk, got %+v", events)
	}
}

// The exact "chatcmpl-keepalive" sentinel remains exempt after a real
// response has started: aggregator gateways emit it mid-stream, and it must
// not trip identity pinning.
func TestGatewayPostStartExactKeepaliveSentinelStaysExempt(t *testing.T) {
	decoder := OpenAIChatCodec{}.NewDecoder(llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy())
	frames := []string{
		`{"id":"chatcmpl-real","object":"chat.completion.chunk","created":1,"model":"real","choices":[{"index":0,"delta":{"role":"assistant","content":"hello"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-keepalive","object":"chat.completion.chunk","created":0,"model":"keepalive","choices":[{"index":0,"delta":{},"finish_reason":null}]}`,
		`{"id":"chatcmpl-real","object":"chat.completion.chunk","created":1,"model":"real","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	}
	for _, frame := range frames {
		if _, _, err := decoder.Push([]byte("data: " + frame + "\n\n")); err != nil {
			t.Fatalf("post-start exact keepalive sentinel must stay exempt: %v", err)
		}
	}
	events, _, err := decoder.Push([]byte("data: [DONE]\n\n"))
	if err != nil {
		t.Fatal(err)
	}
	if len(events) == 0 || events[len(events)-1].Type != llmprotocol.EventResponseCompleted {
		t.Fatalf("stream must complete after the post-start sentinel, got %+v", events)
	}
}
