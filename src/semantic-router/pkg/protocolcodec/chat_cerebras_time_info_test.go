package protocolcodec

import (
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestChatResponseAcceptsCerebrasTimeInfo(t *testing.T) {
	body := []byte(`{"id":"chatcmpl_1","object":"chat.completion","created":1,"model":"llama3.1-8b","choices":[{"index":0,"message":{"role":"assistant","content":"answer"},"finish_reason":"stop"}],"time_info":{"queue_time":0.05,"prompt_time":0.21,"completion_time":0.44,"total_time":0.7,"created":1760000000},"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`)
	response, _, diagnostics, err := (OpenAIChatCodec{}).DecodeResponse(body, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("Cerebras time_info was rejected: %v", err)
	}
	if response.Usage.Total.Value == nil || *response.Usage.Total.Value != 5 {
		t.Fatalf("usage changed beside time_info: %+v", response.Usage)
	}
	if !hasDiagnostic(diagnostics, "time_info", llmprotocol.DiagnosticDropped) {
		t.Fatalf("time_info was accepted silently: %+v", diagnostics)
	}
}

func TestChatStreamAcceptsCerebrasTimeInfo(t *testing.T) {
	payload := []byte(
		"data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"model\":\"llama3.1-8b\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"answer\"},\"finish_reason\":null}]}\n\n" +
			"data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"model\":\"llama3.1-8b\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"time_info\":{\"queue_time\":0.05,\"prompt_time\":0.21,\"completion_time\":0.44,\"total_time\":0.7,\"created\":1760000000}}\n\n" +
			"data: [DONE]\n\n",
	)
	stream, err := NewBuiltinEngine().NewStream(
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIChatV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
	)
	if err != nil {
		t.Fatal(err)
	}
	_, events, diagnostics, err := stream.Push(payload)
	if err != nil {
		t.Fatalf("Cerebras time_info was rejected on the stream path: %v", err)
	}
	if _, _, _, err := stream.Finalize(nil); err != nil {
		t.Fatal(err)
	}
	text := ""
	for _, event := range events {
		if event.Type == llmprotocol.EventOutputTextDelta {
			text += event.Delta
		}
	}
	if text != "answer" {
		t.Fatalf("stream text lost beside time_info: %q", text)
	}
	if !hasDiagnostic(diagnostics, "stream.time_info", llmprotocol.DiagnosticDropped) {
		t.Fatalf("stream time_info was accepted silently: %+v", diagnostics)
	}
}

func TestChatTimeInfoStaysStrictOnUnknownFields(t *testing.T) {
	body := []byte(`{"id":"chatcmpl_1","object":"chat.completion","created":1,"model":"llama3.1-8b","choices":[{"index":0,"message":{"role":"assistant","content":"answer"},"finish_reason":"stop"}],"timing_breakdown":{"queue_time":0.05}}`)
	if _, _, _, err := (OpenAIChatCodec{}).DecodeResponse(body, llmprotocol.DefaultPolicy()); err == nil {
		t.Fatal("undeclared top-level field was accepted")
	} else if !strings.Contains(err.Error(), "non-canonical") {
		t.Fatalf("unexpected error class: %v", err)
	}
}
