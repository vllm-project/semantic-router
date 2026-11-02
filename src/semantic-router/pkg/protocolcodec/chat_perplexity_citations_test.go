package protocolcodec

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestChatResponseAcceptsPerplexityCitations(t *testing.T) {
	body := []byte(`{"id":"chatcmpl_1","object":"chat.completion","created":1,"model":"sonar","choices":[{"index":0,"message":{"role":"assistant","content":"answer"},"finish_reason":"stop"}],"citations":["https://example.com/a","https://example.com/b"],"search_results":[{"title":"A","url":"https://example.com/a","date":"2026-01-01"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`)
	response, _, diagnostics, err := (OpenAIChatCodec{}).DecodeResponse(body, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("Perplexity citations were rejected: %v", err)
	}
	if response.Usage.Total.Value == nil || *response.Usage.Total.Value != 5 {
		t.Fatalf("usage changed beside the citations: %+v", response.Usage)
	}
	if !hasDiagnostic(diagnostics, "citations", llmprotocol.DiagnosticDropped) {
		t.Fatalf("citations were accepted silently: %+v", diagnostics)
	}
	if !hasDiagnostic(diagnostics, "search_results", llmprotocol.DiagnosticDropped) {
		t.Fatalf("search_results were accepted silently: %+v", diagnostics)
	}
}

func TestChatStreamAcceptsPerplexityCitations(t *testing.T) {
	payload := []byte(
		"data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"model\":\"sonar\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"answer\"},\"finish_reason\":null}]}\n\n" +
			"data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"model\":\"sonar\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"citations\":[\"https://example.com/a\"],\"search_results\":[{\"title\":\"A\",\"url\":\"https://example.com/a\"}]}\n\n" +
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
		t.Fatalf("Perplexity citations were rejected on the stream path: %v", err)
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
		t.Fatalf("stream text lost beside the citations: %q", text)
	}
	if !hasDiagnostic(diagnostics, "stream.citations", llmprotocol.DiagnosticDropped) {
		t.Fatalf("stream citations were accepted silently: %+v", diagnostics)
	}
	if !hasDiagnostic(diagnostics, "stream.search_results", llmprotocol.DiagnosticDropped) {
		t.Fatalf("stream search_results were accepted silently: %+v", diagnostics)
	}
}

func TestChatCitationsStayStrictOnUnknownFields(t *testing.T) {
	body := []byte(`{"id":"chatcmpl_1","object":"chat.completion","created":1,"model":"sonar","choices":[{"index":0,"message":{"role":"assistant","content":"answer"},"finish_reason":"stop"}],"source_links":["https://example.com/a"]}`)
	if _, _, _, err := (OpenAIChatCodec{}).DecodeResponse(body, llmprotocol.DefaultPolicy()); err == nil {
		t.Fatal("undeclared top-level field was accepted")
	}
}
