package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestProtocolMarkerNamesUsePublicWireTokens(t *testing.T) {
	tests := map[llmprotocol.WireFormat]string{
		llmprotocol.OpenAIChatV1:        "openai",
		llmprotocol.OpenAIResponsesV1:   "responses",
		llmprotocol.AnthropicMessagesV1: "anthropic",
		llmprotocol.OpenAIImagesV1:      "images",
		llmprotocol.OpenAIVideosV1:      "videos",
		"":                              "openai",
	}
	for format, want := range tests {
		if got := normalizeProtocol(string(format)); got != want {
			t.Fatalf("normalizeProtocol(%q) = %q, want %q", format, got, want)
		}
	}
}

// The trace names both wire protocols of every cell, the same-protocol ones
// included, so a debug caller always sees which translation handled the call.
func TestTraceNamesBothProtocolsOfEveryCell(t *testing.T) {
	formats := []llmprotocol.WireFormat{
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIResponsesV1,
		llmprotocol.AnthropicMessagesV1,
	}
	for _, source := range formats {
		for _, target := range formats {
			ctx := debugRequestContext()
			ctx.SourceFormat, ctx.TargetFormat = source, target
			trace := traceMembersOf(t, responseTrace(ctx))
			if trace[headers.TraceClientProtocol] != normalizeProtocol(string(source)) || trace[headers.TraceUpstreamProtocol] != normalizeProtocol(string(target)) {
				t.Fatalf("%s -> %s: trace = %v", source, target, trace)
			}
		}
	}
}
