package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestProtocolMarkerNamesUsePublicWireTokens(t *testing.T) {
	tests := map[llmprotocol.WireFormat]string{
		llmprotocol.OpenAIChatV1:        "openai",
		llmprotocol.OpenAIResponsesV1:   "responses",
		llmprotocol.AnthropicMessagesV1: "anthropic",
		"":                              "openai",
	}
	for format, want := range tests {
		if got := normalizeProtocol(string(format)); got != want {
			t.Fatalf("normalizeProtocol(%q) = %q, want %q", format, got, want)
		}
	}
}

func TestProtocolMarkersDescribeEveryCrossProtocolCell(t *testing.T) {
	formats := []llmprotocol.WireFormat{
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIResponsesV1,
		llmprotocol.AnthropicMessagesV1,
	}
	for _, source := range formats {
		for _, target := range formats {
			builder := newResponseHeaderMutationBuilder()
			builder.addProtocolMarkers(&RequestContext{SourceFormat: source, TargetFormat: target})
			mutation := builder.mutation()
			if source == target {
				if mutation != nil {
					t.Fatalf("same-format %s emitted protocol markers", source)
				}
				continue
			}
			if mutation == nil || len(mutation.SetHeaders) != 2 {
				t.Fatalf("cross-format %s -> %s did not emit both protocol markers", source, target)
			}
		}
	}
}
