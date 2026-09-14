package protocolcodec

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestProviderCodecRetainsUnchangedRequestBytes(t *testing.T) {
	for _, test := range providerPrefixCodecCases() {
		t.Run(string(test.format), func(t *testing.T) {
			policy := llmprotocol.DefaultPolicy()
			request := providerPrefixRequest(1, false)
			baseline, _, err := test.codec.EncodeRequest(request, llmprotocol.Envelope{}, policy)
			if err != nil {
				t.Fatalf("encode baseline request: %v", err)
			}

			envelope := llmprotocol.Envelope{
				Format:     test.format,
				Generation: request.Generation,
				Request:    baseline,
			}
			replayed, _, err := test.codec.EncodeRequest(request, envelope, policy)
			if err != nil {
				t.Fatalf("replay unchanged request: %v", err)
			}
			if !bytes.Equal(replayed, baseline) {
				t.Fatalf("unchanged request was re-encoded:\n got: %s\nwant: %s", replayed, baseline)
			}
		})
	}
}

func TestProviderCodecAppendsToolsWithoutReencodingRetainedDefinition(t *testing.T) {
	for _, test := range providerPrefixCodecCases() {
		t.Run(string(test.format), func(t *testing.T) {
			policy := llmprotocol.DefaultPolicy()
			initial := providerPrefixRequest(1, false)
			baseline, _, err := test.codec.EncodeRequest(initial, llmprotocol.Envelope{}, policy)
			if err != nil {
				t.Fatalf("encode baseline request: %v", err)
			}

			grown := providerPrefixRequest(2, true)
			grownBody, _, err := test.codec.EncodeRequest(grown, llmprotocol.Envelope{
				Format:     test.format,
				Generation: initial.Generation,
				Request:    baseline,
			}, policy)
			if err != nil {
				t.Fatalf("encode grown request: %v", err)
			}

			before := providerToolDefinitions(t, baseline)
			after := providerToolDefinitions(t, grownBody)
			if len(before) != 1 || len(after) != 2 {
				t.Fatalf("unexpected tool counts: before=%d after=%d", len(before), len(after))
			}
			if !bytes.Equal(before[0], after[0]) {
				t.Fatalf("retained provider definition changed:\n got: %s\nwant: %s", after[0], before[0])
			}
		})
	}
}

type providerPrefixCodecCase struct {
	format llmprotocol.WireFormat
	codec  MessageCodec
}

func providerPrefixCodecCases() []providerPrefixCodecCase {
	return []providerPrefixCodecCase{
		{format: llmprotocol.OpenAIChatV1, codec: OpenAIChatCodec{}},
		{format: llmprotocol.OpenAIResponsesV1, codec: OpenAIResponsesCodec{}},
		{format: llmprotocol.AnthropicMessagesV1, codec: AnthropicMessagesCodec{}},
	}
}

func providerPrefixRequest(generation uint64, appendTool bool) llmprotocol.Request {
	request := llmprotocol.Request{
		Generation: generation,
		Model:      "prefix-test-model",
		Messages: []llmprotocol.Message{{
			Role: llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentText,
				Text: "Keep the retained tool definition stable.",
			}},
		}},
		Tools: []llmprotocol.Tool{{
			Name:        "retained_tool",
			Description: "The retained definition.",
			Strict:      providerPrefixBoolPointer(true),
			InputSchema: json.RawMessage(`{"type":"object","properties":{"value":{"type":"string"}}}`),
		}},
	}
	if appendTool {
		request.Tools = append(request.Tools, llmprotocol.Tool{
			Name:        "new_tool",
			Description: "The newly admitted definition.",
			InputSchema: json.RawMessage(`{"type":"object"}`),
		})
	}
	return request
}

func providerPrefixBoolPointer(value bool) *bool {
	return &value
}

func providerToolDefinitions(t *testing.T, body []byte) []json.RawMessage {
	t.Helper()
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(body, &envelope); err != nil {
		t.Fatalf("decode provider body: %v", err)
	}
	var tools []json.RawMessage
	if err := json.Unmarshal(envelope["tools"], &tools); err != nil {
		t.Fatalf("decode provider tools: %v", err)
	}
	return tools
}
