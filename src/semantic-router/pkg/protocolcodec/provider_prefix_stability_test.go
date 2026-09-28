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

func TestProviderCodecAppendsToolsWithExactSerializedPrefix(t *testing.T) {
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

			before := providerToolsJSON(t, baseline)
			after := providerToolsJSON(t, grownBody)
			if len(before) < 2 || before[0] != '[' || before[len(before)-1] != ']' {
				t.Fatalf("baseline tools are not a JSON array: %s", before)
			}
			if len(after) < 2 || after[0] != '[' || after[len(after)-1] != ']' {
				t.Fatalf("grown tools are not a JSON array: %s", after)
			}

			// The retained definitions must remain a byte-for-byte prefix of
			// the provider-visible tools array. Comparing parsed elements alone
			// would miss a re-encoding that changes separators or surrounding
			// bytes and can therefore invalidate a provider prompt-cache prefix.
			expectedPrefix := append([]byte(nil), before[:len(before)-1]...)
			expectedPrefix = append(expectedPrefix, ',')
			if !bytes.HasPrefix(after, expectedPrefix) {
				t.Fatalf("grown tools did not append after the retained serialized prefix:\n got: %s\nwant prefix: %s", after, expectedPrefix)
			}
		})
	}
}

func TestProviderCodecStatelessBaselineUsesCurrentToolOrder(t *testing.T) {
	for _, test := range providerPrefixCodecCases() {
		t.Run(string(test.format), func(t *testing.T) {
			policy := llmprotocol.DefaultPolicy()
			initial := providerPrefixRequest(1, false)
			baseline, _, err := test.codec.EncodeRequest(initial, llmprotocol.Envelope{}, policy)
			if err != nil {
				t.Fatalf("encode baseline request: %v", err)
			}

			stateless := providerPrefixRequest(2, true)
			stateless.Tools[0], stateless.Tools[1] = stateless.Tools[1], stateless.Tools[0]
			body, _, err := test.codec.EncodeRequest(stateless, llmprotocol.Envelope{}, policy)
			if err != nil {
				t.Fatalf("encode stateless request: %v", err)
			}

			before := providerToolDefinitions(t, baseline)
			after := providerToolDefinitions(t, body)
			if len(before) != 1 || len(after) != 2 {
				t.Fatalf("unexpected tool counts: before=%d after=%d", len(before), len(after))
			}
			firstName, secondName := providerToolName(t, after[0]), providerToolName(t, after[1])
			if firstName != "new_tool" || secondName != "retained_tool" {
				t.Fatalf("stateless provider order = [%s %s], want [new_tool retained_tool]", firstName, secondName)
			}
			toolsJSON := providerToolsJSON(t, body)
			baselineToolsJSON := providerToolsJSON(t, baseline)
			if len(baselineToolsJSON) < 2 || baselineToolsJSON[0] != '[' || baselineToolsJSON[len(baselineToolsJSON)-1] != ']' {
				t.Fatalf("baseline tools are not a JSON array: %s", baselineToolsJSON)
			}
			expectedPrefix := append([]byte(nil), baselineToolsJSON[:len(baselineToolsJSON)-1]...)
			expectedPrefix = append(expectedPrefix, ',')
			if bytes.HasPrefix(toolsJSON, expectedPrefix) {
				t.Fatalf("stateless baseline unexpectedly reused the retained provider prefix: %s", toolsJSON)
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
	rawTools := providerToolsJSON(t, body)
	var definitions []json.RawMessage
	if err := json.Unmarshal(rawTools, &definitions); err != nil {
		t.Fatalf("decode provider tools: %v", err)
	}
	return definitions
}

func providerToolsJSON(t *testing.T, body []byte) []byte {
	t.Helper()
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(body, &envelope); err != nil {
		t.Fatalf("decode provider body: %v", err)
	}
	tools, ok := envelope["tools"]
	if !ok || len(tools) == 0 {
		t.Fatalf("provider body does not contain tools: %s", body)
	}
	return append([]byte(nil), tools...)
}

func providerToolName(t *testing.T, raw json.RawMessage) string {
	t.Helper()
	var tool struct {
		Name     string `json:"name"`
		Function *struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if err := json.Unmarshal(raw, &tool); err != nil {
		t.Fatalf("decode provider tool: %v", err)
	}
	if tool.Function != nil && tool.Function.Name != "" {
		return tool.Function.Name
	}
	return tool.Name
}
