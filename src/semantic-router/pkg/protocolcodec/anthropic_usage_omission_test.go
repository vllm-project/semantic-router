package protocolcodec

import (
	"bytes"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A backend that omits usage (aggregators, gateways, or plain OpenAI-compatible
// servers without token accounting) must not turn a successful non-streaming
// completion into a translation failure for Anthropic clients. The Messages
// wire requires usage fields, so the encoder emits the explicit zero-valued
// usage object and records an accounting omission diagnostic instead.
func TestAnthropicBufferedReplyToleratesOmittedUsage(t *testing.T) {
	body := []byte(`{
		"id": "chatcmpl-no-usage",
		"object": "chat.completion",
		"created": 1,
		"model": "test-model",
		"choices": [
			{
				"index": 0,
				"message": {"role": "assistant", "content": "hello"},
				"finish_reason": "stop"
			}
		]
	}`)

	result, err := NewBuiltinEngine().TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, body, nil)
	if err != nil {
		t.Fatalf("TranslateResponse with omitted usage: %v", err)
	}
	if !bytes.Contains(result.Body, []byte(`"usage"`)) {
		t.Fatalf("Anthropic reply must carry a usage object: %s", result.Body)
	}
	if !bytes.Contains(result.Body, []byte(`"input_tokens":0`)) || !bytes.Contains(result.Body, []byte(`"output_tokens":0`)) {
		t.Fatalf("Anthropic reply must emit explicit zero-valued usage: %s", result.Body)
	}
	assertDiagnosticFields(t, result.Diagnostics, "usage")
}

// Same-format OpenAI targets already tolerated omitted usage (the field is
// simply absent from the relayed body); assert that stays true after the
// Anthropic encoder change.
func TestOpenAIChatBufferedReplyToleratesOmittedUsage(t *testing.T) {
	body := []byte(`{
		"id": "chatcmpl-no-usage",
		"object": "chat.completion",
		"created": 1,
		"model": "test-model",
		"choices": [
			{
				"index": 0,
				"message": {"role": "assistant", "content": "hello"},
				"finish_reason": "stop"
			}
		]
	}`)

	result, err := NewBuiltinEngine().TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, body, nil)
	if err != nil {
		t.Fatalf("TranslateResponse same-format with omitted usage: %v", err)
	}
	if !bytes.Contains(result.Body, []byte("hello")) {
		t.Fatalf("same-format reply lost content: %s", result.Body)
	}
}
