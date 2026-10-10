package protocolcodec

import (
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestOpenAITransportErrorAcceptsVLLMHTTPStatusCodes(t *testing.T) {
	for _, source := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1} {
			t.Run(string(source)+"/"+string(target), func(t *testing.T) {
				body := []byte(`{"error":{"message":"Maximum context length is 32768 tokens; reduce the input or requested output.","type":"BadRequestError","param":"input_tokens","code":400}}`)
				translated, err := NewBuiltinEngine().TranslateTransportError(source, target, body, nil)
				if err != nil {
					t.Fatal(err)
				}
				actual := translated.TransportError.Error
				if actual.Category != llmprotocol.ErrorInvalidRequest || actual.Code != "400" || actual.Parameter != "input_tokens" || actual.Message != "Maximum context length is 32768 tokens; reduce the input or requested output." {
					t.Fatalf("lost upstream context error: %+v", actual)
				}
				roundtrip, err := NewBuiltinEngine().TranslateTransportError(target, target, translated.Body, nil)
				if err != nil || roundtrip.TransportError.Error.Category != llmprotocol.ErrorInvalidRequest || roundtrip.TransportError.Error.Message != actual.Message {
					t.Fatalf("public error invalid: %s (%v)", translated.Body, err)
				}
			})
		}
	}
}

func TestOpenAITransportErrorCodeRetainsStrictValidation(t *testing.T) {
	for _, code := range []string{`true`, `{}`, `[]`, `400.5`, `400.0`, `399`, `600`, `1e400`} {
		t.Run(code, func(t *testing.T) {
			body := []byte(fmt.Sprintf(`{"error":{"type":"BadRequestError","code":%s,"message":"invalid"}}`, code))
			if _, err := NewBuiltinEngine().TranslateTransportError(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, body, nil); err == nil {
				t.Fatal("malformed code accepted")
			}
		})
	}
	for _, code := range []string{`null`, `"context_length_exceeded"`, `400`, `500`} {
		body := []byte(fmt.Sprintf(`{"error":{"type":"BadRequestError","code":%s,"message":"invalid"}}`, code))
		if _, err := NewBuiltinEngine().TranslateTransportError(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, body, nil); err != nil {
			t.Fatalf("valid code %s rejected: %v", code, err)
		}
	}
}
