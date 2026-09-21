package protocolcodec

import (
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestDynamoRequestRejectsClientTokenData(t *testing.T) {
	for _, format := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		for _, value := range []string{"[0]", "[]", "null"} {
			for _, stream := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/%s/stream=%t", format, value, stream), func(t *testing.T) {
					input := `"messages":[{"role":"user","content":"Tell me about the weather."}]`
					if format == llmprotocol.OpenAIResponsesV1 {
						input = `"input":"Tell me about the weather."`
					}
					// The caller supplies independent tokens, not a verified rendering
					// of the benign text. Empty and null values must not become absence.
					body := []byte(fmt.Sprintf(`{"model":"model-a",%s,"stream":%t,"nvext":{"token_data":%s}}`, input, stream, value))
					engine := NewBuiltinEngine()
					_, _, _, err := engine.DecodeRequestForMutation(format, body)
					assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_dynamo_token_data")
					replayed, err := engine.TranslateRequest(format, format, body, nil)
					assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_dynamo_token_data")
					if len(replayed.Body) != 0 {
						t.Fatal("rejected token_data reached source replay")
					}
					mutated := false
					result, err := engine.TranslateRequest(format, format, body, func(_ *llmprotocol.Request) error {
						mutated = true
						return nil
					})
					assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_dynamo_token_data")
					if mutated || len(result.Body) != 0 {
						t.Fatal("rejected token_data reached mutation or outbound encoding")
					}
				})
			}
		}
	}
}

func TestDynamoRequestWithoutTokenDataAllowsMutation(t *testing.T) {
	for _, format := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		t.Run(string(format), func(t *testing.T) {
			input := `"messages":[{"role":"user","content":"Tell me about the weather."}]`
			if format == llmprotocol.OpenAIResponsesV1 {
				input = `"input":"Tell me about the weather."`
			}
			body := []byte(fmt.Sprintf(`{"model":"model-a",%s,"nvext":{"cache_salt":"tenant-a"}}`, input))
			result, err := NewBuiltinEngine().TranslateRequest(format, format, body, func(request *llmprotocol.Request) error {
				request.Model = "model-b"
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
			assertJSONField(t, result.Body, "model", "model-b")
			assertNestedJSONField(t, result.Body, "nvext", "cache_salt", "tenant-a")
		})
	}
}
