package protocolcodec

import (
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestAnthropicPartialCacheTranslationDoesNotDoubleCount(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, details := range []string{`{"cached_tokens":3}`, `{"created_cache_tokens":2}`} {
		response := decodeCacheUsageResponse(t, engine, llmprotocol.OpenAIChatV1, cacheUsageFixture(t, llmprotocol.OpenAIChatV1, false, details), false)
		for _, streaming := range []bool{false, true} {
			var body []byte
			var diagnostics llmprotocol.Diagnostics
			var err error
			if streaming {
				body, diagnostics, err = engine.EncodeResponseStream(llmprotocol.AnthropicMessagesV1, response, llmprotocol.StreamContext{PublicModel: response.Model})
			} else {
				result, encodeErr := engine.EncodeResponse(llmprotocol.AnthropicMessagesV1, response, llmprotocol.Envelope{})
				body, diagnostics, err = result.Body, result.Diagnostics, encodeErr
			}
			if err != nil {
				t.Fatal(err)
			}
			assertDiagnosticFields(t, diagnostics, "usage.cache")
			decoded := decodeCacheUsageResponse(t, engine, llmprotocol.AnthropicMessagesV1, body, streaming)
			if tokenValue(decoded.Usage.InputTotal) != 10 || tokenValue(decoded.Usage.Total) != 12 {
				t.Fatalf("partial cache evidence counted twice: %+v", decoded.Usage)
			}
			if response.Usage.InputUncached.Value != nil {
				t.Fatal("client representation changed authoritative settlement evidence")
			}
		}
	}
}

func TestResponsesFailedStreamRejectsConflictingCacheAliases(t *testing.T) {
	body := cacheUsageFixture(t, llmprotocol.OpenAIResponsesV1, true, `{"cache_write_tokens":1,"created_cache_tokens":2}`)
	body = []byte(strings.ReplaceAll(string(body), "response.completed", "response.failed"))
	body = []byte(strings.ReplaceAll(string(body), `"status":"completed","usage":`, `"status":"failed","error":{"code":"provider_error","message":"failed"},"usage":`))
	_, _, err := NewBuiltinEngine().DecodeResponseStream(llmprotocol.OpenAIResponsesV1, body, llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "conflicting_cache_usage")
}
