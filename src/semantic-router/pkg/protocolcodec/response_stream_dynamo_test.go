package protocolcodec

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestEncodeResponseStreamPreservesDynamoEnvelope(t *testing.T) {
	engine := NewBuiltinEngine()
	decoded, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, []byte(`{"id":"r1","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"nvext":{"token_ids":[1,2]}}`), nil)
	require.NoError(t, err)
	for _, format := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		t.Run(string(format), func(t *testing.T) {
			envelope := decoded.Envelope
			envelope.Format = format
			wire, _, err := engine.EncodeResponseStreamWithEnvelope(format, decoded.Response, envelope, llmprotocol.StreamContext{Source: format, Target: format})
			require.NoError(t, err)
			require.NoError(t, engine.ValidateEncodedResponse(wire, true))
			count := 0
			for _, line := range strings.Split(string(wire), "\n") {
				if !strings.HasPrefix(line, "data: ") || strings.TrimPrefix(line, "data: ") == "[DONE]" {
					continue
				}
				var frame map[string]json.RawMessage
				require.NoError(t, json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &frame))
				if format == llmprotocol.OpenAIResponsesV1 && frame["response"] != nil {
					require.NoError(t, json.Unmarshal(frame["response"], &frame))
				}
				if raw := frame["nvext"]; raw != nil {
					require.JSONEq(t, `{"token_ids":[1,2]}`, string(raw))
					count++
				}
			}
			require.Equal(t, 1, count, "metadata must survive exactly once")
		})
	}
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1} {
		wire, _, err := engine.EncodeResponseStreamWithEnvelope(target, decoded.Response, decoded.Envelope, llmprotocol.StreamContext{Source: target, Target: target})
		require.ErrorContains(t, err, "unsupported_dynamo_nvext_translation")
		require.Empty(t, wire)
	}
}
