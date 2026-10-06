package protocolcodec

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestChatUsageServiceTierCompatibility(t *testing.T) {
	for _, tier := range []string{"", "standard", "priority"} {
		name := tier
		if name == "" {
			name = "canonical_control"
		}
		t.Run(name, func(t *testing.T) {
			extra := ""
			if tier != "" {
				extra = `,"service_tier":"` + tier + `"`
			}
			body := []byte(`{"id":"c1","object":"chat.completion","created":1,"model":"mistral","choices":[{"index":0,"message":{"role":"assistant","content":"Hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2` + extra + `}}`)
			out, err := NewBuiltinEngine().TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, body, nil)
			t.Logf("buffered error=%v output=%s", err, out.Body)
			if err != nil {
				t.Errorf("chat completion should be accepted: %v", err)
			}
			decoder := OpenAIChatCodec{}.NewDecoder(llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy())
			chunk := []byte(`data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"mistral","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2` + extra + `}}` + "\n\n")
			_, _, err = decoder.Push(chunk)
			t.Logf("stream error=%v", err)
			if err != nil {
				t.Errorf("stream completion should be accepted: %v", err)
			}
		})
	}
}

func TestChatUsageServiceTierRejectsInvalidValues(t *testing.T) {
	for _, value := range []string{`"unknown"`, `1`, `{}`} {
		body := []byte(`{"id":"c1","object":"chat.completion","created":1,"model":"mistral","choices":[{"index":0,"message":{"role":"assistant","content":"Hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"service_tier":` + value + `}}`)
		_, err := NewBuiltinEngine().TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, body, nil)
		assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json")
	}
}
