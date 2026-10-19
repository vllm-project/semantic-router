package protocolcodec

import (
	"bytes"
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const deepSeekChatResponseFixture = `{
  "id":"27471542-6310-4b0c-a94a-d46321e17cdd","object":"chat.completion","created":1792389933,"model":"deepseek-flash",
  "choices":[{"index":0,"message":{"role":"assistant","content":"2","reasoning_content":"We need answer simple."},"logprobs":null,"finish_reason":"stop"}],
  "usage":{"prompt_tokens":34,"completion_tokens":18,"total_tokens":52,
    "prompt_tokens_details":{"cached_tokens":0},
    "completion_tokens_details":{"reasoning_tokens":16},
    "prompt_cache_hit_tokens":0,"prompt_cache_miss_tokens":34},
  "system_fingerprint":"e72d3c7f663d3e1383b00b3884cb4501"
}`

const deepSeekChatResponseFlatOnlyFixture = `{
  "id":"27471542-6310-4b0c-a94a-d46321e17cdd","object":"chat.completion","created":1792389933,"model":"deepseek-flash",
  "choices":[{"index":0,"message":{"role":"assistant","content":"2"},"logprobs":null,"finish_reason":"stop"}],
  "usage":{"prompt_tokens":34,"completion_tokens":18,"total_tokens":52,
    "prompt_cache_hit_tokens":10,"prompt_cache_miss_tokens":24},
  "system_fingerprint":"e72d3c7f663d3e1383b00b3884cb4501"
}`

const deepSeekChatStreamUsageChunk = `data: {"id":"8921111a-f3db-44b4-b188-3fcaa54710eb",` +
	`"object":"chat.completion.chunk","created":1792389809,"model":"deepseek-flash",` +
	`"system_fingerprint":"e72d3c7f663d3e1383b00b3884cb4501",` +
	`"choices":[{"index":0,"delta":{"content":""},"logprobs":null,"finish_reason":"stop"}],` +
	`"usage":{"prompt_tokens":5,"completion_tokens":9,"total_tokens":14,` +
	`"prompt_tokens_details":{"cached_tokens":0},` +
	`"prompt_cache_hit_tokens":0,"prompt_cache_miss_tokens":5}}` + "\n\n" +
	`data: [DONE]` + "\n\n"

func TestOpenAIChatResponseAcceptsDeepSeekPromptCacheTokens(t *testing.T) {
	response, _, _, err := NewBuiltinEngine().DecodeResponse(llmprotocol.OpenAIChatV1, []byte(deepSeekChatResponseFixture))
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if response.Usage.Total.Value == nil || *response.Usage.Total.Value != 52 {
		t.Fatalf("usage.Total = %+v, want 52", response.Usage.Total)
	}
	if response.Usage.InputTotal.Value == nil || *response.Usage.InputTotal.Value != 34 {
		t.Fatalf("usage.InputTotal = %+v, want 34", response.Usage.InputTotal)
	}
	if response.Usage.OutputTotal.Value == nil || *response.Usage.OutputTotal.Value != 18 {
		t.Fatalf("usage.OutputTotal = %+v, want 18", response.Usage.OutputTotal)
	}
	if response.Usage.InputCacheRead.Value == nil || *response.Usage.InputCacheRead.Value != 0 {
		t.Fatalf("usage.InputCacheRead = %+v, want 0", response.Usage.InputCacheRead)
	}
	if response.Usage.InputUncached.Value == nil || *response.Usage.InputUncached.Value != 34 {
		t.Fatalf("usage.InputUncached = %+v, want 34", response.Usage.InputUncached)
	}
	if response.Usage.OutputReasoning.Value == nil || *response.Usage.OutputReasoning.Value != 16 {
		t.Fatalf("usage.OutputReasoning = %+v, want 16", response.Usage.OutputReasoning)
	}
	if response.Usage.OutputOther.Value == nil || *response.Usage.OutputOther.Value != 2 {
		t.Fatalf("usage.OutputOther = %+v, want 2", response.Usage.OutputOther)
	}
}

func TestOpenAIChatResponsePromptCacheTokensFallback(t *testing.T) {
	response, _, _, err := NewBuiltinEngine().DecodeResponse(llmprotocol.OpenAIChatV1, []byte(deepSeekChatResponseFlatOnlyFixture))
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if response.Usage.InputCacheRead.Value == nil || *response.Usage.InputCacheRead.Value != 10 {
		t.Fatalf("usage.InputCacheRead = %+v, want 10", response.Usage.InputCacheRead)
	}
	if response.Usage.InputUncached.Value == nil || *response.Usage.InputUncached.Value != 24 {
		t.Fatalf("usage.InputUncached = %+v, want 24", response.Usage.InputUncached)
	}
}

func TestOpenAIChatResponseTranslatesWithoutLeakingDeepSeekExtensions(t *testing.T) {
	for _, target := range []llmprotocol.WireFormat{
		llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1,
	} {
		t.Run(string(target), func(t *testing.T) {
			result, err := NewBuiltinEngine().TranslateResponse(llmprotocol.OpenAIChatV1, target, []byte(deepSeekChatResponseFixture), nil)
			if err != nil {
				t.Fatalf("TranslateResponse(%s): %v", target, err)
			}
			forbidden := []string{`"prompt_cache_hit_tokens"`, `"prompt_cache_miss_tokens"`}
			for _, field := range forbidden {
				if bytes.Contains(result.Body, []byte(field)) {
					t.Errorf("public %s reply leaked %s: %s", target, field, result.Body)
				}
			}
		})
	}
}

func TestOpenAIChatStreamAcceptsDeepSeekPromptCacheTokens(t *testing.T) {
	engine := NewBuiltinEngine()
	stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, llmprotocol.StreamContext{
		Context:     context.Background(),
		PublicModel: "deepseek-flash",
	})
	if err != nil {
		t.Fatalf("NewStream() error = %v", err)
	}
	_, events, _, err := stream.Push([]byte(deepSeekChatStreamUsageChunk))
	if err != nil {
		t.Fatalf("stream.Push() error = %v", err)
	}
	var usageEvent *llmprotocol.Event
	for _, event := range events {
		if event.Type == llmprotocol.EventUsageUpdated {
			usageEvent = &event
			break
		}
	}
	if usageEvent == nil || usageEvent.Usage == nil {
		t.Fatalf("expected usage.updated event with usage, got: %+v", events)
	}
	if usageEvent.Usage.Total.Value == nil || *usageEvent.Usage.Total.Value != 14 {
		t.Errorf("usageEvent.Usage.Total = %+v, want 14", usageEvent.Usage.Total)
	}
	if usageEvent.Usage.InputUncached.Value == nil || *usageEvent.Usage.InputUncached.Value != 5 {
		t.Errorf("usageEvent.Usage.InputUncached = %+v, want 5", usageEvent.Usage.InputUncached)
	}
	if usageEvent.Usage.InputCacheRead.Value == nil || *usageEvent.Usage.InputCacheRead.Value != 0 {
		t.Errorf("usageEvent.Usage.InputCacheRead = %+v, want 0", usageEvent.Usage.InputCacheRead)
	}
	_, finalEvents, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatalf("stream.Finalize() error = %v", err)
	}
	var completedEvent *llmprotocol.Event
	for _, event := range finalEvents {
		if event.Type == llmprotocol.EventResponseCompleted {
			completedEvent = &event
			break
		}
	}
	if completedEvent == nil {
		t.Fatalf("expected response.completed event, got: %+v", finalEvents)
	}
}
