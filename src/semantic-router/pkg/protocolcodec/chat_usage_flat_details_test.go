package protocolcodec

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestChatStreamAcceptsFlatUsageDetailFields(t *testing.T) {
	payload := []byte(
		"data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"first \"},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5,\"cached_tokens\":1,\"reasoning_tokens\":1}}\n\n" +
			"data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"second\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5,\"cached_tokens\":1,\"reasoning_tokens\":1}}\n\n" +
			"data: [DONE]\n\n",
	)
	stream, err := NewBuiltinEngine().NewStream(
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIChatV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
	)
	if err != nil {
		t.Fatal(err)
	}
	_, events, _, err := stream.Push(payload)
	if err != nil {
		t.Fatal(err)
	}
	_, finalEvents, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	text, terminal := "", 0
	var usage *llmprotocol.Usage
	for _, event := range append(events, finalEvents...) {
		switch event.Type {
		case llmprotocol.EventOutputTextDelta:
			text += event.Delta
		case llmprotocol.EventResponseCompleted:
			terminal++
			usage = event.Usage
		}
	}
	if text != "first second" || terminal != 1 || usage == nil {
		t.Fatalf("flat usage detail stream lost its terminal usage: text=%q terminals=%d usage=%+v", text, terminal, usage)
	}
	if usage.InputCacheRead.Value == nil || *usage.InputCacheRead.Value != 1 {
		t.Fatalf("flat cached_tokens not decoded as input cache read: %+v", usage.InputCacheRead)
	}
	if usage.OutputReasoning.Value == nil || *usage.OutputReasoning.Value != 1 {
		t.Fatalf("flat reasoning_tokens not decoded as output reasoning: %+v", usage.OutputReasoning)
	}
	if usage.OutputOther.Value == nil || *usage.OutputOther.Value != 1 {
		t.Fatalf("derived output-other missing beside flat reasoning_tokens: %+v", usage.OutputOther)
	}
	if usage.InputTotal.Value == nil || *usage.InputTotal.Value != 3 ||
		usage.OutputTotal.Value == nil || *usage.OutputTotal.Value != 2 ||
		usage.Total.Value == nil || *usage.Total.Value != 5 {
		t.Fatalf("flat detail stream totals wrong: %+v", usage)
	}
}

func TestChatResponseAcceptsFlatUsageDetailFields(t *testing.T) {
	body := []byte(`{"id":"chatcmpl_1","object":"chat.completion","created":1,"model":"provider-model","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5,"cached_tokens":1,"reasoning_tokens":1}}`)
	response, _, _, err := (OpenAIChatCodec{}).DecodeResponse(body, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	usage := response.Usage
	if usage.InputCacheRead.Value == nil || *usage.InputCacheRead.Value != 1 ||
		usage.OutputReasoning.Value == nil || *usage.OutputReasoning.Value != 1 ||
		usage.OutputOther.Value == nil || *usage.OutputOther.Value != 1 ||
		usage.Total.Value == nil || *usage.Total.Value != 5 {
		t.Fatalf("buffered flat usage detail fields decoded wrong: %+v", usage)
	}
}

func TestChatFlatUsageDetailsCanonicalizeOnSameFormatReplay(t *testing.T) {
	body := []byte(`{"id":"chatcmpl_1","object":"chat.completion","created":1,"model":"provider-model","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5,"cached_tokens":1,"reasoning_tokens":1}}`)
	engine, err := NewEngine(NewBuiltinRegistry(), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	result, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, body, nil)
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Usage struct {
			CachedTokens        *int64 `json:"cached_tokens"`
			ReasoningTokens     *int64 `json:"reasoning_tokens"`
			PromptTokensDetails *struct {
				CachedTokens *int64 `json:"cached_tokens"`
			} `json:"prompt_tokens_details"`
			CompletionTokensDetails *struct {
				ReasoningTokens *int64 `json:"reasoning_tokens"`
			} `json:"completion_tokens_details"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(result.Body, &wire); err != nil {
		t.Fatal(err)
	}
	if wire.Usage.CachedTokens != nil || wire.Usage.ReasoningTokens != nil {
		t.Fatalf("same-format reply replayed flat usage detail fields: %s", result.Body)
	}
	if wire.Usage.PromptTokensDetails == nil || wire.Usage.PromptTokensDetails.CachedTokens == nil ||
		*wire.Usage.PromptTokensDetails.CachedTokens != 1 {
		t.Fatalf("flat cached_tokens missing from canonical nested details: %s", result.Body)
	}
	if wire.Usage.CompletionTokensDetails == nil || wire.Usage.CompletionTokensDetails.ReasoningTokens == nil ||
		*wire.Usage.CompletionTokensDetails.ReasoningTokens != 1 {
		t.Fatalf("flat reasoning_tokens missing from canonical nested details: %s", result.Body)
	}
}

func TestChatUsageNestedDetailsWinOverFlatFields(t *testing.T) {
	body := []byte(`{"id":"chatcmpl_1","object":"chat.completion","created":1,"model":"provider-model","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5,"cached_tokens":9,"reasoning_tokens":9,"prompt_tokens_details":{"cached_tokens":1},"completion_tokens_details":{"reasoning_tokens":1}}}`)
	response, _, _, err := (OpenAIChatCodec{}).DecodeResponse(body, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	usage := response.Usage
	if usage.InputCacheRead.Value == nil || *usage.InputCacheRead.Value != 1 ||
		usage.OutputReasoning.Value == nil || *usage.OutputReasoning.Value != 1 {
		t.Fatalf("nested details must win over flat fields: %+v", usage)
	}
}

func TestChatUsageFlatDetailFieldsStayStrictOnBadShapes(t *testing.T) {
	for name, usage := range map[string]string{
		"string_cached":    `"cached_tokens":"many"`,
		"object_reasoning": `"reasoning_tokens":{"value":1}`,
	} {
		t.Run(name, func(t *testing.T) {
			body := []byte(`{"id":"chatcmpl_1","object":"chat.completion","created":1,"model":"provider-model","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5,` + usage + `}}`)
			if _, _, _, err := (OpenAIChatCodec{}).DecodeResponse(body, llmprotocol.DefaultPolicy()); err == nil {
				t.Fatalf("malformed flat detail field was accepted")
			} else if !strings.Contains(err.Error(), "upstream") {
				t.Fatalf("unexpected error class: %v", err)
			}
		})
	}
}
