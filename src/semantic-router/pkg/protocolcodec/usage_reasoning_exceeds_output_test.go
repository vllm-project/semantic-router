package protocolcodec

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func assertOutputUsageKeepsOnlyTotal(t *testing.T, usage llmprotocol.Usage, wantTotal int64) {
	t.Helper()
	if usage.State != llmprotocol.UsageAvailable ||
		usage.OutputTotal.Value == nil || *usage.OutputTotal.Value != wantTotal {
		t.Fatalf("usage = %+v, want available with output total %d", usage, wantTotal)
	}
	if usage.OutputReasoning.Value != nil || usage.OutputOther.Value != nil {
		t.Fatalf("inconsistent breakdown must stay unknown: reasoning=%+v other=%+v", usage.OutputReasoning, usage.OutputOther)
	}
}

func TestOpenAIChatResponseKeepsTotalsWhenReasoningExceedsCompletion(t *testing.T) {
	body := []byte(`{"id":"r1","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"completion_tokens_details":{"reasoning_tokens":9}}}`)
	response, _, _, err := NewBuiltinEngine().DecodeResponse(llmprotocol.OpenAIChatV1, body)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	assertOutputUsageKeepsOnlyTotal(t, response.Usage, 5)
}

func TestOpenAIChatResponseStillRejectsNegativeReasoning(t *testing.T) {
	body := []byte(`{"id":"r1","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"completion_tokens_details":{"reasoning_tokens":-1}}}`)
	if _, _, _, err := NewBuiltinEngine().DecodeResponse(llmprotocol.OpenAIChatV1, body); err == nil {
		t.Fatal("negative reasoning tokens must still be rejected")
	}
}

func TestOpenAIResponsesResponseKeepsTotalsWhenReasoningExceedsOutput(t *testing.T) {
	body := []byte(`{"id":"resp_1","object":"response","model":"m","status":"completed","output":[{"type":"message","id":"o1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ok","annotations":[]}]}],` +
		`"usage":{"input_tokens":10,"input_tokens_details":{"cached_tokens":0},"output_tokens":5,"output_tokens_details":{"reasoning_tokens":9},"total_tokens":15}}`)
	response, _, _, err := NewBuiltinEngine().DecodeResponse(llmprotocol.OpenAIResponsesV1, body)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	assertOutputUsageKeepsOnlyTotal(t, response.Usage, 5)
}

func TestOpenAIChatStreamKeepsResponseWhenReasoningExceedsCompletion(t *testing.T) {
	decoder := OpenAIChatCodec{}.NewDecoder(
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"},
		llmprotocol.DefaultPolicy(),
	)
	payload := []byte(
		"data: {\"id\":\"r1\",\"model\":\"model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"},\"finish_reason\":null}]}\n\n" +
			"data: {\"id\":\"r1\",\"model\":\"model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
			"data: {\"id\":\"r1\",\"model\":\"model\",\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15,\"completion_tokens_details\":{\"reasoning_tokens\":9}}}\n\n" +
			"data: [DONE]\n\n",
	)
	events, _, err := decoder.Push(payload)
	if err != nil {
		t.Fatalf("Push() error = %v", err)
	}
	terminal := events[len(events)-1]
	if terminal.Type != llmprotocol.EventResponseCompleted || terminal.Usage == nil {
		t.Fatalf("terminal = %+v", terminal)
	}
	assertOutputUsageKeepsOnlyTotal(t, *terminal.Usage, 5)
}

func TestChatResponseWithInconsistentReasoningReachesEveryClientFormat(t *testing.T) {
	body := []byte(`{"id":"r1","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"completion_tokens_details":{"reasoning_tokens":9}}}`)
	engine := NewBuiltinEngine()
	response, envelope, _, err := engine.DecodeResponse(llmprotocol.OpenAIChatV1, body)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	for _, format := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1} {
		if _, err := engine.EncodeResponse(format, response, envelope); err != nil {
			t.Errorf("EncodeResponse(%s) error = %v", format, err)
		}
	}
}
