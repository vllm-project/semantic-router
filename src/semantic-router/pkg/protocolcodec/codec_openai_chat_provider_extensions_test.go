package protocolcodec

import (
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Reply shapes returned by xAI's and Groq's OpenAI-compatible Chat endpoints.
const xAIChatResponseFixture = `{
  "id":"xai-1","object":"chat.completion","created":7,"model":"grok-4.20-0309-non-reasoning",
  "choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"ok","refusal":null}}],
  "usage":{"prompt_tokens":9,"completion_tokens":1,"total_tokens":10,
    "prompt_tokens_details":{"text_tokens":9,"audio_tokens":0,"image_tokens":0,"cached_tokens":0},
    "completion_tokens_details":{"reasoning_tokens":0,"audio_tokens":0,
      "accepted_prediction_tokens":0,"rejected_prediction_tokens":0},
    "num_sources_used":0,"cost_in_usd_ticks":123000},
  "system_fingerprint":"fp_1"
}`

const groqChatResponseFixture = `{
  "id":"chatcmpl-groq","object":"chat.completion","created":7,"model":"openai/gpt-oss-20b",
  "choices":[{"index":0,"logprobs":null,"finish_reason":"stop",
    "message":{"role":"assistant","content":"ok","reasoning":"short"}}],
  "usage":{"queue_time":0.037,"prompt_tokens":18,"prompt_time":0.0007,
    "completion_tokens":5,"completion_time":0.46,"total_tokens":23,"total_time":0.46},
  "system_fingerprint":"fp_179b0f92c9","x_groq":{"id":"req_01jbd6g2qdfw2adyrt2az8hz4w"},
  "service_tier":"on_demand","usage_breakdown":null
}`

func TestOpenAIChatResponseAcceptsXAIUsageExtensions(t *testing.T) {
	response, _, diagnostics, err := NewBuiltinEngine().DecodeResponse(llmprotocol.OpenAIChatV1, []byte(xAIChatResponseFixture))
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if response.Usage.Total.Value == nil || *response.Usage.Total.Value != 10 {
		t.Fatalf("usage = %+v", response.Usage)
	}
	assertDiagnosticFields(t, diagnostics, "usage.cost_in_usd_ticks", "usage.prompt_tokens_details.text_tokens")
}

func TestOpenAIChatResponseAcceptsGroqExecutionMetadata(t *testing.T) {
	response, _, diagnostics, err := NewBuiltinEngine().DecodeResponse(llmprotocol.OpenAIChatV1, []byte(groqChatResponseFixture))
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if response.Usage.Total.Value == nil || *response.Usage.Total.Value != 23 {
		t.Fatalf("usage = %+v", response.Usage)
	}
	assertDiagnosticFields(
		t, diagnostics,
		"x_groq", "usage.queue_time", "usage.prompt_time", "usage.completion_time", "usage.total_time",
	)
}

func TestOpenAIChatResponseReportsGroqUsageBreakdown(t *testing.T) {
	body := strings.Replace(groqChatResponseFixture, `"usage_breakdown":null`, `"usage_breakdown":{"models":[]}`, 1)
	_, _, diagnostics, err := NewBuiltinEngine().DecodeResponse(llmprotocol.OpenAIChatV1, []byte(body))
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	assertDiagnosticFields(
		t, diagnostics,
		"usage_breakdown", "x_groq", "usage.queue_time", "usage.prompt_time", "usage.completion_time", "usage.total_time",
	)
}

func TestOpenAIChatResponseTranslatesProviderExtensionReplies(t *testing.T) {
	for name, fixture := range map[string]string{"xai": xAIChatResponseFixture, "groq": groqChatResponseFixture} {
		t.Run(name, func(t *testing.T) {
			translated, err := NewBuiltinEngine().TranslateResponse(
				llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, []byte(fixture), nil,
			)
			if err != nil {
				t.Fatalf("TranslateResponse() error = %v", err)
			}
			if len(translated.Response.Output) != 1 || translated.Response.Usage.Total.Value == nil {
				t.Fatalf("translated response = %+v", translated.Response)
			}
		})
	}
}

func TestOpenAIChatResponseStillRejectsUnknownProviderFields(t *testing.T) {
	body := strings.Replace(groqChatResponseFixture, `"x_groq"`, `"x_unknown":1,"x_groq"`, 1)
	_, _, _, err := NewBuiltinEngine().DecodeResponse(llmprotocol.OpenAIChatV1, []byte(body))
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json")
}

func TestChatStreamAcceptsGroqChunkMetadata(t *testing.T) {
	decoder := OpenAIChatCodec{}.NewDecoder(
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"},
		llmprotocol.DefaultPolicy(),
	)
	payload := []byte(
		"data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"model\",\"x_groq\":{\"id\":\"req_1\"},\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"},\"finish_reason\":null,\"logprobs\":null}]}\n\n",
	)
	events, diagnostics, err := decoder.Push(payload)
	if err != nil {
		t.Fatalf("Groq Chat stream chunk was rejected: %v", err)
	}
	if len(events) < 2 {
		t.Fatalf("Chat stream events = %+v", events)
	}
	if len(diagnostics) != 1 || diagnostics[0].Field != "stream.x_groq" || diagnostics[0].Action != llmprotocol.DiagnosticDropped {
		t.Fatalf("x_groq omission was not explicit: %+v", diagnostics)
	}
}
