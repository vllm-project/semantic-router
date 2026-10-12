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

func TestChatStreamReportsProviderUsageInFinalChunk(t *testing.T) {
	decoder := OpenAIChatCodec{}.NewDecoder(
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"},
		llmprotocol.DefaultPolicy(),
	)
	payload := []byte(
		"data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"model\",\"choices\":[],\"usage\":{\"prompt_tokens\":18,\"completion_tokens\":5,\"total_tokens\":23,\"queue_time\":0.037,\"prompt_time\":0.0007,\"completion_time\":0.46,\"total_time\":0.46,\"cost_in_usd_ticks\":123000}}\n\n",
	)
	_, diagnostics, err := decoder.Push(payload)
	if err != nil {
		t.Fatalf("final usage chunk was rejected: %v", err)
	}
	assertDiagnosticFields(
		t, diagnostics,
		"stream.usage.queue_time", "stream.usage.prompt_time", "stream.usage.completion_time",
		"stream.usage.total_time", "stream.usage.cost_in_usd_ticks",
	)
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

// Ollama's OpenAI compatible stream puts a top-level timings object on the usage
// chunk (issue #4585). It is generation metadata, so the stream goes through
// and the drop is reported, whichever provider profile serves it.
func TestChatStreamAcceptsOllamaTimingsOnUsageChunk(t *testing.T) {
	decoder := OpenAIChatCodec{}.NewDecoder(
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"},
		llmprotocol.DefaultPolicy(),
	)
	payload := []byte(
		"data: {\"id\":\"chatcmpl-633\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"model\",\"system_fingerprint\":\"fp_ollama\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":null}]}\n\n" +
			"data: {\"id\":\"chatcmpl-633\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"model\",\"system_fingerprint\":\"fp_ollama\",\"choices\":[],\"usage\":{\"prompt_tokens\":30,\"prompt_tokens_details\":{\"cached_tokens\":29},\"completion_tokens\":10,\"total_tokens\":40},\"timings\":{\"prompt_n\":30,\"prompt_ms\":28,\"prompt_per_token_ms\":0.93,\"prompt_per_second\":1071.4,\"predicted_n\":10,\"predicted_ms\":106,\"predicted_per_token_ms\":10.6,\"predicted_per_second\":94.3}}\n\n" +
			"data: {\"id\":\"chatcmpl-633\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"model\",\"system_fingerprint\":\"fp_ollama\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
			"data: [DONE]\n\n",
	)
	_, diagnostics, err := decoder.Push(payload)
	if err != nil {
		t.Fatalf("usage chunk with timings was rejected: %v", err)
	}
	if len(diagnostics) != 1 || diagnostics[0].Field != "stream.timings" || diagnostics[0].Action != llmprotocol.DiagnosticDropped {
		t.Fatalf("timings omission was not explicit: %+v", diagnostics)
	}
}

// timings is accepted only as an object, and every other unknown field is still
// rejected.
func TestChatStreamTimingsStaysStrict(t *testing.T) {
	for name, extra := range map[string]string{
		"timings is a string":     `"timings":"x"`,
		"timings is an array":     `"timings":[1]`,
		"unrelated unknown field": `"surprise":{"a":1}`,
	} {
		t.Run(name, func(t *testing.T) {
			decoder := OpenAIChatCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"},
				llmprotocol.DefaultPolicy(),
			)
			payload := []byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"model\",\"choices\":[]," + extra + "}\n\n")
			if _, _, err := decoder.Push(payload); err == nil {
				t.Fatalf("chunk with %s was accepted", extra)
			}
		})
	}
}

// Mistral's OpenAI-compatible stream emits chunks with a top-level p field
// containing a string marker (issue #4632). It is transport metadata, so the
// stream goes through and the drop is reported as stream.p.
func TestChatStreamAcceptsMistralPField(t *testing.T) {
	decoder := OpenAIChatCodec{}.NewDecoder(
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"},
		llmprotocol.DefaultPolicy(),
	)
	payload := []byte(
		"data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1791279240,\"model\":\"mistral-medium-latest\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" you today?\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":16,\"total_tokens\":26,\"completion_tokens\":10,\"prompt_tokens_details\":{\"cached_tokens\":0},\"service_tier\":\"standard\"},\"p\":\"abcdefghijklm\"}\n\n" +
			"data: [DONE]\n\n",
	)
	_, diagnostics, err := decoder.Push(payload)
	if err != nil {
		t.Fatalf("chunk with p was rejected: %v", err)
	}
	assertDiagnosticFields(t, diagnostics, "stream.p", "stream.usage.service_tier")
}

// p is accepted only as a string, and non-string types or other unknown fields are still
// rejected.
func TestChatStreamPFieldStaysStrict(t *testing.T) {
	for name, extra := range map[string]string{
		"p is an object":          `"p":{"a":1}`,
		"p is an array":           `"p":[1]`,
		"p is an integer":         `"p":123`,
		"unrelated unknown field": `"surprise":{"a":1}`,
	} {
		t.Run(name, func(t *testing.T) {
			decoder := OpenAIChatCodec{}.NewDecoder(
				llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"},
				llmprotocol.DefaultPolicy(),
			)
			payload := []byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"model\",\"choices\":[]," + extra + "}\n\n")
			if _, _, err := decoder.Push(payload); err == nil {
				t.Fatalf("chunk with %s was accepted", extra)
			}
		})
	}
}
