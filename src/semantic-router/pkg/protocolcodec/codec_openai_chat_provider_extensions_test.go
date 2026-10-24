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

// LiteLLM proxied vLLM reply with provider_specific_fields on both the
// choice and the message (issue #4936). The useful content is already in
// canonical fields; the decorations are dropped and the response is
// re-encoded so they are not replayed to the client.
const litellmChatResponseFixture = `{
  "id":"chatcmpl-litellm","object":"chat.completion","created":1791782944,
  "model":"Qwen/Qwen3.8-27B",
  "system_fingerprint":"vllm-0.26.1rc1.dev1177+ga9a17e709-tp2-58b79d32",
  "choices":[{
    "finish_reason":"stop","index":0,
    "message":{
      "content":"\n\nok","role":"assistant",
      "reasoning_content":"We need to respond with exactly ok.",
      "provider_specific_fields":{"reasoning":"We need to respond with exactly ok.\n","refusal":null}
    },
    "provider_specific_fields":{"routed_experts":null,"token_ids":null,"stop_reason":null}
  }],
  "usage":{"completion_tokens":29,"prompt_tokens":57,"total_tokens":86,
    "completion_tokens_details":{"reasoning_tokens":25}}
}`

func TestOpenAIChatResponseAcceptsLiteLLMProviderSpecificFields(t *testing.T) {
	response, envelope, diagnostics, err := NewBuiltinEngine().DecodeResponse(llmprotocol.OpenAIChatV1, []byte(litellmChatResponseFixture))
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if response.Usage.Total.Value == nil || *response.Usage.Total.Value != 86 {
		t.Fatalf("usage = %+v", response.Usage)
	}
	if len(response.Output) != 1 {
		t.Fatalf("output = %+v", response.Output)
	}
	assertDiagnosticFields(t, diagnostics,
		"choices.message.provider_specific_fields",
		"choices.provider_specific_fields",
	)
	if !envelope.ResponseReencodeRequired {
		t.Fatal("LiteLLM decorations must force a re-encode")
	}
}

func TestOpenAIChatResponseTranslatesLiteLLMReply(t *testing.T) {
	translated, err := NewBuiltinEngine().TranslateResponse(
		llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, []byte(litellmChatResponseFixture), nil,
	)
	if err != nil {
		t.Fatalf("TranslateResponse() error = %v", err)
	}
	if len(translated.Response.Output) != 1 || translated.Response.Usage.Total.Value == nil {
		t.Fatalf("translated response = %+v", translated.Response)
	}
	if strings.Contains(string(translated.Body), "provider_specific_fields") {
		t.Fatal("provider_specific_fields must not appear in the re-encoded response")
	}
}

func TestOpenAIChatRequestRejectsLiteLLMProviderSpecificFields(t *testing.T) {
	raw := []byte(`{"model":"client-model","messages":[{"role":"user","content":"ok","provider_specific_fields":{"reasoning":"x"}}]}`)
	_, _, _, err := NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIChatV1, raw)
	assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_messages_provider_specific_fields")
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

// A provider field that rides every chunk is one omission, not one per chunk.
// Each Push starts its diagnostic list empty, so before the once-per-stream
// guard the repeats reached the caller's accumulation: one increment of
// llm_translation_lossy_total and one structured event per chunk for a single
// loss, and a Push carrying the whole stream spent the 64-entry budget on
// repeats so the final chunk's usage omission was truncated away.
func TestChatStreamReportsAChunkConstantFieldOnce(t *testing.T) {
	repeated := []byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"mistral-medium-latest\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" hi\"}}],\"p\":\"abcdefghijklm\"}\n\n")
	final := []byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1791279240,\"model\":\"mistral-medium-latest\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" you today?\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":16,\"total_tokens\":26,\"completion_tokens\":10,\"prompt_tokens_details\":{\"cached_tokens\":0},\"service_tier\":\"standard\"},\"p\":\"abcdefghijklm\"}\n\n" +
		"data: [DONE]\n\n")

	t.Run("once across pushes", func(t *testing.T) {
		decoder := OpenAIChatCodec{}.NewDecoder(
			llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"},
			llmprotocol.DefaultPolicy(),
		)
		var accumulated llmprotocol.Diagnostics
		for i := 0; i < 99; i++ {
			_, diagnostics, err := decoder.Push(repeated)
			if err != nil {
				t.Fatalf("chunk %d was rejected: %v", i, err)
			}
			accumulated = append(accumulated, diagnostics...)
		}
		_, finalDiagnostics, err := decoder.Push(final)
		if err != nil {
			t.Fatalf("final chunk was rejected: %v", err)
		}
		assertDiagnosticFields(t, accumulated, "stream.p")
		assertDiagnosticFields(t, finalDiagnostics, "stream.usage.service_tier")
	})

	t.Run("the budget keeps distinct losses", func(t *testing.T) {
		decoder := OpenAIChatCodec{}.NewDecoder(
			llmprotocol.StreamContext{Context: context.Background(), PublicModel: "model"},
			llmprotocol.DefaultPolicy(),
		)
		var payload []byte
		for i := 0; i < 99; i++ {
			payload = append(payload, repeated...)
		}
		payload = append(payload, final...)
		_, diagnostics, err := decoder.Push(payload)
		if err != nil {
			t.Fatalf("stream was rejected: %v", err)
		}
		assertDiagnosticFields(t, diagnostics, "stream.p", "stream.usage.service_tier")
	})
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
