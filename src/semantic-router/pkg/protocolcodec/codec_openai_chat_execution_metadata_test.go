package protocolcodec

import (
	"bytes"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const vLLMChatResponseExtensionsFixture = `{
  "id":"chatcmpl-1","object":"chat.completion","created":7,
  "model":"provider-model","service_tier":"default","system_fingerprint":"fp_1",
  "prompt_logprobs":null,"prompt_token_ids":[11,12],"kv_transfer_params":{},
  "choices":[{
    "index":0,"finish_reason":"stop","stop_reason":106,"token_ids":[13],"logprobs":null,
    "message":{"role":"assistant","content":"OK","refusal":null,"annotations":null,
      "audio":null,"function_call":null,"tool_calls":[],"reasoning":null}
  }],
  "usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3,
    "prompt_tokens_details":{"cached_tokens":1,"audio_tokens":0},
    "completion_tokens_details":{"accepted_prediction_tokens":0,"audio_tokens":0,
      "reasoning_tokens":0,"rejected_prediction_tokens":0}}
}`

func TestOpenAIChatResponseAcceptsClosedVLLMExecutionMetadata(t *testing.T) {
	engine := NewBuiltinEngine()
	translated, translateErr := engine.TranslateResponse(
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIChatV1,
		[]byte(vLLMChatResponseExtensionsFixture),
		func(response *llmprotocol.Response) error {
			response.Model = "public-model"
			return nil
		},
	)
	if translateErr != nil {
		t.Fatalf("TranslateResponse() error = %v", translateErr)
	}
	if translated.Response.Model != "public-model" || translated.Response.Usage.Total.Value == nil ||
		*translated.Response.Usage.Total.Value != 3 {
		t.Fatalf("translated response = %+v", translated.Response)
	}
}

func TestOpenAIChatResponseExtensionEnvelopeRoundTripAndUnknownRejection(t *testing.T) {
	engine := NewBuiltinEngine()
	raw := []byte(vLLMChatResponseExtensionsFixture)
	translated, translateErr := engine.TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, raw, nil)
	if translateErr != nil {
		t.Fatalf("TranslateResponse() error = %v", translateErr)
	}
	if !bytes.Equal(translated.Body, raw) {
		t.Fatal("same-format response envelope was not replayed byte-for-byte")
	}

	future := bytes.Replace(raw, []byte(`"kv_transfer_params":{}`), []byte(`"kv_transfer_params":{},"future_field":true`), 1)
	_, translateErr = engine.TranslateResponse(
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIChatV1,
		future,
		func(*llmprotocol.Response) error { return nil },
	)
	if translateErr == nil || !strings.Contains(translateErr.Error(), "invalid_upstream_json") {
		t.Fatalf("future field error = %v", translateErr)
	}
}

func TestOpenAIChatResponseRejectsUnsupportedNonNullOutputExtensions(t *testing.T) {
	engine := NewBuiltinEngine()
	for field, replacement := range map[string]string{
		"audio":         `"audio":{"id":"audio_1","data":"AA==","expires_at":9,"transcript":"OK"}`,
		"function_call": `"function_call":{"name":"legacy","arguments":"{}"}`,
	} {
		t.Run(field, func(t *testing.T) {
			raw := strings.Replace(vLLMChatResponseExtensionsFixture, `"`+field+`":null`, replacement, 1)
			_, translateErr := engine.TranslateResponse(
				llmprotocol.OpenAIChatV1,
				llmprotocol.OpenAIChatV1,
				[]byte(raw),
				func(*llmprotocol.Response) error { return nil },
			)
			if translateErr == nil || !strings.Contains(translateErr.Error(), "unsupported_upstream") {
				t.Fatalf("non-null %s error = %v", field, translateErr)
			}
		})
	}
}

func TestOpenAIChatResponseRejectsInvalidClosedExecutionMetadata(t *testing.T) {
	engine := NewBuiltinEngine()
	tests := map[string]string{
		"non_null_prompt_logprobs": strings.Replace(vLLMChatResponseExtensionsFixture, `"prompt_logprobs":null`, `"prompt_logprobs":[]`, 1),
		"nested_kv_field":          strings.Replace(vLLMChatResponseExtensionsFixture, `"kv_transfer_params":{}`, `"kv_transfer_params":{"future":true}`, 1),
		"unknown_service_tier":     strings.Replace(vLLMChatResponseExtensionsFixture, `"service_tier":"default"`, `"service_tier":"future"`, 1),
		"non_scalar_stop_reason":   strings.Replace(vLLMChatResponseExtensionsFixture, `"stop_reason":106`, `"stop_reason":true`, 1),
		"fractional_stop_reason":   strings.Replace(vLLMChatResponseExtensionsFixture, `"stop_reason":106`, `"stop_reason":1.5`, 1),
		"exponent_stop_reason":     strings.Replace(vLLMChatResponseExtensionsFixture, `"stop_reason":106`, `"stop_reason":1e3`, 1),
		"nested_usage_field": strings.Replace(
			vLLMChatResponseExtensionsFixture,
			`"prompt_tokens_details":{"cached_tokens":1,"audio_tokens":0}`,
			`"prompt_tokens_details":{"cached_tokens":1,"audio_tokens":0,"future":1}`,
			1,
		),
		"oversized_fingerprint": strings.Replace(
			vLLMChatResponseExtensionsFixture,
			`"system_fingerprint":"fp_1"`,
			`"system_fingerprint":"`+strings.Repeat("x", 257)+`"`,
			1,
		),
	}
	for name, raw := range tests {
		t.Run(name, func(t *testing.T) {
			_, translateErr := engine.TranslateResponse(
				llmprotocol.OpenAIChatV1,
				llmprotocol.OpenAIChatV1,
				[]byte(raw),
				func(*llmprotocol.Response) error { return nil },
			)
			if translateErr == nil {
				t.Fatal("invalid closed metadata was accepted")
			}
		})
	}
}

func TestOpenAIChatStreamAcceptsClosedVLLMExecutionMetadataAndRejectsFutureFields(t *testing.T) {
	engine := NewBuiltinEngine()
	context := llmprotocol.StreamContext{PublicModel: "public-model", ProviderModel: "provider-model"}
	stream, streamErr := engine.NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, context)
	if streamErr != nil {
		t.Fatal(streamErr)
	}
	frame := []byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":7," +
		"\"model\":\"provider-model\",\"service_tier\":\"default\",\"system_fingerprint\":\"fp_1\"," +
		"\"prompt_logprobs\":null,\"prompt_token_ids\":[11],\"kv_transfer_params\":{}," +
		"\"choices\":[{\"index\":0,\"finish_reason\":null,\"stop_reason\":null,\"token_ids\":[12]," +
		"\"logprobs\":null,\"delta\":{\"role\":\"assistant\",\"content\":\"OK\",\"audio\":null,\"function_call\":null}}]}\n\n")
	frames, events, _, pushErr := stream.Push(frame)
	if pushErr != nil {
		t.Fatalf("Push() error = %v", pushErr)
	}
	for _, event := range events {
		if event.Model != "public-model" {
			t.Fatalf("event model = %q, want public-model", event.Model)
		}
	}
	if bytes.Contains(bytes.Join(frames, nil), []byte("provider-model")) ||
		!bytes.Contains(bytes.Join(frames, nil), []byte("public-model")) {
		t.Fatalf("rewritten frames = %q", bytes.Join(frames, nil))
	}

	unknown, streamErr := engine.NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, context)
	if streamErr != nil {
		t.Fatal(streamErr)
	}
	future := bytes.Replace(frame, []byte(`"kv_transfer_params":{}`), []byte(`"kv_transfer_params":{},"future_field":true`), 1)
	if _, _, _, pushErr := unknown.Push(future); pushErr == nil || !strings.Contains(pushErr.Error(), "invalid_upstream_json") {
		t.Fatalf("future stream field error = %v", pushErr)
	}

	sameModel, streamErr := engine.NewStream(
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIChatV1,
		llmprotocol.StreamContext{PublicModel: "same-model", ProviderModel: "same-model"},
	)
	if streamErr != nil {
		t.Fatal(streamErr)
	}
	if _, _, _, pushErr := sameModel.Push(future); pushErr == nil || !strings.Contains(pushErr.Error(), "invalid_upstream_json") {
		t.Fatalf("same-model future stream field error = %v", pushErr)
	}
	if _, _, decodeErr := engine.DecodeResponseStream(
		llmprotocol.OpenAIChatV1,
		append(append([]byte(nil), future...), []byte("data: [DONE]\n\n")...),
		llmprotocol.StreamContext{PublicModel: "same-model", ProviderModel: "same-model"},
	); decodeErr == nil || !strings.Contains(decodeErr.Error(), "invalid_upstream_json") {
		t.Fatalf("DecodeResponseStream future field error = %v", decodeErr)
	}

	nested, streamErr := engine.NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, context)
	if streamErr != nil {
		t.Fatal(streamErr)
	}
	nestedFuture := bytes.Replace(frame, []byte(`"kv_transfer_params":{}`), []byte(`"kv_transfer_params":{"future":true}`), 1)
	if _, _, _, pushErr := nested.Push(nestedFuture); pushErr == nil || !strings.Contains(pushErr.Error(), "invalid_upstream_json") {
		t.Fatalf("future nested stream field error = %v", pushErr)
	}

	streamLogprobs, streamErr := engine.NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, context)
	if streamErr != nil {
		t.Fatal(streamErr)
	}
	withLogprobs := bytes.Replace(frame, []byte(`"logprobs":null`), []byte(`"logprobs":{"content":[]}`), 1)
	if _, _, _, pushErr := streamLogprobs.Push(withLogprobs); pushErr == nil || !strings.Contains(pushErr.Error(), "unsupported_upstream_stream_logprobs") {
		t.Fatalf("stream logprobs error = %v", pushErr)
	}
}
