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
  "prompt_logprobs":null,"prompt_token_ids":[11,12],"prompt_text":null,"kv_transfer_params":{},
  "ec_transfer_params":null,"metrics":null,
  "choices":[{
    "index":0,"finish_reason":"stop","stop_reason":106,"token_ids":[13],"logprobs":null,
    "routed_experts":null,
    "message":{"role":"assistant","content":"OK","refusal":null,"annotations":null,
      "audio":null,"function_call":null,"tool_calls":[],"reasoning":null}
  }],
  "usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3,
    "prompt_tokens_details":{"cached_tokens":1,"audio_tokens":0},
    "completion_tokens_details":{"accepted_prediction_tokens":0,"audio_tokens":0,
      "reasoning_tokens":0,"rejected_prediction_tokens":0}}
}`

const legacyVLLMChatResponseExtensionsFixture = `{
  "id":"chatcmpl-legacy","object":"chat.completion","created":7,"model":"provider-model",
  "do_remote_decode":false,"do_remote_prefill":false,"remote_block_ids":null,
  "remote_engine_id":"","remote_host":"","remote_port":0,
  "choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"OK"}}],
  "usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}
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

func TestOpenAIChatResponseAcceptsLegacyVLLMKVTransferMetadata(t *testing.T) {
	engine := NewBuiltinEngine()
	translated, translateErr := engine.TranslateResponse(
		llmprotocol.OpenAIChatV1,
		llmprotocol.AnthropicMessagesV1,
		[]byte(legacyVLLMChatResponseExtensionsFixture),
		nil,
	)
	if translateErr != nil {
		t.Fatalf("TranslateResponse() error = %v", translateErr)
	}
	if translated.Response.Model != "provider-model" || len(translated.Response.Output) != 1 {
		t.Fatalf("translated response = %+v", translated.Response)
	}
	assertDiagnosticFields(t, translated.Diagnostics, "kv_transfer")
}

func TestOpenAIChatResponseAcceptsProviderTokenizedToolArguments(t *testing.T) {
	raw := []byte(`{
		"id":"chatcmpl-tool","object":"chat.completion","created":7,"model":"provider-model",
		"choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":null,
			"tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{}","TokenizedArguments":["{","}"]}}]}}],
		"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}
	}`)
	translated, err := NewBuiltinEngine().TranslateResponse(
		llmprotocol.OpenAIChatV1,
		llmprotocol.AnthropicMessagesV1,
		raw,
		nil,
	)
	if err != nil {
		t.Fatalf("TranslateResponse() error = %v", err)
	}
	if translated.Response.StopReason != llmprotocol.StopToolCall || len(translated.Response.Output) != 1 {
		t.Fatalf("translated response = %+v", translated.Response)
	}
	assertDiagnosticFields(t, translated.Diagnostics, "choices.message.tool_calls.function.TokenizedArguments")
}

func TestOpenAIChatRequestRejectsProviderTokenizedToolArguments(t *testing.T) {
	raw := []byte(`{
		"model":"client-model","messages":[{"role":"user","content":"lookup"},{"role":"assistant","tool_calls":[{
			"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{}","TokenizedArguments":["{","}"]}
		}]}]
	}`)
	_, err := NewBuiltinEngine().TranslateRequest(
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIChatV1,
		raw,
		nil,
	)
	if err == nil || !strings.Contains(err.Error(), "unsupported_messages_tool_calls_function_tokenized_arguments") {
		t.Fatalf("TranslateRequest() error = %v", err)
	}
}

const ollamaChatToolCallResponseFixture = `{
  "id":"chatcmpl-1","object":"chat.completion","created":7,"model":"llama3.2:3b",
  "system_fingerprint":"fp_ollama",
  "choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":"",
    "tool_calls":[{"id":"call_1","index":0,"type":"function",
      "function":{"name":"read_file","arguments":"{\"path\":\"billing.py\"}"}}]}}],
  "usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}
}`

func TestOpenAIChatResponseAcceptsOllamaToolCallIndex(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1} {
		translated, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, target, []byte(ollamaChatToolCallResponseFixture), nil)
		if err != nil {
			t.Fatalf("TranslateResponse(%s) error = %v", target, err)
		}
		if translated.Response.StopReason != llmprotocol.StopToolCall || len(translated.Response.Output) != 1 {
			t.Fatalf("translated response (%s) = %+v", target, translated.Response)
		}
		assertDiagnosticFields(t, translated.Diagnostics, "choices.message.tool_calls.index")
	}

	unknown := strings.Replace(ollamaChatToolCallResponseFixture, `"index":0,"type"`, `"index":0,"future":1,"type"`, 1)
	_, err := conformanceEngine().TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, []byte(unknown), nil)
	if err == nil || !strings.Contains(err.Error(), "invalid_upstream_json") {
		t.Fatalf("unknown tool call field error = %v", err)
	}
}

func TestOpenAIChatRequestAcceptsEchoedToolCallIndex(t *testing.T) {
	raw := []byte(`{
		"model":"client-model","messages":[{"role":"user","content":"read billing.py"},
		{"role":"assistant","content":"","tool_calls":[{"id":"call_1","index":0,"type":"function",
			"function":{"name":"read_file","arguments":"{}"}}]},
		{"role":"tool","tool_call_id":"call_1","content":"ok"}]
	}`)
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1} {
		if _, err := NewBuiltinEngine().TranslateRequest(llmprotocol.OpenAIChatV1, target, raw, nil); err != nil {
			t.Fatalf("TranslateRequest(%s) error = %v", target, err)
		}
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
	_, translateErr = conformanceEngine().TranslateResponse(
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

// vLLM's serving metadata is opaque: any value decodes, even under the
// conformance contract, is reported as dropped from the neutral response and
// is replayed unchanged to a Chat client.
func TestOpenAIChatResponseAcceptsNonNullVLLMServingMetadata(t *testing.T) {
	tests := map[string]struct{ from, to, field string }{
		"prompt_logprobs": {`"prompt_logprobs":null`, `"prompt_logprobs":[null,{"11":{"logprob":-0.5}}]`, "prompt_logprobs"},
		"prompt_text":     {`"prompt_text":null`, `"prompt_text":"rendered prompt"`, "prompt_text"},
		"ec_transfer":     {`"ec_transfer_params":null`, `"ec_transfer_params":{"do_remote_encode":true}`, "ec_transfer_params"},
		"metrics":         {`"metrics":null`, `"metrics":{"time_to_first_token_ms":12.5}`, "metrics"},
		"routed_experts":  {`"routed_experts":null`, `"routed_experts":"k05VTVBZ"`, "choices.routed_experts"},
		"weight_versions": {`"routed_experts":null`, `"routed_experts":null,"weight_versions":[{"version":"v1","start":0,"end":1}]`, "choices.weight_versions"},
		"kv_transfer":     {`"kv_transfer_params":{}`, `"kv_transfer_params":{"do_remote_prefill":true,"remote_engine_id":"e1"}`, "kv_transfer_params"},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			raw := []byte(strings.Replace(vLLMChatResponseExtensionsFixture, test.from, test.to, 1))
			translated, err := conformanceEngine().TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, raw, nil)
			if err != nil {
				t.Fatalf("TranslateResponse() error = %v", err)
			}
			if !bytes.Equal(translated.Body, raw) {
				t.Fatal("same-format response was not replayed byte-for-byte")
			}
			if !hasServingMetadataDiagnostic(translated.Diagnostics, test.field) {
				t.Fatalf("no serving metadata diagnostic for %s in %+v", test.field, translated.Diagnostics)
			}
		})
	}
}

func TestOpenAIChatResponseRejectsInvalidClosedExecutionMetadata(t *testing.T) {
	engine := NewBuiltinEngine()
	tests := map[string]string{
		"unknown_service_tier":   strings.Replace(vLLMChatResponseExtensionsFixture, `"service_tier":"default"`, `"service_tier":"future"`, 1),
		"non_scalar_stop_reason": strings.Replace(vLLMChatResponseExtensionsFixture, `"stop_reason":106`, `"stop_reason":true`, 1),
		"fractional_stop_reason": strings.Replace(vLLMChatResponseExtensionsFixture, `"stop_reason":106`, `"stop_reason":1.5`, 1),
		"exponent_stop_reason":   strings.Replace(vLLMChatResponseExtensionsFixture, `"stop_reason":106`, `"stop_reason":1e3`, 1),
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
			decoder := engine
			if strings.HasPrefix(name, "nested_") {
				// An undeclared member inside a closed object is an unknown
				// field: only the conformance contract rejects it.
				decoder = conformanceEngine()
			}
			_, translateErr := decoder.TranslateResponse(
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
		"\"prompt_logprobs\":null,\"prompt_token_ids\":[11],\"prompt_text\":null,\"kv_transfer_params\":{}," +
		"\"ec_transfer_params\":null,\"metrics\":null," +
		"\"choices\":[{\"index\":0,\"finish_reason\":null,\"stop_reason\":null,\"token_ids\":[12]," +
		"\"logprobs\":null,\"routed_experts\":null,\"delta\":{\"role\":\"assistant\",\"content\":\"OK\",\"audio\":null,\"function_call\":null}}]}\n\n")
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

	future := bytes.Replace(frame, []byte(`"kv_transfer_params":{}`), []byte(`"kv_transfer_params":{},"future_field":true`), 1)
	assertChatStreamRejected(t, conformanceEngine(), context, future, "invalid_upstream_json")
	sameModelContext := llmprotocol.StreamContext{PublicModel: "same-model", ProviderModel: "same-model"}
	assertChatStreamRejected(t, conformanceEngine(), sameModelContext, future, "invalid_upstream_json")
	if _, _, decodeErr := conformanceEngine().DecodeResponseStream(
		llmprotocol.OpenAIChatV1,
		append(append([]byte(nil), future...), []byte("data: [DONE]\n\n")...),
		sameModelContext,
	); decodeErr == nil || !strings.Contains(decodeErr.Error(), "invalid_upstream_json") {
		t.Fatalf("DecodeResponseStream future field error = %v", decodeErr)
	}

	// The handshake inside kv_transfer_params is opaque serving metadata.
	handshake := bytes.Replace(frame, []byte(`"kv_transfer_params":{}`), []byte(`"kv_transfer_params":{"do_remote_decode":true}`), 1)
	handshakeStream, streamErr := conformanceEngine().NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, context)
	if streamErr != nil {
		t.Fatal(streamErr)
	}
	if _, _, diagnostics, pushErr := handshakeStream.Push(handshake); pushErr != nil {
		t.Fatalf("Push() with a KV-transfer handshake error = %v", pushErr)
	} else if !hasServingMetadataDiagnostic(diagnostics, "stream.kv_transfer_params") {
		t.Fatalf("no serving metadata diagnostic for the handshake in %+v", diagnostics)
	}
	withLogprobs := bytes.Replace(frame, []byte(`"logprobs":null`), []byte(`"logprobs":{"content":[]}`), 1)
	assertChatStreamRejected(t, engine, context, withLogprobs, "unsupported_upstream_stream_logprobs")
}

func TestOpenAIChatStreamAcceptsProviderTokenizedToolArguments(t *testing.T) {
	engine := NewBuiltinEngine()
	context := llmprotocol.StreamContext{PublicModel: "public-model", ProviderModel: "provider-model"}
	stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, context)
	if err != nil {
		t.Fatal(err)
	}
	frame := []byte("data: {\"id\":\"chatcmpl-tool\",\"object\":\"chat.completion.chunk\",\"created\":7," +
		"\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"finish_reason\":null,\"delta\":{" +
		"\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"type\":\"function\"," +
		"\"function\":{\"name\":\"lookup\",\"arguments\":\"{}\",\"TokenizedArguments\":[\"{\",\"}\"]}}]}}]}\n\n")
	_, events, diagnostics, pushErr := stream.Push(frame)
	if pushErr != nil {
		t.Fatalf("Push() error = %v", pushErr)
	}
	if len(events) == 0 {
		t.Fatal("Push() emitted no neutral events")
	}
	assertDiagnosticFields(t, diagnostics, "choices.delta.tool_calls.function.TokenizedArguments")
}

func assertChatStreamRejected(
	t *testing.T,
	engine *Engine,
	context llmprotocol.StreamContext,
	payload []byte,
	errorCode string,
) {
	t.Helper()
	stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, context)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, _, pushErr := stream.Push(payload); pushErr == nil || !strings.Contains(pushErr.Error(), errorCode) {
		t.Fatalf("stream error = %v, want %s", pushErr, errorCode)
	}
}
