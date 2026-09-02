package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestDynamoNVExtBufferedChatRoundTrip(t *testing.T) {
	engine := NewBuiltinEngine()
	request := []byte(`{"model":"provider-model","messages":[{"role":"user","content":"hi"}],"cache_salt":"legacy","nvext":{"greed_sampling":true,"annotations":["worker_id","timing"],"extra_fields":["prompt_token_ids"],"cache_salt":"tenant-a","request_timestamp_ms":123.5,"routing_constraints":{"required_taints":["gpu"],"preferred_taints":{"zone-a":0.75}},"router":{"ttft_target":100,"itl_target":20},"agent_hints":{"priority":-2,"strict_priority":1,"osl":2048,"speculative_prefill":true,"latency_sensitivity":0.5}}}`)
	requestResult, err := engine.TranslateRequest(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, request, nil)
	if err != nil {
		t.Fatalf("TranslateRequest() error = %v", err)
	}
	if requestResult.Envelope.Dynamo == nil || requestResult.Envelope.Dynamo.RequestNVExt == nil {
		t.Fatal("decoded request did not retain Dynamo nvext")
	}
	assertJSONField(t, requestResult.Body, "cache_salt", "legacy")
	assertNestedJSONField(t, requestResult.Body, "nvext", "cache_salt", "tenant-a")
	assertNestedJSONField(t, requestResult.Body, "nvext", "request_timestamp_ms", float64(123.5))
	assertNestedJSONField(t, requestResult.Body, "nvext", "extra_fields", []any{"prompt_token_ids"})

	response := []byte(`{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"provider-model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"nvext":{"worker_id":{"prefill_worker_id":1,"decode_worker_id":2},"timing":{"request_received_ms":100,"ttft_ms":4.5},"prompt_token_ids":[1,2],"completion_token_ids":[10,11],"prompt_logprobs":[null,{"42":{"logprob":-0.25,"rank":1,"decoded_token":"hello"}}]}}`)
	responseResult, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, response, nil)
	if err != nil {
		t.Fatalf("TranslateResponse() error = %v", err)
	}
	if responseResult.Envelope.Dynamo == nil || responseResult.Envelope.Dynamo.ResponseNVExt == nil {
		t.Fatal("decoded response did not retain Dynamo nvext")
	}
	assertNestedJSONField(t, responseResult.Body, "nvext", "prompt_token_ids", []any{float64(1), float64(2)})
	assertNestedJSONField(t, responseResult.Body, "nvext", "completion_token_ids", []any{float64(10), float64(11)})
}

func TestDynamoNVExtBufferedRejectsUnknownAndCrossFormat(t *testing.T) {
	engine := NewBuiltinEngine()
	request := []byte(`{"model":"provider-model","messages":[{"role":"user","content":"hi"}],"nvext":{"future":true}}`)
	assertErrorCodeContains(t, translateRequestError(engine, llmprotocol.OpenAIChatV1, request), "invalid_json")

	validRequest := []byte(`{"model":"provider-model","messages":[{"role":"user","content":"hi"}],"nvext":{"greed_sampling":true}}`)
	_, err := engine.TranslateRequest(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, validRequest, nil)
	assertErrorCodeContains(t, err, "unsupported_dynamo_nvext_translation")

	validResponse := []byte(`{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"provider-model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"nvext":{"token_ids":[1]}}`)
	_, err = engine.TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, validResponse, nil)
	assertErrorCodeContains(t, err, "unsupported_dynamo_nvext_translation")
}

func TestDynamoNVExtResponseRequiresOfficialTypedFields(t *testing.T) {
	engine := NewBuiltinEngine()
	base := `{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"provider-model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"nvext":%s}`
	for _, test := range []struct {
		name  string
		nvext string
	}{
		{"timing request timestamp", `{"timing":{"ttft_ms":4.5}}`},
		{"prompt logprob value", `{"prompt_logprobs":[{"42":{"rank":1}}]}`},
		{"timing unknown field", `{"timing":{"request_received_ms":100,"future":true}}`},
		{"prompt logprob unknown field", `{"prompt_logprobs":[{"42":{"logprob":-0.25,"future":true}}]}`},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, []byte(fmt.Sprintf(base, test.nvext)), nil)
			assertErrorCodeContains(t, err, "invalid_")
		})
	}
}

func TestDynamoNVExtStreamPreservesRealChunkWithoutCopyingToTerminalFrames(t *testing.T) {
	stream := newDynamoTestStream(t, llmprotocol.DefaultPolicy(), llmprotocol.OpenAIChatV1)
	payload := []byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}],\"nvext\":{\"worker_id\":{\"decode_worker_id\":2},\"stop_reason\":\"stop\",\"prompt_token_ids\":[1,2]}}\n\n" +
		"data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\",\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n" +
		"data: [DONE]\n\n")
	frames, events, _, err := stream.Push(payload)
	if err != nil {
		t.Fatalf("Push() error = %v", err)
	}
	frames = append(frames, finalizeDynamoStream(t, stream)...)
	if count := bytes.Count(bytes.Join(frames, nil), []byte(`"nvext"`)); count != 1 {
		t.Fatalf("encoded nvext count = %d, want 1; frames = %s", count, bytes.Join(frames, nil))
	}
	var extensionEvents int
	for _, event := range events {
		if event.DynamoNVExt != nil {
			extensionEvents++
			if event.Type != llmprotocol.EventProviderOpaque || event.DynamoNVExt.WorkerID == nil ||
				len(event.DynamoNVExt.PromptTokenIDs) != 2 {
				t.Fatalf("unexpected Dynamo stream event: %#v", event)
			}
		}
	}
	if extensionEvents != 1 {
		t.Fatalf("Dynamo extension events = %d, want 1", extensionEvents)
	}
}

func TestDynamoNVExtStreamRejectsCrossFormatErrorChunkAndCumulativeOverflow(t *testing.T) {
	valid := []byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}],\"nvext\":{\"token_ids\":[1]}}\n\n")
	cross := newDynamoTestStream(t, llmprotocol.DefaultPolicy(), llmprotocol.OpenAIResponsesV1)
	_, _, _, err := cross.Push(valid)
	assertErrorCodeContains(t, err, "unsupported_dynamo_nvext_translation")

	errorChunk := newDynamoTestStream(t, llmprotocol.DefaultPolicy(), llmprotocol.OpenAIChatV1)
	_, _, _, err = errorChunk.Push([]byte("data: {\"error\":{\"type\":\"server_error\",\"message\":\"failed\"},\"nvext\":{\"token_ids\":[1]}}\n\n"))
	assertErrorCodeContains(t, err, "dynamo_nvext_error_chunk")

	policy := llmprotocol.DefaultPolicy()
	rawExtension := []byte(`{"token_ids":[1]}`)
	policy.Limits.DynamoNVExtStreamBytes = len(rawExtension)
	overflow := newDynamoTestStream(t, policy, llmprotocol.OpenAIChatV1)
	if _, _, _, err = overflow.Push(valid); err != nil {
		t.Fatalf("first exact-limit nvext rejected: %v", err)
	}
	_, _, _, err = overflow.Push(valid)
	assertErrorCodeContains(t, err, "dynamo_nvext_stream_size_limit")
}

func TestDynamoNVExtStreamRejectsUnknownAndDeepMetadata(t *testing.T) {
	unknown := newDynamoTestStream(t, llmprotocol.DefaultPolicy(), llmprotocol.OpenAIChatV1)
	_, _, _, err := unknown.Push([]byte("data: {\"object\":\"chat.completion.chunk\",\"choices\":[],\"nvext\":{\"future\":true}}\n\n"))
	assertErrorCodeContains(t, err, "invalid_upstream_json")

	policy := llmprotocol.DefaultPolicy()
	policy.Limits.JSONDepth = 2
	deep := newDynamoTestStream(t, policy, llmprotocol.OpenAIChatV1)
	_, _, _, err = deep.Push([]byte("data: {\"object\":\"chat.completion.chunk\",\"choices\":[],\"nvext\":{\"engine_data\":{\"a\":{\"b\":1}}}}\n\n"))
	assertErrorCodeContains(t, err, "invalid_upstream_json")
}

func newDynamoTestStream(t *testing.T, policy llmprotocol.Policy, target llmprotocol.WireFormat) *StreamEngine {
	t.Helper()
	engine, err := NewEngine(NewBuiltinRegistry(), policy)
	if err != nil {
		t.Fatal(err)
	}
	stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, target, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model", ProviderModel: "provider-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	return stream
}

func finalizeDynamoStream(t *testing.T, stream *StreamEngine) [][]byte {
	t.Helper()
	frames, _, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatalf("Finalize() error = %v", err)
	}
	return frames
}

func translateRequestError(engine *Engine, target llmprotocol.WireFormat, body []byte) error {
	_, err := engine.TranslateRequest(llmprotocol.OpenAIChatV1, target, body, nil)
	return err
}

func assertErrorCodeContains(t *testing.T, err error, code string) {
	t.Helper()
	if err == nil || !strings.Contains(err.Error(), code) {
		t.Fatalf("error = %v, want code %q", err, code)
	}
}

func assertJSONField(t *testing.T, body []byte, key string, want any) {
	t.Helper()
	var object map[string]any
	if err := json.Unmarshal(body, &object); err != nil {
		t.Fatal(err)
	}
	if got := object[key]; !jsonValuesEqual(got, want) {
		t.Fatalf("field %s = %#v, want %#v", key, got, want)
	}
}

func assertNestedJSONField(t *testing.T, body []byte, parent, key string, want any) {
	t.Helper()
	var object map[string]any
	if err := json.Unmarshal(body, &object); err != nil {
		t.Fatal(err)
	}
	nested, ok := object[parent].(map[string]any)
	if !ok {
		t.Fatalf("field %s = %#v, want object", parent, object[parent])
	}
	if got := nested[key]; !jsonValuesEqual(got, want) {
		t.Fatalf("field %s.%s = %#v, want %#v", parent, key, got, want)
	}
}

func jsonValuesEqual(got, want any) bool {
	gotJSON, _ := json.Marshal(got)
	wantJSON, _ := json.Marshal(want)
	return bytes.Equal(gotJSON, wantJSON)
}
