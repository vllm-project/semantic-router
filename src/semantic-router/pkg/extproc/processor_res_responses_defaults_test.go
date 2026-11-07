package extproc

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// The vLLM Responses transfer and per-turn fields at their defaults are known
// members: a same-format reply and its cache entry keep them byte for byte.
func TestExtProcResponsesProviderDefaults(t *testing.T) {
	newContext := func() *RequestContext {
		return &RequestContext{SourceFormat: llmprotocol.OpenAIResponsesV1, TargetFormat: llmprotocol.OpenAIResponsesV1, RequestModel: "public-model", TraceContext: t.Context()}
	}
	t.Run("buffered", func(t *testing.T) {
		router := &OpenAIRouter{}
		ctx := newContext()
		body := extProcResponsesDefaults(t, extProcResponseFixture(llmprotocol.OpenAIResponsesV1))
		reply := router.handleNonStreamingResponseBody(body, ctx, 0)
		if reply.GetImmediateResponse() != nil || reply.GetResponseBody() == nil {
			t.Fatalf("response rejected: %+v", reply)
		}
		if mutation := reply.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil {
			t.Fatalf("same-format forwarding changed: %+v", mutation)
		}
		assertResponsesDefaultsUsage(t, ctx)
		cached := router.cacheableClientResponse(body, false, *ctx.SemanticResponse, ctx)
		if !bytes.Equal(cached, body) {
			t.Fatal("default response changed on cache write")
		}
		if _, err := router.decodeCachedClientResponse(cached, newContext()); err != nil {
			t.Fatalf("cache read: %v", err)
		}
	})
	t.Run("stream", func(t *testing.T) {
		ctx := newContext()
		body := extProcResponsesDefaults(t, extProcStreamFixture(llmprotocol.OpenAIResponsesV1))
		reply := (&OpenAIRouter{}).handleSemanticStreamingResponseBody(body, true, ctx)
		if ctx.StreamingAborted || !ctx.StreamingComplete || reply.GetResponseBody() == nil {
			t.Fatalf("aborted=%t complete=%t reply=%+v", ctx.StreamingAborted, ctx.StreamingComplete, reply)
		}
		if mutation := reply.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil {
			t.Fatalf("same-format stream forwarding changed: %+v", mutation)
		}
		assertResponsesDefaultsUsage(t, ctx)
	})
}

func extProcResponsesDefaults(t *testing.T, body []byte) []byte {
	t.Helper()
	decorate := func(raw []byte) []byte {
		var object map[string]any
		if err := json.Unmarshal(raw, &object); err != nil {
			t.Fatal(err)
		}
		resource := object
		if strings.HasPrefix(string(body), "event:") {
			var ok bool
			resource, ok = object["response"].(map[string]any)
			if !ok {
				return raw
			}
		}
		for _, field := range []string{"kv_transfer_params", "ec_transfer_params", "input_messages", "output_messages"} {
			resource[field] = nil
		}
		if usage, ok := resource["usage"].(map[string]any); ok {
			usage["input_tokens_details"] = map[string]any{"cached_tokens": 0, "input_tokens_per_turn": []any{}, "cached_tokens_per_turn": []any{}}
			usage["output_tokens_details"] = map[string]any{"reasoning_tokens": 0, "output_tokens_per_turn": []any{}, "tool_output_tokens": 0, "tool_output_tokens_per_turn": []any{}}
		}
		result, err := json.Marshal(object)
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	if !strings.HasPrefix(string(body), "event:") {
		return decorate(body)
	}
	lines := strings.Split(string(body), "\n")
	for index, line := range lines {
		if strings.HasPrefix(line, "data: ") {
			lines[index] = "data: " + string(decorate([]byte(strings.TrimPrefix(line, "data: "))))
		}
	}
	return []byte(strings.Join(lines, "\n"))
}

func assertResponsesDefaultsUsage(t *testing.T, ctx *RequestContext) {
	t.Helper()
	if ctx.SemanticResponse == nil || ctx.SemanticResponse.Usage.Total.Value == nil || *ctx.SemanticResponse.Usage.Total.Value != 3 {
		t.Fatalf("usage changed: %+v", ctx.SemanticResponse)
	}
}
