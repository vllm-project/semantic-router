package extproc

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

func TestResponseAPIStreamingResponseRequiresResponsesSSE(t *testing.T) {
	router := &OpenAIRouter{
		Config:            &config.RouterConfig{},
		ResponseAPIFilter: NewResponseAPIFilter(NewMockResponseStore()),
	}
	ctx := newResponseAPIStreamingTestContext("response-api-stream-repro")
	chunk := []byte(`data: {"id":"chatcmpl-repro","object":"chat.completion.chunk","created":123,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"foo"},"finish_reason":null}]}` + "\n\n")

	resp, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: chunk},
	}, ctx)
	require.NoError(t, err)
	require.NotNil(t, resp)

	bodyMutation := resp.GetResponseBody().GetResponse().GetBodyMutation()
	require.NotNil(t, bodyMutation, "Responses API stream:true must rewrite Chat Completions SSE to Responses API SSE")

	wire := string(bodyMutation.GetBody())
	requireInitialResponseAPIStreamingWire(t, wire)

	finalChunk := []byte(`data: {"id":"chatcmpl-repro","object":"chat.completion.chunk","created":123,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"prompt_tokens_details":{"cached_tokens":1},"completion_tokens":1,"completion_tokens_details":{"reasoning_tokens":0},"total_tokens":3}}` + "\n\n" + `data: [DONE]` + "\n\n")
	finalResp, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: finalChunk},
	}, ctx)
	require.NoError(t, err)
	require.NotNil(t, finalResp)

	finalBodyMutation := finalResp.GetResponseBody().GetResponse().GetBodyMutation()
	require.NotNil(t, finalBodyMutation, "Responses API stream:true must translate terminal SSE events")

	finalWire := string(finalBodyMutation.GetBody())
	requireTerminalResponseAPIStreamingWire(t, finalWire)
	requireResponseAPIStreamingLifecycle(t, wire, finalWire)
}

func requireInitialResponseAPIStreamingWire(t *testing.T, wire string) {
	t.Helper()

	require.Contains(t, wire, "response.created")
	require.Contains(t, wire, "response.in_progress")
	require.Contains(t, wire, "response.content_part.added")
	require.Contains(t, wire, "response.output_text.delta")
	require.Contains(t, wire, `"annotations":[]`)
	require.NotContains(t, wire, "sequence_number")
	require.NotContains(t, wire, "chat.completion.chunk")
	require.Contains(t, wire, "event:", "Responses API SSE should include event names")
}

func requireTerminalResponseAPIStreamingWire(t *testing.T, finalWire string) {
	t.Helper()

	require.Contains(t, finalWire, "response.output_text.done")
	require.Contains(t, finalWire, "response.completed")
	require.Contains(t, finalWire, `"annotations":[]`)
	require.NotContains(t, finalWire, "sequence_number")
	require.NotContains(t, finalWire, "data: [DONE]")
	require.Contains(t, finalWire, `"output_text":"foo"`)
	require.Contains(t, finalWire, `"total_tokens":3`)
}

func requireResponseAPIStreamingLifecycle(t *testing.T, wire string, finalWire string) {
	t.Helper()

	createdEvent := responseAPIStreamingEventPayload(t, wire, "response.created")
	completedEvent := responseAPIStreamingEventPayload(t, finalWire, "response.completed")
	createdResponse, ok := createdEvent["response"].(map[string]interface{})
	require.True(t, ok)
	completedResponse, ok := completedEvent["response"].(map[string]interface{})
	require.True(t, ok)
	require.Equal(t, createdResponse["id"], completedResponse["id"])
	require.Equal(t, createdResponse["created_at"], completedResponse["created_at"])
	require.Equal(t, responseapi.StatusInProgress, createdResponse["status"])
	require.Equal(t, responseapi.StatusCompleted, completedResponse["status"])
	require.Equal(t, float64(1), createdResponse["temperature"])
	require.Equal(t, float64(1), createdResponse["top_p"])
	usage, ok := completedResponse["usage"].(map[string]interface{})
	require.True(t, ok)
	require.Equal(t, float64(3), usage["total_tokens"])
	require.Equal(t, map[string]interface{}{"cached_tokens": float64(1)}, usage["input_tokens_details"])
	require.Equal(t, map[string]interface{}{"reasoning_tokens": float64(0)}, usage["output_tokens_details"])
}

func TestResponseAPIStreamingResponseBuffersSplitChatCompletionSSE(t *testing.T) {
	router := &OpenAIRouter{
		Config:            &config.RouterConfig{},
		ResponseAPIFilter: NewResponseAPIFilter(NewMockResponseStore()),
	}
	ctx := newResponseAPIStreamingTestContext("response-api-stream-split")

	firstHalf := []byte(`data: {"id":"chatcmpl-split","object":"chat.completion.chunk"`)
	firstResp, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: firstHalf},
	}, ctx)
	require.NoError(t, err)
	require.NotNil(t, firstResp)

	firstMutation := firstResp.GetResponseBody().GetResponse().GetBodyMutation()
	require.NotNil(t, firstMutation, "partial upstream SSE frames must not pass through untranslated")
	require.Empty(t, firstMutation.GetBody())

	secondHalf := []byte(`,"created":123,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}]}` + "\n\n")
	secondResp, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: secondHalf},
	}, ctx)
	require.NoError(t, err)
	require.NotNil(t, secondResp)

	secondWire := string(secondResp.GetResponseBody().GetResponse().GetBodyMutation().GetBody())
	require.Contains(t, secondWire, "response.output_text.delta")
	require.Contains(t, secondWire, `"delta":"lo"`)
	require.NotContains(t, secondWire, "sequence_number")
	require.NotContains(t, secondWire, "chat.completion.chunk")
}

func TestResponseAPIStreamingResponseTranslatesToolCalls(t *testing.T) {
	store := NewMockResponseStore()
	router := &OpenAIRouter{
		Config:            &config.RouterConfig{},
		ResponseAPIFilter: NewResponseAPIFilter(store),
	}
	ctx := newResponseAPIStreamingTestContext("response-api-stream-tools")
	ctx.ResponseAPICtx.OriginalRequest.PreviousResponseID = "resp_previous"
	chunk := []byte(`data: {"id":"chatcmpl-tools","object":"chat.completion.chunk","created":123,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_weather","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},"finish_reason":null}]}` + "\n\n")

	resp, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: chunk},
	}, ctx)
	require.NoError(t, err)

	wire := string(resp.GetResponseBody().GetResponse().GetBodyMutation().GetBody())
	require.Contains(t, wire, "response.output_item.added")
	require.Contains(t, wire, "response.function_call_arguments.delta")
	require.Contains(t, wire, "call_weather")
	require.NotContains(t, wire, "chat.completion.chunk")

	finalChunk := []byte(`data: {"id":"chatcmpl-tools","object":"chat.completion.chunk","created":123,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}` + "\n\n" + `data: [DONE]` + "\n\n")
	finalResp, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: finalChunk},
	}, ctx)
	require.NoError(t, err)

	finalWire := string(finalResp.GetResponseBody().GetResponse().GetBodyMutation().GetBody())
	require.Contains(t, finalWire, "response.function_call_arguments.done")
	require.Contains(t, finalWire, "response.output_item.done")
	require.Contains(t, finalWire, "response.completed")
	require.Contains(t, finalWire, `"type":"function_call"`)
	require.Contains(t, finalWire, `"arguments":"{\"city\":\"Paris\"}"`)
	require.NotContains(t, finalWire, "data: [DONE]")

	stored, err := store.GetResponse(context.Background(), ctx.ResponseAPICtx.GeneratedResponseID)
	require.NoError(t, err)
	require.Len(t, stored.Output, 1)
	require.Equal(t, "resp_previous", stored.PreviousResponseID)
	require.Equal(t, responseapi.ItemTypeFunctionCall, stored.Output[0].Type)
	require.Equal(t, "call_weather", stored.Output[0].CallID)
}

func TestResponseAPIStreamingResponseTranslatesRefusalAndReasoning(t *testing.T) {
	router := &OpenAIRouter{
		Config:            &config.RouterConfig{},
		ResponseAPIFilter: NewResponseAPIFilter(NewMockResponseStore()),
	}
	ctx := newResponseAPIStreamingTestContext("response-api-stream-non-text")
	chunk := []byte(`data: {"id":"chatcmpl-refusal","object":"chat.completion.chunk","created":123,"model":"gpt-4o","choices":[{"index":0,"delta":{"reasoning_content":"checking policy","refusal":"I can't help with that"},"finish_reason":null}]}` + "\n\n")

	resp, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: chunk},
	}, ctx)
	require.NoError(t, err)

	wire := string(resp.GetResponseBody().GetResponse().GetBodyMutation().GetBody())
	require.Contains(t, wire, "response.reasoning_text.delta")
	require.Contains(t, wire, "response.refusal.delta")
	require.NotContains(t, wire, "chat.completion.chunk")

	finalChunk := []byte(`data: {"id":"chatcmpl-refusal","object":"chat.completion.chunk","created":123,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n" + `data: [DONE]` + "\n\n")
	finalResp, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: finalChunk},
	}, ctx)
	require.NoError(t, err)

	finalWire := string(finalResp.GetResponseBody().GetResponse().GetBodyMutation().GetBody())
	require.Contains(t, finalWire, "response.reasoning_text.done")
	require.Contains(t, finalWire, "response.refusal.done")
	require.Contains(t, finalWire, "response.completed")
	require.NotContains(t, finalWire, "data: [DONE]")
}

func TestResponseAPIStreamingCacheHitUsesResponsesSSE(t *testing.T) {
	store := NewMockResponseStore()
	router := &OpenAIRouter{
		Config:            &config.RouterConfig{},
		ResponseAPIFilter: NewResponseAPIFilter(store),
	}
	ctx := newResponseAPIStreamingTestContext("response-api-cache-hit")
	cached := []byte(`{"id":"chatcmpl-cache","object":"chat.completion","created":123,"model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"cached answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`)

	resp, ok := router.createResponseAPICacheHitResponse(ctx, cached, "math", "math_decision", nil, 0)
	require.True(t, ok)
	require.NotNil(t, resp.GetImmediateResponse())

	wire := string(resp.GetImmediateResponse().Body)
	require.Contains(t, wire, "event: response.created")
	require.Contains(t, wire, "event: response.output_text.delta")
	require.Contains(t, wire, "event: response.completed")
	require.Contains(t, wire, "cached answer")
	require.NotContains(t, wire, "chat.completion.chunk")
	require.NotContains(t, wire, "data: [DONE]")
	require.Equal(t, "text/event-stream; charset=utf-8", immediateHeaderValue(resp, "content-type"))
	require.Equal(t, "true", immediateHeaderValue(resp, "x-vsr-cache-hit"))

	_, err := store.GetResponse(context.Background(), ctx.ResponseAPICtx.GeneratedResponseID)
	require.NoError(t, err)
}

func TestResponseAPIStreamingFastResponseUsesResponsesSSE(t *testing.T) {
	router := &OpenAIRouter{
		Config:            &config.RouterConfig{},
		ResponseAPIFilter: NewResponseAPIFilter(NewMockResponseStore()),
	}
	ctx := newResponseAPIStreamingTestContext("response-api-fast-response")

	resp, ok := router.createResponseAPIFastResponse(ctx, "blocked by policy", "guard_decision")
	require.True(t, ok)
	require.NotNil(t, resp.GetImmediateResponse())

	wire := string(resp.GetImmediateResponse().Body)
	require.Contains(t, wire, "event: response.output_text.delta")
	require.Contains(t, wire, "blocked by policy")
	require.NotContains(t, wire, "chat.completion.chunk")
	require.NotContains(t, wire, "data: [DONE]")
	require.Equal(t, "text/event-stream; charset=utf-8", immediateHeaderValue(resp, "content-type"))
	require.Equal(t, "true", immediateHeaderValue(resp, "x-vsr-fast-response"))
}

func responseAPIStreamingEventPayload(t *testing.T, wire string, event string) map[string]interface{} {
	t.Helper()

	for _, frame := range strings.Split(wire, "\n\n") {
		if !strings.Contains(frame, "event: "+event) {
			continue
		}
		for _, line := range strings.Split(frame, "\n") {
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			var payload map[string]interface{}
			require.NoError(t, json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &payload))
			return payload
		}
	}

	require.Failf(t, "missing event", "event %q not found in stream: %s", event, wire)
	return nil
}

func immediateHeaderValue(resp *ext_proc.ProcessingResponse, key string) string {
	if resp == nil || resp.GetImmediateResponse() == nil || resp.GetImmediateResponse().Headers == nil {
		return ""
	}
	for _, header := range resp.GetImmediateResponse().Headers.SetHeaders {
		if strings.EqualFold(header.Header.Key, key) {
			if len(header.Header.RawValue) > 0 {
				return string(header.Header.RawValue)
			}
			return header.Header.Value
		}
	}
	return ""
}

func newResponseAPIStreamingTestContext(requestID string) *RequestContext {
	return &RequestContext{
		RequestID:               requestID,
		RequestModel:            "gpt-4o",
		StartTime:               time.Now(),
		ProcessingStartTime:     time.Now(),
		TraceContext:            context.Background(),
		IsStreamingResponse:     true,
		ExpectStreamingResponse: true,
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			OriginalRequest: &responseapi.ResponseAPIRequest{
				Model:  "gpt-4o",
				Input:  json.RawMessage(`"hello"`),
				Stream: true,
			},
			GeneratedResponseID: responseapi.GenerateResponseID(),
		},
	}
}
