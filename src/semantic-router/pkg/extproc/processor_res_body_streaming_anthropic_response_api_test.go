package extproc

import (
	"context"
	"strings"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// A /v1/responses stream backed by an api_format:anthropic model must
// receive Response API SSE events, not the chat.completion.chunk frames the
// Anthropic streaming handler produces as its intermediate representation.

func newAnthropicResponseAPIStreamingTestContext(requestID string) *RequestContext {
	ctx := newResponseAPIStreamingTestContext(requestID)
	ctx.RequestModel = "claude-sonnet-4-5"
	ctx.ResponseAPICtx.OriginalRequest.Model = "claude-sonnet-4-5"
	ctx.APIFormat = config.APIFormatAnthropic
	return ctx
}

func anthropicSSEFrames(frames ...[2]string) []byte {
	var out strings.Builder
	for _, frame := range frames {
		out.WriteString("event: " + frame[0] + "\n")
		out.WriteString("data: " + frame[1] + "\n\n")
	}
	return []byte(out.String())
}

// anthropicStreamFixture is the canonical upstream Anthropic SSE stream
// shared by the streaming cell tests: every client cell (OpenAI,
// Response API) consumes the same five frames, so the tests differ only
// in the client context and the wire-format assertions.
var anthropicStreamFixture = [][2]string{
	{"message_start", `{"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-5","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":0}}}`},
	{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
	{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}`},
	{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}`},
	{"message_stop", `{"type":"message_stop"}`},
}

func TestResponseAPIStreamingAnthropicBackendEmitsResponsesSSE(t *testing.T) {
	store := NewMockResponseStore()
	router := &OpenAIRouter{
		Config:            &config.RouterConfig{},
		ResponseAPIFilter: NewResponseAPIFilter(store),
	}
	ctx := newAnthropicResponseAPIStreamingTestContext("response-api-anthropic-stream")

	firstChunk := anthropicSSEFrames(anthropicStreamFixture[:3]...)
	resp, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: firstChunk},
	}, ctx)
	require.NoError(t, err)
	require.NotNil(t, resp)

	bodyMutation := resp.GetResponseBody().GetResponse().GetBodyMutation()
	require.NotNil(t, bodyMutation, "Anthropic-backend Responses stream must rewrite upstream SSE")

	wire := string(bodyMutation.GetBody())
	requireInitialResponseAPIStreamingWire(t, wire)
	require.Contains(t, wire, `"delta":"Hi"`)
	require.NotContains(t, wire, "message_start")
	require.NotContains(t, wire, "content_block_delta")

	finalChunk := anthropicSSEFrames(anthropicStreamFixture[3:]...)
	finalResp, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: finalChunk},
	}, ctx)
	require.NoError(t, err)
	require.NotNil(t, finalResp)

	finalBodyMutation := finalResp.GetResponseBody().GetResponse().GetBodyMutation()
	require.NotNil(t, finalBodyMutation, "terminal Anthropic SSE must translate to Response API events")

	finalWire := string(finalBodyMutation.GetBody())
	require.Contains(t, finalWire, "response.output_text.done")
	require.Contains(t, finalWire, "response.completed")
	require.Contains(t, finalWire, `"output_text":"Hi"`)
	require.NotContains(t, finalWire, "data: [DONE]")
	require.NotContains(t, finalWire, "chat.completion.chunk")
	require.NotContains(t, finalWire, "message_stop")
	requireAnthropicResponseAPIStreamingLifecycle(t, wire, finalWire)
	require.True(t, ctx.StreamingComplete)

	stored, err := store.GetResponse(context.Background(), ctx.ResponseAPICtx.GeneratedResponseID)
	require.NoError(t, err)
	require.Equal(t, responseapi.StatusCompleted, stored.Status)
}

func requireAnthropicResponseAPIStreamingLifecycle(t *testing.T, wire string, finalWire string) {
	t.Helper()

	createdEvent := responseAPIStreamingEventPayload(t, wire, "response.created")
	completedEvent := responseAPIStreamingEventPayload(t, finalWire, "response.completed")
	createdResponse, ok := createdEvent["response"].(map[string]interface{})
	require.True(t, ok)
	completedResponse, ok := completedEvent["response"].(map[string]interface{})
	require.True(t, ok)
	require.Equal(t, createdResponse["id"], completedResponse["id"])
	require.Equal(t, responseapi.StatusInProgress, createdResponse["status"])
	require.Equal(t, responseapi.StatusCompleted, completedResponse["status"])
	usage, ok := completedResponse["usage"].(map[string]interface{})
	require.True(t, ok)
	require.Equal(t, float64(1), usage["output_tokens"])
}

func TestResponseAPIStreamingAnthropicBackendSuppressesUntranslatedFrames(t *testing.T) {
	router := &OpenAIRouter{
		Config:            &config.RouterConfig{},
		ResponseAPIFilter: NewResponseAPIFilter(NewMockResponseStore()),
	}
	ctx := newAnthropicResponseAPIStreamingTestContext("response-api-anthropic-noise")

	// ping and content_block_stop translate to no OpenAI chunks; without a
	// body mutation Envoy would forward the raw Anthropic frames to the
	// /v1/responses client, interleaved into the Response API stream.
	noiseChunk := anthropicSSEFrames(
		[2]string{"ping", `{"type":"ping"}`},
		[2]string{"content_block_stop", `{"type":"content_block_stop","index":0}`},
	)
	resp, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: noiseChunk},
	}, ctx)
	require.NoError(t, err)
	require.NotNil(t, resp)

	bodyMutation := resp.GetResponseBody().GetResponse().GetBodyMutation()
	require.NotNil(t, bodyMutation, "untranslatable Anthropic frames must be replaced, not passed through")
	require.Empty(t, bodyMutation.GetBody())
}

func TestAnthropicStreamingOpenAIClientKeepsChatCompletionChunks(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{},
	}
	ctx := &RequestContext{
		RequestID:           "openai-client-anthropic-stream",
		RequestModel:        "claude-sonnet-4-5",
		StartTime:           time.Now(),
		ProcessingStartTime: time.Now(),
		TraceContext:        context.Background(),
		IsStreamingResponse: true,
		APIFormat:           config.APIFormatAnthropic,
	}

	chunk := anthropicSSEFrames(anthropicStreamFixture...)
	resp, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: chunk},
	}, ctx)
	require.NoError(t, err)
	require.NotNil(t, resp)

	wire := string(resp.GetResponseBody().GetResponse().GetBodyMutation().GetBody())
	require.Contains(t, wire, "chat.completion.chunk")
	require.NotContains(t, wire, "response.created")
}
