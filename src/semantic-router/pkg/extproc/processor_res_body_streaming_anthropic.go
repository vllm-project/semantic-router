package extproc

import (
	"fmt"
	"strings"
	"sync"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/anthropic"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// anthropicStreamStates holds per-request Anthropic→OpenAI SSE translation state.
// Entries are keyed by RequestID (or request context pointer when RequestID is unset)
// and must be removed on stream completion or ext_proc stream teardown (see
// releaseAnthropicStreamState and handleProcessReceiveError).
var anthropicStreamStates sync.Map

func anthropicStreamStateKey(ctx *RequestContext) string {
	if ctx == nil {
		return ""
	}
	if ctx.RequestID != "" {
		return ctx.RequestID
	}
	return fmt.Sprintf("%p", ctx)
}

func getAnthropicStreamState(ctx *RequestContext) *anthropic.StreamState {
	key := anthropicStreamStateKey(ctx)
	if key == "" {
		return anthropic.NewStreamState()
	}
	if v, ok := anthropicStreamStates.Load(key); ok {
		return v.(*anthropic.StreamState)
	}
	state := anthropic.NewStreamState()
	actual, _ := anthropicStreamStates.LoadOrStore(key, state)
	return actual.(*anthropic.StreamState)
}

func releaseAnthropicStreamState(ctx *RequestContext) {
	if key := anthropicStreamStateKey(ctx); key != "" {
		anthropicStreamStates.Delete(key)
	}
}

// handleAnthropicStreamingResponseBody translates Anthropic SSE into OpenAI
// chat.completion.chunk SSE, then reuses the standard streaming accumulator path.
func (r *OpenAIRouter) handleAnthropicStreamingResponseBody(
	responseBody []byte,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	recordStreamingTTFT(ctx)
	ensureStreamingState(ctx)

	streamState := getAnthropicStreamState(ctx)
	transformed, streamDone, err := anthropic.TransformSSEChunkToOpenAI(
		responseBody,
		streamState,
		ctx.RequestModel,
	)
	if err != nil {
		logging.Errorf("Failed to transform Anthropic streaming chunk: %v", err)
		releaseAnthropicStreamState(ctx)
		return buildResponseBodyContinueResponse(nil, nil)
	}
	if len(transformed) == 0 {
		if streamDone {
			r.finalizeStreamingResponse(ctx)
			releaseAnthropicStreamState(ctx)
		}
		return buildResponseBodyContinueResponse(nil, nil)
	}

	chunkStr := string(transformed)
	ctx.StreamingChunks = append(ctx.StreamingChunks, chunkStr)
	r.parseStreamingChunk(chunkStr, ctx)

	if strings.Contains(chunkStr, "data: [DONE]") || streamDone {
		r.finalizeStreamingResponse(ctx)
		releaseAnthropicStreamState(ctx)
	}

	return buildResponseBodyContinueResponse(&ext_proc.BodyMutation{
		Mutation: &ext_proc.BodyMutation_Body{Body: transformed},
	}, nil)
}
