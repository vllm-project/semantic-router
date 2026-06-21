package extproc

import (
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/anthropic"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// handleAnthropicStreamingResponseBody translates Anthropic SSE into OpenAI
// chat.completion.chunk SSE, then reuses the standard streaming accumulator path.
func (r *OpenAIRouter) handleAnthropicStreamingResponseBody(
	responseBody []byte,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	recordStreamingTTFT(ctx)
	ensureStreamingState(ctx)

	if ctx.AnthropicStream == nil {
		ctx.AnthropicStream = anthropic.NewStreamState()
	}

	// Pass IRExtensions through so the merged Anthropic→OpenAI cell
	// captures cache counters, stop_reason round-trip fields, and
	// per-block thinking signatures onto the request-scoped sidecar.
	// Even for an OpenAI client the IR fields keep observability,
	// router replay, and downstream non-streaming responses consistent
	// with the streaming case.
	transformed, streamDone, err := anthropic.TransformSSEChunkToOpenAI(
		responseBody,
		ctx.AnthropicStream,
		ctx.RequestModel,
		ctx.IRExtensions,
	)
	if err != nil {
		logging.Errorf("Failed to transform Anthropic streaming chunk: %v", err)
		return buildResponseBodyContinueResponse(nil, nil)
	}
	if len(transformed) == 0 {
		if streamDone {
			r.finalizeStreamingResponse(ctx)
		}
		return buildResponseBodyContinueResponse(nil, nil)
	}

	chunkStr := string(transformed)
	ctx.HasStreamingChunks = true
	r.parseStreamingChunk(chunkStr, ctx)

	if strings.Contains(chunkStr, "data: [DONE]") || streamDone {
		r.finalizeStreamingResponse(ctx)
	}

	return buildResponseBodyContinueResponse(&ext_proc.BodyMutation{
		Mutation: &ext_proc.BodyMutation_Body{Body: transformed},
	}, nil)
}
