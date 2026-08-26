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
		return buildResponseBodyContinueResponse(r.anthropicStreamingChunkMutation(nil, ctx))
	}
	if len(transformed) == 0 {
		if streamDone {
			r.finalizeStreamingResponse(ctx)
		}
		return buildResponseBodyContinueResponse(r.anthropicStreamingChunkMutation(nil, ctx))
	}

	chunkStr := string(transformed)
	ctx.HasStreamingChunks = true
	r.parseStreamingChunk(chunkStr, ctx)

	// Build the outbound chunk before finalizing: the Response API
	// [DONE]-driven terminal events read the accumulated streaming state,
	// and finalization must observe the same ordering as
	// handleStreamingResponseBody's Response API branch.
	bodyMutation, headerMutation := r.anthropicStreamingChunkMutation(transformed, ctx)

	if strings.Contains(chunkStr, "data: [DONE]") || streamDone {
		r.finalizeStreamingResponse(ctx)
	}

	return buildResponseBodyContinueResponse(bodyMutation, headerMutation)
}

// anthropicStreamingChunkMutation wraps a transformed chat.completion.chunk
// SSE fragment in the body/header mutations the client expects. Response API
// clients have no dedicated ClientProtocol value (Responses-ness lives on
// ctx.ResponseAPICtx), so they reach this handler like plain OpenAI clients;
// their chunks go through the same Response API streaming mutation that
// handleStreamingResponseBody applies for OpenAI-format backends. The
// mutation is applied even for empty fragments so raw Anthropic frames
// (ping, content_block_stop) are always replaced and never leak to a
// /v1/responses stream. OpenAI clients keep the legacy contract: their
// chunk bytes pass through untouched when there is nothing to emit.
func (r *OpenAIRouter) anthropicStreamingChunkMutation(
	transformed []byte,
	ctx *RequestContext,
) (*ext_proc.BodyMutation, *ext_proc.HeaderMutation) {
	if isResponseAPIRequest(ctx) {
		return r.buildResponseAPIStreamingBodyMutation(transformed, ctx), responseAPIStreamingHeaderMutation()
	}
	if len(transformed) == 0 {
		return nil, nil
	}
	return &ext_proc.BodyMutation{
		Mutation: &ext_proc.BodyMutation_Body{Body: transformed},
	}, nil
}
