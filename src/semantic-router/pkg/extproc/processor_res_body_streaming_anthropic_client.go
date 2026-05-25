package extproc

import (
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/anthropic"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// handleAnthropicClientStreamingResponseBody dispatches streaming
// responses for clients that called the /v1/messages endpoint
// (ClientProtocol == "anthropic"). The body Envoy hands us may be in
// either Anthropic SSE (when the upstream is an Anthropic backend) or
// OpenAI SSE (when the upstream is an OpenAI backend); both arrive on
// this path because the inbound parser already pinned the client to
// the Anthropic protocol. The function:
//
//  1. translates Anthropic upstream SSE to OpenAI SSE first, so the
//     accounting accumulator, plugins, and the outbound emitter all
//     see the uniform OpenAI shape vsr's pipeline expects;
//  2. runs the standard OpenAI streaming accumulator to update the
//     per-request StreamingMetadata / usage (and flag HasStreamingChunks)
//     so metrics, cache, and replay continue to work;
//  3. re-emits the OpenAI chunk as Anthropic SSE via
//     anthropic.EmitAnthropicSSEChunk and returns those bytes via
//     BodyMutation so the client sees Anthropic-shape events.
//
// This is the symmetric counterpart to
// handleAnthropicStreamingResponseBody (which handles OpenAI clients
// hitting an Anthropic backend); the two share the AnthropicStream
// state but only one runs per request because ClientProtocol is pinned
// at the inbound parser.
func (r *OpenAIRouter) handleAnthropicClientStreamingResponseBody(
	responseBody []byte,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	recordStreamingTTFT(ctx)
	ensureStreamingState(ctx)
	if ctx.AnthropicStream == nil {
		ctx.AnthropicStream = anthropic.NewStreamState()
	}

	openAIBytes, err := r.translateUpstreamToOpenAI(responseBody, ctx)
	if err != nil {
		logging.Errorf("Failed to translate upstream streaming chunk to OpenAI: %v", err)
		return buildResponseBodyContinueResponse(nil, nil)
	}

	chunkStr := string(openAIBytes)
	if chunkStr != "" {
		ctx.HasStreamingChunks = true
		r.parseStreamingChunk(chunkStr, ctx)
	}

	anthropicBytes, streamDone, err := anthropic.EmitAnthropicSSEChunk(
		openAIBytes, ctx.AnthropicStream, ctx.IRExtensions, ctx.RequestModel,
	)
	if err != nil {
		logging.Errorf("Failed to emit Anthropic SSE chunk: %v", err)
		return buildResponseBodyContinueResponse(nil, nil)
	}

	if streamDone || strings.Contains(chunkStr, "data: [DONE]") {
		r.finalizeStreamingResponse(ctx)
	}

	// Always replace the body, even when our emitter produced zero
	// bytes for this chunk. A nil BodyMutation makes Envoy forward the
	// raw upstream chunk unmodified, which would leak Anthropic-shape
	// bytes from the backend onto the wire on top of the Anthropic
	// events we have already synthesized — most visibly a second
	// message_stop event after the emitter's own message_stop. Once we
	// have entered this handler we own the wire, so an empty mutation
	// is the correct way to swallow upstream noise.
	return buildResponseBodyContinueResponse(&ext_proc.BodyMutation{
		Mutation: &ext_proc.BodyMutation_Body{Body: anthropicBytes},
	}, nil)
}

// translateUpstreamToOpenAI normalizes the upstream SSE bytes into the
// OpenAI chunk shape. Anthropic upstreams go through
// TransformSSEChunkToOpenAI so plugins and the emitter see uniform IR;
// OpenAI upstreams pass through unchanged. Returns the normalized
// bytes and any translator error.
func (r *OpenAIRouter) translateUpstreamToOpenAI(
	responseBody []byte,
	ctx *RequestContext,
) ([]byte, error) {
	if ctx.APIFormat != config.APIFormatAnthropic {
		return responseBody, nil
	}
	translated, _, err := anthropic.TransformSSEChunkToOpenAI(
		responseBody, ctx.AnthropicStream, ctx.RequestModel, ctx.IRExtensions,
	)
	if err != nil {
		return nil, err
	}
	return translated, nil
}
