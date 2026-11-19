package extproc

import (
	"strconv"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Internal hops skip final response policy and persistence, which belong to
// the parent request, while retaining the ordinary provider transport contract.
// A native provider response must be returned in the caller's Chat wire format.
func (r *OpenAIRouter) handleLooperResponseBody(
	responseBody []byte,
	endOfStream bool,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	if ctx == nil || !ctx.LooperRequest {
		return nil
	}
	if isUpstreamTransportError(ctx) {
		return r.handleUpstreamTransportError(responseBody, ctx)
	}
	if ctx.IsStreamingResponse {
		return r.handleLooperStreamingResponseBody(responseBody, endOfStream, ctx)
	}

	response := buildResponseBodyContinueResponse(nil, nil)
	var semantic *llmprotocol.Response
	if requiresClientResponseRewrite(ctx) {
		var err error
		if semantic, err = r.decodeClientResponse(responseBody, ctx); err != nil {
			return r.invalidUpstreamResponse(ctx, "The selected model returned an invalid response")
		}
		responseBody, err = r.encodeClientResponse(*semantic, ctx)
		if err != nil {
			return r.invalidUpstreamResponse(ctx, "The selected model returned an incompatible response")
		}
		setResponseBodyMutation(response, responseBody)
		setResponseContentType(response, "application/json")
	}
	if cost, currency, priced := r.settleHopSpend(ctx, semantic, responseBody); priced {
		// The Looper prices the call at this, not at its Chat projection.
		setResponseBodyHeader(response, headers.VSRCost, strconv.FormatFloat(cost, 'f', -1, 64))
		setResponseBodyHeader(response, headers.VSRCostCurrency, currency)
	}
	// Same-wire Chat remains byte-preserving. An empty completion can still
	// carry paid usage, which the calling algorithm classifies and accounts.
	r.attachRouterReplayResponse(ctx, responseBody, true)
	return response
}

func (r *OpenAIRouter) handleLooperStreamingResponseBody(
	responseBody []byte,
	endOfStream bool,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	if err := r.ensureSemanticResponseStream(ctx); err != nil {
		return r.invalidUpstreamResponse(ctx, "The selected model returned an incompatible stream")
	}
	buffers := semanticStreamBuffers{}
	buffers.push(responseBody, ctx)
	if endOfStream {
		buffers.finalize(ctx)
		ctx.StreamingComplete = true
		clean := buffers.streamErr == nil && !ctx.StreamingAborted
		semantic, err := ctx.SemanticStreamState.response()
		if err != nil {
			ctx.StreamingAborted = true
		} else {
			r.settleHopSpend(ctx, semantic, nil)
			if body, err := r.encodeClientResponse(*semantic, ctx); err == nil {
				r.attachRouterReplayResponse(ctx, body, true)
			}
		}
		r.endStreamRouterReplay(ctx, clean)
	}
	return buffers.processingResponse(ctx)
}
