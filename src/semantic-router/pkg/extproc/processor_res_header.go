package extproc

import (
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

// handleResponseHeaders processes the response headers.
func (r *OpenAIRouter) handleResponseHeaders(v *ext_proc.ProcessingRequest_ResponseHeaders, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	if looperResp := r.handleLooperResponseHeaders(v, ctx); looperResp != nil {
		return looperResp, nil
	}

	outcome := evaluateResponseHeaderOutcome(v, ctx)
	finishUpstreamResponseSpan(ctx, outcome)
	maybeRecordResponseHeaderTTFT(ctx)
	r.updateRouterReplayStatus(ctx, outcome.statusCode, ctx != nil && ctx.IsStreamingResponse)

	headerMutation := buildResponseHeaderMutation(ctx, outcome.isSuccessful)
	return buildResponseHeadersContinueResponse(headerMutation, ctx != nil && ctx.IsStreamingResponse), nil
}
