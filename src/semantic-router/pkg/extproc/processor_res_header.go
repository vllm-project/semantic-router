package extproc

import (
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

// handleResponseHeaders processes the response headers.
func (r *OpenAIRouter) handleResponseHeaders(v *ext_proc.ProcessingRequest_ResponseHeaders, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	if skipResp := r.handleSkipProcessingResponseHeaders(v, ctx); skipResp != nil {
		return skipResp, nil
	}
	if looperResp := r.handleLooperResponseHeaders(v, ctx); looperResp != nil {
		return looperResp, nil
	}

	outcome := evaluateResponseHeaderOutcome(v, ctx)
	if ctx != nil {
		// Persist the upstream status so the later response-body cache-write
		// path can avoid caching non-2xx error bodies (cache poisoning).
		ctx.UpstreamStatusCode = outcome.statusCode
	}
	finishUpstreamResponseSpan(ctx, outcome)
	maybeRecordResponseHeaderTTFT(ctx)
	r.updateRouterReplayStatus(ctx, outcome.statusCode, ctx != nil && ctx.IsStreamingResponse)
	r.observeRouterLearningProviderStatus(ctx, outcome.statusCode)

	headerMutation := buildResponseHeaderMutation(ctx, outcome.isSuccessful)
	return buildResponseHeadersContinueResponse(headerMutation, ctx != nil && ctx.IsStreamingResponse), nil
}
