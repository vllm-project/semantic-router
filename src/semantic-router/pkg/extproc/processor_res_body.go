package extproc

import (
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// handleResponseBody processes the response body.
func (r *OpenAIRouter) handleResponseBody(v *ext_proc.ProcessingRequest_ResponseBody, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	if skipResponse := r.handleSkipProcessingResponseBody(v.ResponseBody.Body, ctx); skipResponse != nil {
		return skipResponse, nil
	}

	completionLatency := time.Since(ctx.StartTime)
	if !ctx.IsStreamingResponse || v.ResponseBody.GetEndOfStream() {
		defer recordSessionTurnOutcome(ctx, responseUsageMetrics{})
	}

	responseBody, decoded, failure := r.decodeUpstreamResponseBody(v.ResponseBody.Body, ctx)
	if failure != nil {
		return failure, nil
	}
	response := r.handleDecodedResponseBody(responseBody, v.ResponseBody.GetEndOfStream(), completionLatency, ctx)
	deliverUpstreamResponseBody(response, ctx, responseBody, decoded)
	return response, nil
}

// handleDecodedResponseBody runs the response phases on the upstream body as
// the router reads it, after any content coding is removed.
func (r *OpenAIRouter) handleDecodedResponseBody(
	responseBody []byte,
	endOfStream bool,
	completionLatency time.Duration,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	if looperResponse := r.handleLooperResponseBody(responseBody, endOfStream, ctx); looperResponse != nil {
		return looperResponse
	}

	if isUpstreamTransportError(ctx) {
		return r.handleUpstreamTransportError(responseBody, ctx)
	}

	if ctx.IsStreamingResponse {
		return r.handleSemanticStreamingResponseBody(responseBody, endOfStream, ctx)
	}
	recoveredBody, recoveryErr := r.handleContextRecoveryFollowup(
		ctx.TraceContext,
		responseBody,
		ctx,
	)
	if recoveryErr != nil {
		logging.ComponentWarnEvent("extproc", "context_recovery_followup_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"error":      recoveryErr.Error(),
		})
		if contextRecoveryFailClosed(ctx) {
			return r.createErrorResponse(502, "Context recovery followup failed")
		}
		responseBody = r.redactContextRecoveryToolCalls(responseBody, ctx)
	} else {
		responseBody = recoveredBody
	}

	return r.handleNonStreamingResponseBody(responseBody, ctx, completionLatency)
}

func contextRecoveryFailClosed(ctx *RequestContext) bool {
	if ctx == nil || ctx.VSRSelectedDecision == nil {
		return false
	}
	plugin := ctx.VSRSelectedDecision.GetContextCompressionConfig()
	return plugin != nil &&
		plugin.EffectiveFailureMode() == config.ContextCompressionFailureClosed
}

func buildResponseBodyContinueResponse(
	bodyMutation *ext_proc.BodyMutation,
	headerMutation *ext_proc.HeaderMutation,
) *ext_proc.ProcessingResponse {
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ResponseBody{
			ResponseBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					Status:         ext_proc.CommonResponse_CONTINUE,
					HeaderMutation: headerMutation,
					BodyMutation:   bodyMutation,
				},
			},
		},
	}
}
