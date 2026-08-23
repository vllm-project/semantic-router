package extproc

import (
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
)

// handleResponseBody processes the response body.
func (r *OpenAIRouter) handleResponseBody(v *ext_proc.ProcessingRequest_ResponseBody, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	completionLatency := time.Since(ctx.StartTime)

	// Decrement active request count for queue depth estimation.
	defer metrics.DecrementModelActiveRequests(ctx.RequestModel)

	if looperResponse := r.handleLooperResponseBody(v.ResponseBody.Body, ctx); looperResponse != nil {
		return looperResponse, nil
	}

	responseBody := v.ResponseBody.Body

	if ctx.IsStreamingResponse {
		return r.handleSemanticStreamingResponseBody(responseBody, v.ResponseBody.GetEndOfStream(), ctx), nil
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
			usage := r.takeNeutralResponseUsage(ctx)
			if settleErr := r.completeAndSettlePrimaryInference(ctx, usage, 502); settleErr != nil {
				logging.Errorf("failed to settle context recovery response: %v", settleErr)
				return r.createInferenceAccessError(quotaruntime.AdmissionUnavailable, nil), nil
			}
			return r.createErrorResponse(502, "Context recovery followup failed"), nil
		}
		responseBody = r.redactContextRecoveryToolCalls(responseBody, ctx)
	} else {
		responseBody = recoveredBody
	}

	return r.handleNonStreamingResponseBody(responseBody, ctx, completionLatency), nil
}

func contextRecoveryFailClosed(ctx *RequestContext) bool {
	if ctx == nil || ctx.VSRSelectedDecision == nil {
		return false
	}
	plugin := ctx.VSRSelectedDecision.GetContextCompressionConfig()
	return plugin != nil &&
		plugin.EffectiveFailureMode() == config.ContextCompressionFailureClosed
}

func (r *OpenAIRouter) handleLooperResponseBody(
	responseBody []byte,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	if !ctx.LooperRequest {
		return nil
	}

	logging.Debugf("[Looper] Capturing response body for router replay")
	r.attachRouterReplayResponse(ctx, responseBody, true)
	return buildResponseBodyContinueResponse(nil, nil)
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
