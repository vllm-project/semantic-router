package extproc

import (
	"fmt"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/anthropic"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// handleResponseBody processes the response body.
func (r *OpenAIRouter) handleResponseBody(v *ext_proc.ProcessingRequest_ResponseBody, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	if skipResponse := r.handleSkipProcessingResponseBody(v.ResponseBody.Body, ctx); skipResponse != nil {
		return skipResponse, nil
	}

	completionLatency := time.Since(ctx.StartTime)

	// Decrement active request count for queue depth estimation.
	defer metrics.DecrementModelActiveRequests(ctx.RequestModel)

	if looperResponse := r.handleLooperResponseBody(v.ResponseBody.Body, ctx); looperResponse != nil {
		return looperResponse, nil
	}

	// Anthropic clients on the streaming path always go through the
	// outbound emitter so the client sees Anthropic-shape SSE,
	// regardless of upstream APIFormat. The dispatcher branch order
	// matters: this check precedes the older
	// handleAnthropicStreamingResponseBody branch because the
	// double-Anthropic cell (Anthropic client + Anthropic backend)
	// would otherwise match the legacy branch and the client would
	// receive OpenAI SSE.
	if ctx.IsStreamingResponse && ctx.ClientProtocol == config.ClientProtocolAnthropic {
		return r.handleAnthropicClientStreamingResponseBody(v.ResponseBody.Body, ctx), nil
	}

	// Legacy branch: OpenAI client hitting an Anthropic backend. The
	// extra ClientProtocol guard prevents the new Anthropic-client
	// branch from being shadowed by this one.
	if ctx.IsStreamingResponse && ctx.APIFormat == config.APIFormatAnthropic &&
		ctx.ClientProtocol != config.ClientProtocolAnthropic {
		return r.handleAnthropicStreamingResponseBody(v.ResponseBody.Body, ctx), nil
	}

	responseBody, anthropicTransformed, err := r.normalizeProviderResponseBody(v.ResponseBody.Body, ctx)
	if err != nil {
		return r.createErrorResponse(502, fmt.Sprintf("Response transformation error: %v", err)), nil
	}

	if ctx.IsStreamingResponse {
		return r.handleStreamingResponseBody(responseBody, ctx), nil
	}

	return r.handleNonStreamingResponseBody(responseBody, ctx, completionLatency, anthropicTransformed), nil
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

func (r *OpenAIRouter) normalizeProviderResponseBody(
	responseBody []byte,
	ctx *RequestContext,
) ([]byte, bool, error) {
	if ctx.APIFormat != config.APIFormatAnthropic {
		return responseBody, false, nil
	}

	// Pass IRExtensions through so the Anthropic-only stop reason, cache
	// usage counters, server-tool counts, and thinking-block signatures
	// land on the per-request sidecar. The Anthropic outbound emitter
	// (translateResponseBodyForClient → EmitAnthropicResponse) replays
	// them on the response body so an Anthropic client sees the same
	// fields the upstream actually produced.
	transformedBody, err := anthropic.ToOpenAIResponseBodyWithExt(responseBody, ctx.RequestModel, ctx.IRExtensions)
	if err != nil {
		logging.Errorf("Failed to transform Anthropic response to OpenAI format: %v", err)
		return nil, false, err
	}

	logging.Infof(
		"Transformed Anthropic response to OpenAI format, original size: %d, transformed size: %d",
		len(responseBody),
		len(transformedBody),
	)
	return transformedBody, true, nil
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
