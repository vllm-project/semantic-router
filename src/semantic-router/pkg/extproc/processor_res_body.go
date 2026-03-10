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
	completionLatency := time.Since(ctx.StartTime)

	// Decrement active request count for queue depth estimation.
	defer metrics.DecrementModelActiveRequests(ctx.RequestModel)

	if looperResponse := r.handleLooperResponseBody(v.ResponseBody.Body, ctx); looperResponse != nil {
		return looperResponse, nil
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

	transformedBody, err := anthropic.ToOpenAIResponseBody(responseBody, ctx.RequestModel)
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
