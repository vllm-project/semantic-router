package extproc

import (
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/anthropic"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (r *OpenAIRouter) handleNonStreamingResponseBody(
	responseBody []byte,
	ctx *RequestContext,
	completionLatency time.Duration,
	initialBodyTransformed bool,
) *ext_proc.ProcessingResponse {
	usage := parseResponseUsage(responseBody, ctx.RequestModel)
	r.reportNonStreamingUsage(ctx, completionLatency, usage)
	r.calibrateTokenEstimator(ctx, usage.promptTokens)
	r.updateResponseCache(ctx, responseBody)

	finalBody, clientTransformed := r.translateResponseBodyForClient(ctx, responseBody)
	bodyMutation, headerMutation := buildInitialResponseMutations(
		finalBody,
		initialBodyTransformed || clientTransformed,
	)

	if jailbreakResponse := r.performResponseJailbreakDetection(ctx, responseBody); jailbreakResponse != nil {
		return jailbreakResponse
	}
	if hallucinationResponse := r.performHallucinationDetection(ctx, responseBody); hallucinationResponse != nil {
		return hallucinationResponse
	}

	r.scheduleResponseMemoryStore(ctx, responseBody)
	r.markUnverifiedFactualResponse(ctx)

	response := r.applyResponseWarnings(ctx, responseBody, bodyMutation, headerMutation)
	r.updateRouterReplayHallucinationStatus(ctx)
	r.attachRouterReplayResponse(ctx, finalBody, true)
	return response
}

// translateResponseBodyForClient dispatches body rewriting based on the
// inbound client protocol. Returns the rewritten body and a flag
// indicating whether a rewrite happened, so callers know whether to
// emit a body mutation downstream.
//
// Branch order matters: ClientProtocol "anthropic" and Response API are
// mutually exclusive (different inbound paths — /v1/messages vs
// /v1/responses), but pinning the Anthropic branch first keeps the
// dispatcher reading top-down by likelihood.
func (r *OpenAIRouter) translateResponseBodyForClient(
	ctx *RequestContext,
	responseBody []byte,
) ([]byte, bool) {
	if ctx != nil && ctx.ClientProtocol == config.ClientProtocolAnthropic {
		translated, err := anthropic.EmitAnthropicResponse(responseBody, ctx.IRExtensions, ctx.RequestModel)
		if err != nil {
			logging.Errorf("Anthropic outbound emit failed: %v", err)
			return anthropic.EmitAnthropicError("api_error", err.Error()), true
		}
		return translated, true
	}

	if !isResponseAPIRequest(ctx) || r.ResponseAPIFilter == nil {
		return responseBody, false
	}

	translatedBody, err := r.ResponseAPIFilter.TranslateResponse(
		ctx.TraceContext,
		ctx.ResponseAPICtx,
		responseBody,
	)
	if err != nil {
		logging.Errorf("Response API translation error: %v", err)
		return responseBody, false
	}

	logging.Infof("Response API: Translated response to Response API format")
	return translatedBody, true
}

func buildInitialResponseMutations(
	finalBody []byte,
	bodyWasTransformed bool,
) (*ext_proc.BodyMutation, *ext_proc.HeaderMutation) {
	if !bodyWasTransformed {
		return nil, nil
	}

	return &ext_proc.BodyMutation{
			Mutation: &ext_proc.BodyMutation_Body{
				Body: finalBody,
			},
		}, &ext_proc.HeaderMutation{
			RemoveHeaders: []string{"content-length"},
		}
}

func (r *OpenAIRouter) markUnverifiedFactualResponse(ctx *RequestContext) {
	if ctx.VSRSelectedDecision == nil {
		return
	}

	hallucinationConfig := ctx.VSRSelectedDecision.GetHallucinationConfig()
	if hallucinationConfig != nil && hallucinationConfig.Enabled {
		r.checkUnverifiedFactualResponse(ctx)
	}
}

func (r *OpenAIRouter) applyResponseWarnings(
	ctx *RequestContext,
	responseBody []byte,
	bodyMutation *ext_proc.BodyMutation,
	headerMutation *ext_proc.HeaderMutation,
) *ext_proc.ProcessingResponse {
	response := buildResponseBodyContinueResponse(bodyMutation, headerMutation)
	modifiedBody := responseBody
	needsBodyMutation := false

	if ctx.ResponseJailbreakDetected {
		modifiedBody, response = r.applyResponseJailbreakWarning(response, ctx, modifiedBody)
	}
	if ctx.HallucinationDetected {
		modifiedBody, response = r.applyHallucinationWarning(response, ctx, modifiedBody)
		if string(modifiedBody) != string(responseBody) {
			needsBodyMutation = true
		}
	}
	if ctx.UnverifiedFactualResponse {
		modifiedBody, response = r.applyUnverifiedFactualWarning(response, ctx, modifiedBody)
		if string(modifiedBody) != string(responseBody) {
			needsBodyMutation = true
		}
	}
	if needsBodyMutation {
		bodyResponse := response.Response.(*ext_proc.ProcessingResponse_ResponseBody)
		bodyResponse.ResponseBody.Response.BodyMutation = &ext_proc.BodyMutation{
			Mutation: &ext_proc.BodyMutation_Body{
				Body: modifiedBody,
			},
		}
	}
	return response
}

func isResponseAPIRequest(ctx *RequestContext) bool {
	return ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.IsResponseAPIRequest
}
