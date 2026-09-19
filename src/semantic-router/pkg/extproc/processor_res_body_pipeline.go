package extproc

import (
	"bytes"
	"errors"
	"strconv"
	"strings"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func (r *OpenAIRouter) handleNonStreamingResponseBody(
	responseBody []byte,
	ctx *RequestContext,
	completionLatency time.Duration,
) *ext_proc.ProcessingResponse {
	semanticResponse, err := r.decodeClientResponse(responseBody, ctx)
	if err != nil {
		metrics.RecordRequestError(ctx.RequestModel, "parse_error")
		decodeEvent := map[string]interface{}{
			"request_id":     ctx.RequestID,
			"backend_format": ctx.TargetFormat,
			"client_format":  ctx.SourceFormat,
			"error":          err.Error(),
		}
		// Log the private cause while keeping the client-facing message generic.
		if cause := errors.Unwrap(err); cause != nil {
			decodeEvent["cause"] = cause.Error()
		}
		logging.ComponentErrorEvent("extproc", "neutral_response_decode_failed", decodeEvent)
		return r.createErrorResponse(502, "The selected model returned an invalid response")
	}
	clientBody := responseBody
	rewriteClientBody := requiresClientResponseRewrite(ctx)
	if rewriteClientBody {
		clientBody, err = r.encodeClientResponse(*semanticResponse, ctx)
		if err != nil {
			return r.createErrorResponse(502, "The selected model returned an incompatible response")
		}
	}
	usage := r.takeNeutralResponseUsage(ctx)
	r.reportNonStreamingUsage(ctx, completionLatency, usage)
	r.calibrateTokenEstimator(ctx, usage.promptTokens)

	r.updateResponseCache(ctx, r.cacheableClientResponse(clientBody, rewriteClientBody, *semanticResponse, ctx))

	blocked, finalBody, headerOptions := r.finalizeResponsePolicy(ctx, semanticResponse, clientBody)
	if blocked != nil {
		return blocked
	}

	response := buildResponseBodyContinueResponse(nil, nil)
	if len(headerOptions) > 0 {
		response.GetResponseBody().GetResponse().HeaderMutation = &ext_proc.HeaderMutation{
			SetHeaders: headerOptions,
		}
	}
	if (rewriteClientBody || !bytes.Equal(finalBody, clientBody)) && response.GetResponseBody().GetResponse().GetBodyMutation() == nil {
		setResponseBodyMutation(response, finalBody)
	}
	return response
}

// finalizeResponsePolicy runs the shared response-stage processing across normal and fallback paths:
// scoring response signals, evaluating guardrail plugins (jailbreak, hallucination) for blocking or warning,
// executing memory suppression decisions, applying warnings and cost headers,
// persisting Responses objects, and updating replay audit records.
func (r *OpenAIRouter) finalizeResponsePolicy(
	ctx *RequestContext,
	semanticResponse *llmprotocol.Response,
	clientBody []byte,
) (*ext_proc.ProcessingResponse, []byte, []*core.HeaderValueOption) {
	if r == nil || ctx == nil || semanticResponse == nil {
		return nil, clientBody, nil
	}

	// The response-stage signal is scored from the declared rules before any
	// plugin runs, so the observation exists whether or not the selected
	// decision carries a plugin; the plugins below then consume it. Recorded
	// before a block returns, so a blocked response leaves the same evidence in
	// Router Replay as a delivered one.
	r.observeResponseStageSignals(ctx, semanticAssistantContent(semanticResponse))

	if jailbreakResponse := r.performSemanticResponseJailbreakDetection(ctx, semanticResponse); jailbreakResponse != nil {
		r.scheduleSemanticResponseMemoryStore(ctx, semanticResponse)
		return jailbreakResponse, nil, nil
	}
	if hallucinationResponse := r.performSemanticHallucinationDetection(ctx, semanticResponse); hallucinationResponse != nil {
		r.recordUnscheduledResponseMemoryStore(ctx, "policy_blocked", "hallucination_blocked", false)
		return hallucinationResponse, nil, nil
	}

	r.scheduleSemanticResponseMemoryStore(ctx, semanticResponse)
	r.markUnverifiedFactualResponse(ctx)

	response, finalBody := r.applySemanticResponseWarnings(ctx, semanticResponse, clientBody)
	addResponseCostHeaders(ctx, response)
	r.persistResponseObject(ctx)
	r.updateRouterReplayHallucinationStatus(ctx)
	r.attachRouterReplayResponse(ctx, finalBody, true)

	var headerOptions []*core.HeaderValueOption
	if bodyResp := response.GetResponseBody(); bodyResp != nil && bodyResp.GetResponse() != nil {
		if hm := bodyResp.GetResponse().GetHeaderMutation(); hm != nil {
			headerOptions = hm.GetSetHeaders()
		}
	}

	return nil, finalBody, headerOptions
}

// observeResponseStageSignals scores the response-stage rules against the
// answer and records the observation in Router Replay. Both response paths
// share it, so a streamed response leaves the evidence a buffered one leaves.
//
// Only the buffered path goes on to enforce. A streamed answer exists as a
// whole for the first time when its bytes are already with the client, so no
// plugin can block or rewrite it and none runs; the observation is all that is
// still possible, and the record says so.
func (r *OpenAIRouter) observeResponseStageSignals(ctx *RequestContext, assistantContent string) {
	r.evaluateResponseJailbreakSignal(ctx, assistantContent)
	r.evaluateHallucinationSignal(ctx, assistantContent)
	r.recordRouterReplayResponseJailbreak(ctx)
	r.recordRouterReplayHallucination(ctx)
}

func (r *OpenAIRouter) applySemanticResponseWarnings(
	ctx *RequestContext,
	semanticResponse *llmprotocol.Response,
	originalBody []byte,
) (*ext_proc.ProcessingResponse, []byte) {
	response := buildResponseBodyContinueResponse(nil, nil)
	changed := false
	var codes []string
	var bodyChanged bool

	bodyChanged, code := r.applySemanticHallucinationWarning(ctx, semanticResponse)
	changed = changed || bodyChanged
	codes = appendNonEmpty(codes, code)
	bodyChanged, code = r.applySemanticUnverifiedFactualWarning(ctx, semanticResponse)
	changed = changed || bodyChanged
	codes = appendNonEmpty(codes, code)
	codes = appendNonEmpty(codes, r.responseJailbreakWarningCode(ctx))

	if len(codes) > 0 {
		setResponseWarningsHeader(response, codes)
	}
	addResponseStageSignalHeaders(ctx, response)
	if !changed {
		return response, originalBody
	}
	encoded, err := r.encodeClientResponse(*semanticResponse, ctx)
	if err != nil {
		logging.ComponentErrorEvent("extproc", "neutral_response_warning_encode_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"format":     ctx.SourceFormat,
			"error":      err.Error(),
		})
		return response, originalBody
	}
	setResponseBodyMutation(response, encoded)
	return response, encoded
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

func appendNonEmpty(codes []string, code string) []string {
	if code == "" {
		return codes
	}
	return append(codes, code)
}

// addResponseStageSignalHeaders writes the response-stage matches in the body
// phase: x-vsr-matched-jailbreak is rewritten once the response-direction
// rules have been scored, and x-vsr-matched-hallucination is written once the
// answer has been checked. The response headers phase wrote the request-stage
// matches before the body existed, so the debug headers would otherwise never
// show a response-stage match. Same gate as the request-stage signal headers:
// only when debug is requested.
func addResponseStageSignalHeaders(ctx *RequestContext, response *ext_proc.ProcessingResponse) {
	if ctx == nil || !debugHeadersRequested(ctx) {
		return
	}
	if len(ctx.VSRMatchedResponseJailbreak) > 0 {
		matched := make([]string, 0, len(ctx.VSRMatchedJailbreak)+len(ctx.VSRMatchedResponseJailbreak))
		matched = append(matched, ctx.VSRMatchedJailbreak...)
		matched = append(matched, ctx.VSRMatchedResponseJailbreak...)
		setResponseBodyHeader(response, headers.VSRMatchedJailbreak, strings.Join(matched, ","))
	}
	if len(ctx.VSRMatchedHallucination) > 0 {
		setResponseBodyHeader(response, headers.VSRMatchedHallucination, strings.Join(ctx.VSRMatchedHallucination, ","))
	}
}

// setResponseWarningsHeader writes the consolidated x-vsr-response-warnings header
// (comma-separated codes) onto the response, merging with any existing mutation.
func setResponseWarningsHeader(response *ext_proc.ProcessingResponse, codes []string) {
	setResponseBodyHeader(response, headers.VSRResponseWarnings, strings.Join(codes, ","))
}

// addResponseCostHeaders reports the priced cost of a buffered response. A
// streamed response has already sent its headers by the time usage arrives.
func addResponseCostHeaders(ctx *RequestContext, response *ext_proc.ProcessingResponse) {
	if ctx == nil || !ctx.RequestCostPriced {
		return
	}
	setResponseBodyHeader(response, headers.VSRCost, strconv.FormatFloat(ctx.RequestCost, 'f', -1, 64))
	if ctx.RequestCostCurrency != "" {
		setResponseBodyHeader(response, headers.VSRCostCurrency, ctx.RequestCostCurrency)
	}
}

// setResponseBodyHeader sets one response header from the body phase, merging
// with any header mutation the response already carries.
func setResponseBodyHeader(response *ext_proc.ProcessingResponse, key, value string) {
	bodyResponse, ok := response.Response.(*ext_proc.ProcessingResponse_ResponseBody)
	if !ok {
		return
	}
	if bodyResponse.ResponseBody.Response == nil {
		bodyResponse.ResponseBody.Response = &ext_proc.CommonResponse{}
	}
	opt := &core.HeaderValueOption{
		Header: &core.HeaderValue{
			Key:      key,
			RawValue: []byte(value),
		},
	}
	if hm := bodyResponse.ResponseBody.Response.HeaderMutation; hm != nil {
		hm.SetHeaders = append(hm.SetHeaders, opt)
		return
	}
	bodyResponse.ResponseBody.Response.HeaderMutation = &ext_proc.HeaderMutation{
		SetHeaders: []*core.HeaderValueOption{opt},
	}
}

func setResponseBodyMutation(response *ext_proc.ProcessingResponse, body []byte) {
	bodyResponse, ok := response.Response.(*ext_proc.ProcessingResponse_ResponseBody)
	if !ok {
		return
	}
	if bodyResponse.ResponseBody.Response == nil {
		bodyResponse.ResponseBody.Response = &ext_proc.CommonResponse{}
	}
	bodyResponse.ResponseBody.Response.BodyMutation = &ext_proc.BodyMutation{
		Mutation: &ext_proc.BodyMutation_Body{
			Body: body,
		},
	}
	if bodyResponse.ResponseBody.Response.HeaderMutation == nil {
		bodyResponse.ResponseBody.Response.HeaderMutation = &ext_proc.HeaderMutation{}
	}
	// A body rewrite invalidates the upstream byte count. Let Envoy derive the
	// correct framing instead of forwarding a stale content-length.
	ensureHeaderRemoved(bodyResponse.ResponseBody.Response.HeaderMutation, "content-length")
}

func setResponseContentType(response *ext_proc.ProcessingResponse, contentType string) {
	bodyResponse, ok := response.Response.(*ext_proc.ProcessingResponse_ResponseBody)
	if !ok {
		return
	}
	if bodyResponse.ResponseBody.Response == nil {
		bodyResponse.ResponseBody.Response = &ext_proc.CommonResponse{}
	}
	if bodyResponse.ResponseBody.Response.HeaderMutation == nil {
		bodyResponse.ResponseBody.Response.HeaderMutation = &ext_proc.HeaderMutation{}
	}
	mutation := bodyResponse.ResponseBody.Response.HeaderMutation
	for _, option := range mutation.SetHeaders {
		if option.GetHeader().GetKey() != "content-type" {
			continue
		}
		option.Header.Value = ""
		option.Header.RawValue = []byte(contentType)
		option.AppendAction = core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD
		return
	}
	mutation.SetHeaders = append(mutation.SetHeaders, &core.HeaderValueOption{
		Header: &core.HeaderValue{
			Key:      "content-type",
			RawValue: []byte(contentType),
		},
		AppendAction: core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD,
	})
}

func isResponseAPIRequest(ctx *RequestContext) bool {
	return ctx != nil && ctx.SourceFormat == llmprotocol.OpenAIResponsesV1
}

// cacheableClientResponse returns the bytes that may be persisted as the public
// response. The cache is read back under the strict canonical contract by
// decodeCachedClientResponse, deliberately without a backend vendor allowance,
// because a cache partition is keyed on the ingress protocol and may be served
// to a request that never touches the same backend.
//
// A same-format response is forwarded to the client verbatim, so on a backend
// with a vendor allowance those bytes still carry the provider's decorations.
// Persisting them would store an entry the strict reader rejects: the first
// Azure response would poison its own partition and every later hit would fail.
// Re-encoding from the neutral response yields the canonical equivalent. When
// that encode fails, nothing is cached - a miss is recoverable, a poisoned
// entry is not.
func (r *OpenAIRouter) cacheableClientResponse(
	clientBody []byte,
	rewritten bool,
	response llmprotocol.Response,
	ctx *RequestContext,
) []byte {
	if rewritten || ctx == nil || !ctx.ResponseVendorExtensions {
		return clientBody
	}
	canonical, err := r.encodeClientResponse(response, ctx)
	if err != nil {
		logging.ComponentWarnEvent("extproc", "cache_write_skipped_noncanonical_response", map[string]interface{}{
			"request_id": ctx.RequestID,
			"format":     ctx.SourceFormat,
			"vendor":     string(ctx.ResponseVendor),
			"error":      err.Error(),
		})
		return nil
	}
	return canonical
}
