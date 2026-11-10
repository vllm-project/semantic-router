package extproc

import (
	"bytes"
	"errors"
	"slices"
	"strconv"
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
	diagnosticsBeforeBody := len(ctx.ProtocolDiagnostics)
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
	decodedDiagnosticsEnd := len(ctx.ProtocolDiagnostics)
	clientBody := responseBody
	// Same-format Chat usually forwards upstream bytes. When codec translation
	// changed those bytes (for example, by dropping provider decorations), send
	// the translated response instead.
	rewriteClientBody := requiresClientResponseRewrite(ctx) ||
		ctx.SourceFormat == llmprotocol.OpenAIChatV1 && ctx.ResponseBodyNeedsRewrite
	if rewriteClientBody {
		clientBody, err = r.encodeClientResponse(*semanticResponse, ctx)
		if err != nil {
			return r.createErrorResponse(502, "The selected model returned an incompatible response")
		}
		ctx.ProtocolDiagnostics = deduplicateReencodedResponseDiagnostics(
			ctx.ProtocolDiagnostics, diagnosticsBeforeBody, decodedDiagnosticsEnd,
		)
	}
	usage := r.takeNeutralResponseUsage(ctx)
	observeUpstreamResponse(ctx, semanticResponse.Model, usage)
	r.reportNonStreamingUsage(ctx, completionLatency, usage)
	r.calibrateTokenEstimator(ctx, usage.promptTokens)

	r.updateKVAddressRegistry(ctx)
	cacheable := r.cacheableClientResponse(clientBody, rewriteClientBody, *semanticResponse, ctx)

	blocked, finalBody, headerOptions := r.finalizeResponsePolicy(ctx, semanticResponse, clientBody)
	warnings, hasBufferedWarnings := bufferedWarnings(ctx)
	if blocked != nil {
		return blocked
	}
	// The cache keeps the answer the client receives: the guard passed it, and
	// a rule with details wrote its note into it.
	if !bytes.Equal(finalBody, clientBody) {
		cacheable = finalBody
	}
	r.updateResponseCache(ctx, cacheable)

	response := buildResponseBodyContinueResponse(nil, nil)
	if len(headerOptions) > 0 {
		response.GetResponseBody().GetResponse().HeaderMutation = &ext_proc.HeaderMutation{
			SetHeaders: headerOptions,
		}
	}
	if hasBufferedWarnings {
		setResponseBodyHeaderOverwrite(response, headers.VSRWarnings, warnings)
	}
	if (rewriteClientBody || !bytes.Equal(finalBody, clientBody)) && response.GetResponseBody().GetResponse().GetBodyMutation() == nil {
		setResponseBodyMutation(response, finalBody)
	}
	return response
}

// Translation validates and renders the client wire before response policy
// runs. The buffered path renders it again to apply response mutations; a
// warning emitted by both renders describes one loss, not two.
func deduplicateReencodedResponseDiagnostics(
	diagnostics llmprotocol.Diagnostics,
	existingStart, reencodeStart int,
) llmprotocol.Diagnostics {
	existing := diagnostics[existingStart:reencodeStart]
	encoded := diagnostics[reencodeStart:]
	kept := diagnostics[:reencodeStart]
	for _, diagnostic := range encoded {
		if !slices.Contains(existing, diagnostic) {
			kept = append(kept, diagnostic)
		}
	}
	return kept
}

// finalizeResponsePolicy runs the shared response-stage processing across normal and fallback paths:
// the selected decision's response_guard, which blocks or annotates the answer, memory suppression,
// warnings and cost headers, persisting Responses objects, and updating replay audit records.
func (r *OpenAIRouter) finalizeResponsePolicy(
	ctx *RequestContext,
	semanticResponse *llmprotocol.Response,
	clientBody []byte,
) (*ext_proc.ProcessingResponse, []byte, []*core.HeaderValueOption) {
	plan := r.prepareResponsePolicy(ctx, semanticResponse, clientBody)
	plan.commit()
	return plan.blocked, plan.finalBody, plan.headerOptions()
}

// responsePolicyPlan separates response-policy evaluation and body mutation
// from irreversible persistence. Fallback can therefore validate the final
// public wire before committing memory, Responses, and replay side effects.
type responsePolicyPlan struct {
	blocked *ext_proc.ProcessingResponse
	// refusal is the error a streaming client receives for blocked.
	refusal *llmprotocol.ProtocolError
	// finalBody is the answer with the guard's notes, which Router Replay
	// keeps at commit: a caller that sends the client another form of it
	// replaces it first.
	finalBody  []byte
	response   *ext_proc.ProcessingResponse
	commitFunc func()
}

func (p *responsePolicyPlan) commit() {
	if p == nil || p.commitFunc == nil {
		return
	}
	commit := p.commitFunc
	p.commitFunc = nil
	commit()
}

func (p *responsePolicyPlan) headerOptions() []*core.HeaderValueOption {
	if p == nil || p.response == nil {
		return nil
	}
	bodyResp := p.response.GetResponseBody()
	if bodyResp == nil || bodyResp.GetResponse() == nil || bodyResp.GetResponse().GetHeaderMutation() == nil {
		return nil
	}
	return bodyResp.GetResponse().GetHeaderMutation().GetSetHeaders()
}

func (r *OpenAIRouter) prepareResponsePolicy(
	ctx *RequestContext,
	semanticResponse *llmprotocol.Response,
	clientBody []byte,
) *responsePolicyPlan {
	if r == nil || ctx == nil || semanticResponse == nil {
		return &responsePolicyPlan{finalBody: clientBody}
	}

	// The guard reads the answer before anything rewrites it, and what it read
	// is recorded before a block returns, so a blocked response leaves the same
	// evidence in Router Replay as a delivered one.
	recordPrimaryOutputDigest(ctx, semanticResponse)
	ctx.ResponseGuard = r.runResponseGuard(ctx, semanticAssistantContent(semanticResponse))
	logResponseGuard(ctx, ctx.ResponseGuard)
	if refused := r.delegatedCredentialRefusal(ctx); refused != nil {
		return &responsePolicyPlan{blocked: refused, refusal: ctx.ImmediateProtocolError}
	}
	if rule := ctx.ResponseGuard.blocking(); rule != nil {
		return &responsePolicyPlan{
			blocked: r.createErrorResponse(403, responseGuardBlockMessage(rule)),
			refusal: responseGuardRefusal(rule),
			commitFunc: func() {
				r.recordRouterReplayResponseGuard(ctx)
				r.recordUnscheduledResponseMemoryStore(ctx, "policy_blocked", "response_blocked", false)
			},
		}
	}

	memoryResponse := semanticResponse
	if _, _, suppressed := r.suppressedResponseMemoryStore(ctx); !suppressed {
		if cloned, err := cloneSemanticResponseForCommit(semanticResponse); err == nil {
			memoryResponse = cloned
		} else {
			logging.ComponentErrorEvent("extproc", "response_policy_commit_snapshot_failed", map[string]interface{}{
				"request_id": ctx.RequestID,
				"error":      err.Error(),
			})
		}
	}

	response, finalBody := r.applySemanticResponseWarnings(ctx, semanticResponse, clientBody)
	plan := &responsePolicyPlan{finalBody: finalBody, response: response}
	plan.commitFunc = func() {
		r.recordRouterReplayResponseGuard(ctx)
		r.scheduleSemanticResponseMemoryStore(ctx, memoryResponse)
		addResponseCostHeaders(ctx, response)
		r.persistResponseObject(ctx)
		r.attachRouterReplayResponse(ctx, plan.finalBody, true)
	}
	return plan
}

func (r *OpenAIRouter) applySemanticResponseWarnings(
	ctx *RequestContext,
	semanticResponse *llmprotocol.Response,
	originalBody []byte,
) (*ext_proc.ProcessingResponse, []byte) {
	response := buildResponseBodyContinueResponse(nil, nil)
	changed := r.applyResponseGuardNotes(ctx.ResponseGuard, semanticResponse)
	addResponseStageTrace(ctx, response)
	if !changed {
		return response, originalBody
	}
	diagnosticsBeforeEncode := len(ctx.ProtocolDiagnostics)
	encoded, err := r.encodeClientResponse(*semanticResponse, ctx)
	if err != nil {
		ctx.ProtocolDiagnostics = ctx.ProtocolDiagnostics[:diagnosticsBeforeEncode]
		logging.ComponentErrorEvent("extproc", "neutral_response_warning_encode_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"format":     ctx.SourceFormat,
			"error":      err.Error(),
		})
		return response, originalBody
	}
	ctx.ProtocolDiagnostics = deduplicateReencodedResponseDiagnostics(
		ctx.ProtocolDiagnostics, 0, diagnosticsBeforeEncode,
	)
	setResponseBodyMutation(response, encoded)
	return response, encoded
}

// addResponseStageTrace rewrites x-vsr-trace once the guard has read the
// answer, so it carries the guard's outcome and its signals the
// response-stage matches: the response headers phase wrote the trace before
// the body existed. Only for a caller allowed the debug headers.
func addResponseStageTrace(ctx *RequestContext, response *ext_proc.ProcessingResponse) {
	if ctx.ResponseGuard == nil {
		return
	}
	if trace := responseTrace(ctx); trace != "" {
		setResponseBodyHeaderOverwrite(response, headers.VSRTrace, trace)
	}
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
	setResponseBodyHeaderWithAction(response, key, value, core.HeaderValueOption_APPEND_IF_EXISTS_OR_ADD)
}

func setResponseBodyHeaderOverwrite(response *ext_proc.ProcessingResponse, key, value string) {
	setResponseBodyHeaderWithAction(response, key, value, core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD)
}

func setResponseBodyHeaderWithAction(response *ext_proc.ProcessingResponse, key, value string, action core.HeaderValueOption_HeaderAppendAction) {
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
		AppendAction: action,
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
// response. A cache partition is keyed on the ingress protocol and may be
// served to a request that never touches the same backend, so an entry holds
// only the canonical contract.
//
// A same-format response is forwarded to the client verbatim, so when the
// upstream added members its wire contract does not declare, such as a
// gateway's own cost or trace fields, those bytes still carry them. Persisting
// them would replay one request's gateway metadata to later clients.
// Re-encoding from the neutral response yields the canonical equivalent. When
// that encode fails, nothing is cached: a miss is recoverable, a wrong entry
// is not.
func (r *OpenAIRouter) cacheableClientResponse(
	clientBody []byte,
	rewritten bool,
	response llmprotocol.Response,
	ctx *RequestContext,
) []byte {
	if rewritten || ctx == nil || !ctx.ResponseUnknownFields {
		return clientBody
	}
	canonical, err := r.encodeCanonicalClientResponse(response, ctx)
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
