package extproc

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dispatchauthority"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingcontext"
)

// DispatchCapabilityRuntime is the only backend-dispatch authority borrowed by
// immutable Router generations. Implementations are process-owned.
type DispatchCapabilityRuntime interface {
	dispatchauthority.CapabilityRuntime
	dispatchauthority.FallbackCapabilityRuntime
	dispatchauthority.OutcomeRuntime
}

type verifiedDispatchGrantContextKey struct{}

func withVerifiedDispatchGrant(
	ctx context.Context,
	grant dispatchauthority.VerifiedGrant,
) context.Context {
	return context.WithValue(ctx, verifiedDispatchGrantContextKey{}, grant)
}

func verifiedDispatchGrantFromContext(ctx context.Context) (dispatchauthority.VerifiedGrant, bool) {
	if ctx == nil {
		return dispatchauthority.VerifiedGrant{}, false
	}
	grant, ok := ctx.Value(verifiedDispatchGrantContextKey{}).(dispatchauthority.VerifiedGrant)
	return grant, ok
}

func consumeDispatchGrant(
	headerMap *core.HeaderMap,
	runtime DispatchCapabilityRuntime,
	ctx context.Context,
	generation routingcontext.Generation,
	requestID string,
) (dispatchauthority.VerifiedGrant, error) {
	if headerMap == nil || runtime == nil {
		return dispatchauthority.VerifiedGrant{}, fmt.Errorf("dispatch grant verifier is unavailable")
	}
	value := ""
	filtered := headerMap.Headers[:0]
	for _, header := range headerMap.Headers {
		if header == nil || !strings.EqualFold(strings.TrimSpace(header.Key), headers.VSRDispatchGrant) {
			filtered = append(filtered, header)
			continue
		}
		if value != "" {
			value = "invalid"
		} else {
			value = strings.TrimSpace(extractHeaderValue(header))
		}
		header.Value = ""
		header.RawValue = nil
	}
	headerMap.Headers = filtered
	if value == "" || value == "invalid" {
		return dispatchauthority.VerifiedGrant{}, fmt.Errorf("dispatch grant is missing or duplicated")
	}
	return runtime.VerifyGrant(ctx, value, dispatchauthority.GrantVerificationRequest{
		Generation: generation, RequestID: requestID,
	})
}

func (r *OpenAIRouter) issueBackendDispatchCapability(
	ctx *RequestContext,
	model string,
	body []byte,
) (string, error) {
	if r == nil || r.DispatchCapabilities == nil || ctx == nil {
		return "", fmt.Errorf("backend dispatch capability runtime is unavailable")
	}
	params, found := r.Config.ModelConfig[model]
	if !found || params.ResourceID == "" || params.ResourceRevision <= 0 {
		return "", fmt.Errorf("model %q lacks immutable runtime identity", model)
	}
	format := ctx.SourceFormat
	if format == "" {
		format = llmprotocol.OpenAIChatV1
	}
	path := requestWirePath(format)
	query := ""
	final := dispatchauthority.FinalRequest{
		Model:  dispatchauthority.ModelIdentity{ID: params.ResourceID, Revision: params.ResourceRevision},
		Method: http.MethodPost, Path: path, Query: query, WireFormat: format, Body: body,
	}
	if grant, ok := verifiedDispatchGrantFromContext(ctx.TraceContext); ok {
		return r.DispatchCapabilities.IssueFromGrant(ctx.TraceContext, grant, final)
	}
	generation, ok := routingcontext.GenerationFrom(ctx.TraceContext)
	if !ok {
		return "", fmt.Errorf("managed routing generation is unavailable")
	}
	candidates, fallback, admission, err := dispatchChainAuthorization(ctx, model)
	if err != nil {
		return "", err
	}
	if admission != nil {
		return r.DispatchCapabilities.IssueMeteredChain(dispatchauthority.MeteredChainIssueRequest{
			Admission: *admission, Candidates: candidates, Fallback: fallback,
			RequestID: ctx.RequestID, Final: dispatchauthority.ChainFinalRequest{
				Method: final.Method, Path: final.Path, Query: final.Query, WireFormat: final.WireFormat, Body: final.Body,
			},
		})
	}
	return r.DispatchCapabilities.IssueRoutingOnlyChain(ctx.TraceContext, dispatchauthority.RoutingOnlyChainIssueRequest{
		Generation: generation, Candidates: candidates, Fallback: fallback,
		RequestID: ctx.RequestID, Final: dispatchauthority.ChainFinalRequest{
			Method: final.Method, Path: final.Path, Query: final.Query, WireFormat: final.WireFormat, Body: final.Body,
		},
	})
}

func dispatchChainAuthorization(
	ctx *RequestContext,
	model string,
) ([]dispatchauthority.CandidateIssue, backendinvoker.FallbackPolicy, *accessruntime.Admission, error) {
	if ctx == nil || ctx.ManagedDispatch == nil {
		return nil, backendinvoker.FallbackPolicy{}, nil, fmt.Errorf("dispatch chain is unavailable")
	}
	dispatchState := ctx.ManagedDispatch
	dispatchState.mu.Lock()
	if dispatchState.primaryDispatchID == "" || dispatchState.primaryCandidateCount < 1 ||
		dispatchState.primaryCandidateCount > len(dispatchState.dispatches) {
		dispatchState.mu.Unlock()
		return nil, backendinvoker.FallbackPolicy{}, nil, fmt.Errorf("dispatch chain is unavailable")
	}
	candidates := make([]dispatchauthority.CandidateIssue, 0, dispatchState.primaryCandidateCount)
	for index, dispatch := range dispatchState.dispatches[:dispatchState.primaryCandidateCount] {
		if dispatch == nil || !dispatch.planned || dispatch.id == "" ||
			(index == 0 && (dispatch.id != dispatchState.primaryDispatchID || dispatch.model != model)) {
			dispatchState.mu.Unlock()
			return nil, backendinvoker.FallbackPolicy{}, nil, fmt.Errorf("dispatch chain identity mismatch")
		}
		candidates = append(candidates, dispatchauthority.CandidateIssue{
			Dispatch: accessruntime.DispatchFacts{
				DispatchID: dispatch.id, Ordinal: dispatch.ordinal,
				DispatchPlanDigest: dispatch.planDigest,
			},
			Model:    dispatchauthority.ModelIdentity{ID: dispatch.modelID, Revision: dispatch.modelRevision},
			Priority: dispatch.priority,
		})
	}
	fallback := backendinvoker.FallbackPolicy{On: append(
		[]backendinvoker.FallbackTrigger(nil), dispatchState.fallback.On...,
	)}
	dispatchState.mu.Unlock()
	if ctx.InferenceAccess == nil {
		return candidates, fallback, nil, nil
	}
	accessState := ctx.InferenceAccess
	accessState.mu.Lock()
	defer accessState.mu.Unlock()
	if accessState.admission == nil {
		return nil, backendinvoker.FallbackPolicy{}, nil, fmt.Errorf("inference admission is unavailable")
	}
	admission := *accessState.admission
	return candidates, fallback, &admission, nil
}

func (r *OpenAIRouter) buildBackendDispatchResponse(
	model string,
	body []byte,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	state := &routeHeaderState{
		setHeaders: []*core.HeaderValueOption{
			{Header: &core.HeaderValue{Key: headers.SelectedModel, RawValue: []byte(model)}},
			{Header: &core.HeaderValue{Key: ":path", RawValue: []byte(requestWirePath(ctx.SourceFormat))}},
		},
		removeHeaders: []string{
			"authorization", "proxy-authorization", "x-api-key", "api-key", "cookie",
			"content-length", backendinvoker.DispatchCapabilityHeader,
			headers.VSRDispatchGrant, headers.VSRInternalAuth,
		},
	}
	appendContentLengthHeader(&state.setHeaders, len(body))
	r.applyDecisionHeaderMutations(state, ctx)
	return buildRequestBodyContinueResponse(state, &ext_proc.BodyMutation{
		Mutation: &ext_proc.BodyMutation_Body{Body: body},
	}, true)
}

// finalizeBackendDispatchResponse binds the short-lived backend capability to
// the exact body Envoy will forward. Request plugins are intentionally allowed
// to finish first; minting against an intermediate body would either reject a
// legitimate request at the private dispatch listener or authorize bytes that
// are no longer present on the wire.
func (r *OpenAIRouter) finalizeBackendDispatchResponse(
	model string,
	response *ext_proc.ProcessingResponse,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	common := response.GetRequestBody().GetResponse()
	if common == nil || common.GetBodyMutation() == nil {
		return r.createInferenceAccessError(quotaruntime.AdmissionUnavailable, nil)
	}
	body, err := r.encodeDispatchRequest(ctx)
	if err != nil {
		return r.createInferenceAccessError(quotaruntime.AdmissionUnavailable, nil)
	}
	common.BodyMutation = &ext_proc.BodyMutation{
		Mutation: &ext_proc.BodyMutation_Body{Body: body},
	}
	if common.HeaderMutation == nil {
		common.HeaderMutation = &ext_proc.HeaderMutation{}
	}
	setHeaderValue(common.HeaderMutation, ":path", requestWirePath(ctx.SourceFormat))
	setHeaderValue(common.HeaderMutation, "content-length", fmt.Sprintf("%d", len(body)))
	capability, err := r.issueBackendDispatchCapability(ctx, model, body)
	if err != nil {
		return r.createInferenceAccessError(quotaruntime.AdmissionUnavailable, nil)
	}
	if ctx.ManagedDispatch != nil {
		ctx.ManagedDispatch.mu.Lock()
		ctx.ManagedDispatch.capabilityIssued = true
		ctx.ManagedDispatch.requestDigest = backendinvoker.RequestDigest(
			http.MethodPost, requestWirePath(ctx.SourceFormat), "", body,
		)
		ctx.ManagedDispatch.mu.Unlock()
	}
	setHeaderValue(common.HeaderMutation, backendinvoker.DispatchCapabilityHeader, capability)
	return response
}

func (r *OpenAIRouter) backendDispatchBody(
	request *llmprotocol.Request,
	model string,
	ctx *RequestContext,
) ([]byte, error) {
	if request == nil || ctx == nil {
		return nil, fmt.Errorf("neutral inference request is unavailable")
	}
	changed := request.Model != model || request.Stream != ctx.ExpectStreamingResponse
	request.Model = model
	request.Stream = ctx.ExpectStreamingResponse
	if changed {
		request.Generation++
	}
	return r.encodeDispatchRequest(ctx)
}
