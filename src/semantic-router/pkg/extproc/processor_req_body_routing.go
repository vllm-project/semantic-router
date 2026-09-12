package extproc

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

type routeHeaderState struct {
	setHeaders    []*core.HeaderValueOption
	removeHeaders []string
	profile       *config.ProviderProfile
}

type providerDispatch struct {
	logicalModel   string
	upstreamModel  string
	backendAddress string
	backendName    string
	profile        *config.ProviderProfile
	targetFormat   llmprotocol.WireFormat
	decisionName   string
	useReasoning   bool
}

// prepareProviderDispatch is the only point where a neutral request becomes a
// provider-bound request. Routing and plugins mutate semantic state first;
// the selected backend codec owns the final wire representation.
func (r *OpenAIRouter) prepareProviderDispatch(
	request *llmprotocol.Request,
	logicalModel string,
	decisionName string,
	useReasoning bool,
	ctx *RequestContext,
) (*providerDispatch, error) {
	if request == nil || ctx == nil || r == nil || r.Config == nil {
		return nil, status.Error(codes.Internal, "neutral inference request is unavailable")
	}
	dispatch, err := r.resolveProviderDispatch(logicalModel, decisionName, useReasoning)
	if err != nil {
		return nil, err
	}
	changed, err := r.prepareProviderRequest(request, dispatch, ctx)
	if err != nil {
		return nil, err
	}
	if changed {
		request.Generation++
	}
	required := llmprotocol.RequiredCapabilities(*request)
	if protocolErr := r.rejectDispatchCapabilityMismatch(request, dispatch.targetFormat, ctx); protocolErr != nil {
		rerouted, ok := r.rerouteToQualifiedDecisionModel(request, dispatch, required, ctx)
		if !ok {
			return nil, protocolErr
		}
		dispatch = rerouted
		ctx.ImmediateProtocolError = nil
		// The reasoning mode is model-scoped (family/effort come from the
		// model's reasoning config), so a reroute must re-apply it against the
		// rerouted model. System prompt and request params are decision-scoped
		// and were already applied to the shared request, so they are not
		// re-applied here.
		if dispatch.targetFormat != llmprotocol.OpenAIChatV1 {
			r.applySemanticReasoningMode(
				request, dispatch.logicalModel, dispatch.targetFormat, dispatch.useReasoning, ctx.VSRSelectedDecision,
			)
		}
	}
	ctx.TargetFormat = dispatch.targetFormat
	ctx.SemanticRequest = request
	// Per-model accounting (token tracking, TTFB, usage attribution) keys off
	// the model that actually serves the request. A capability reroute may
	// redirect this request to a sibling modelRef, so RequestModel must be the
	// final dispatch model, not the decision-selected one.
	ctx.RequestModel = dispatch.logicalModel
	logging.ComponentDebugEvent("extproc", "provider_dispatch_prepared", map[string]interface{}{
		"request_id":  ctx.RequestID,
		"model":       dispatch.logicalModel,
		"backend":     dispatch.backendName,
		"wire_format": dispatch.targetFormat,
	})
	return dispatch, nil
}

func (r *OpenAIRouter) codecCapabilitiesForFormat(format llmprotocol.WireFormat) (llmprotocol.CapabilitySet, bool) {
	if r == nil || r.ProtocolCodecs == nil {
		return protocolcodec.NewBuiltinRegistry().CapabilitiesFor(format)
	}
	return r.ProtocolCodecs.CapabilitiesFor(format)
}

// rejectDispatchCapabilityMismatch applies the same wire-fidelity gate the
// codec engine enforces at encode, but at dispatch time so a request whose
// required capabilities the chosen backend wire cannot express surfaces as a
// protocol error instead of a generic 500 from encoding. This is the first
// slice of capability-driven dispatch: it turns "crash with 500" into "clean
// 400 unsupported_capability" and gives future capability-aware re-routing a
// single check point.
func (r *OpenAIRouter) rejectDispatchCapabilityMismatch(
	request *llmprotocol.Request,
	format llmprotocol.WireFormat,
	ctx *RequestContext,
) error {
	available, ok := r.codecCapabilitiesForFormat(format)
	if !ok {
		// Unknown wire formats are rejected earlier by wireFormatForModel.
		return nil
	}
	if err := llmprotocol.RequireCapabilities(format, available, llmprotocol.RequiredCapabilities(*request)); err != nil {
		var protocolError *llmprotocol.ProtocolError
		if errors.As(err, &protocolError) && ctx != nil {
			ctx.ImmediateProtocolError = protocolError
		}
		return err
	}
	return nil
}

// declaredModelCapabilities parses a model's configured capability
// declarations. A model with no declaration (or an unparsable one) is treated
// as unannotated and stays eligible on wire expressibility alone; the declared
// filter only steers among annotated candidates.
func (r *OpenAIRouter) declaredModelCapabilities(model string) (llmprotocol.CapabilitySet, bool) {
	if r == nil || r.Config == nil {
		return llmprotocol.CapabilitySet{}, false
	}
	params, ok := r.Config.ModelConfig[model]
	if !ok || len(params.Capabilities) == 0 {
		return llmprotocol.CapabilitySet{}, false
	}
	declared, err := llmprotocol.ParseCapabilities(params.Capabilities)
	if err != nil {
		return llmprotocol.CapabilitySet{}, false
	}
	return declared, true
}

// rerouteToQualifiedDecisionModel tries to satisfy the required capabilities
// by dispatching to another modelRef offered by the selected decision, when
// the originally selected model's wire format cannot express them. It returns
// the re-resolved dispatch and whether a qualified candidate was found.
//
// Candidates are considered in modelRef order; the first whose wire format can
// express every required capability wins. This is capability-driven selection
// at the dispatch seam: routing prefers a qualified backend over a clean
// rejection, and only rejects when no candidate qualifies.
func (r *OpenAIRouter) rerouteToQualifiedDecisionModel(
	request *llmprotocol.Request,
	selected *providerDispatch,
	required llmprotocol.CapabilitySet,
	ctx *RequestContext,
) (*providerDispatch, bool) {
	if r == nil || r.Config == nil || request == nil || selected == nil || ctx == nil {
		return nil, false
	}
	decision := ctx.VSRSelectedDecision
	if decision == nil || decision.Name == "" {
		return nil, false
	}
	model := r.findQualifiedRerouteModel(decision, selected, required)
	if model == "" {
		return nil, false
	}
	candidate, err := r.resolveProviderDispatch(model, decision.Name, selected.useReasoning)
	if err != nil {
		return nil, false
	}
	request.Model = candidate.upstreamModel
	ctx.TargetFormat = candidate.targetFormat
	logging.ComponentDebugEvent("extproc", "provider_dispatch_rerouted", map[string]interface{}{
		"request_id":  ctx.RequestID,
		"from":        selected.logicalModel,
		"to":          model,
		"wire_format": candidate.targetFormat,
	})
	return candidate, true
}

// findQualifiedRerouteModel returns the first decision modelRef, in declared
// order, that can express every required capability and is not the currently
// selected model; "" when no sibling qualifies.
func (r *OpenAIRouter) findQualifiedRerouteModel(
	decision *config.Decision,
	selected *providerDispatch,
	required llmprotocol.CapabilitySet,
) string {
	for _, modelRef := range decision.ModelRefs {
		model := modelRef.Model
		if model == "" || model == selected.logicalModel {
			continue
		}
		if r.qualifiedRerouteCandidate(model, required) != "" {
			return model
		}
	}
	return ""
}

// qualifiedRerouteCandidate reports the model's wire format when the model can
// express every required capability, or "" when it cannot serve the request.
// Expressibility is judged on the wire format's codec capability set first,
// then narrowed by the model's own declared capabilities when annotated.
func (r *OpenAIRouter) qualifiedRerouteCandidate(model string, required llmprotocol.CapabilitySet) llmprotocol.WireFormat {
	format, err := wireFormatForModel(r.Config.GetModelAPIFormat(model))
	if err != nil {
		return ""
	}
	if set, ok := r.codecCapabilitiesForFormat(format); !ok || !set.Contains(required) {
		return ""
	}
	if declared, ok := r.declaredModelCapabilities(model); ok && !declared.Contains(required.TaskCapabilities()) {
		return ""
	}
	return format
}

func (r *OpenAIRouter) resolveProviderDispatch(
	logicalModel string,
	decisionName string,
	useReasoning bool,
) (*providerDispatch, error) {
	backendAddress, backendName, found, err := r.Config.ResolvePrimaryBackendForModel(logicalModel)
	if err != nil {
		return nil, fmt.Errorf("resolve backend for model %q: %w", logicalModel, err)
	}
	if !found {
		return nil, fmt.Errorf("model %q has no configured backend", logicalModel)
	}
	profile, err := r.Config.GetProviderProfileForEndpoint(backendName)
	if err != nil {
		return nil, fmt.Errorf("resolve provider profile for model %q: %w", logicalModel, err)
	}
	targetFormat, err := wireFormatForModel(r.Config.GetModelAPIFormat(logicalModel))
	if err != nil {
		return nil, fmt.Errorf("model %q: %w", logicalModel, err)
	}
	return &providerDispatch{
		logicalModel: logicalModel, upstreamModel: r.Config.ResolveExternalModelID(logicalModel, backendName),
		backendAddress: backendAddress, backendName: backendName,
		profile: profile, targetFormat: targetFormat,
		decisionName: decisionName, useReasoning: useReasoning,
	}, nil
}

func (r *OpenAIRouter) prepareProviderRequest(
	request *llmprotocol.Request,
	dispatch *providerDispatch,
	ctx *RequestContext,
) (bool, error) {
	changed, err := r.materializeResponseObjectContext(request, ctx)
	if err != nil {
		return false, err
	}
	inlined, err := r.resolveImageFileReferences(request)
	if err != nil {
		return false, err
	}
	changed = inlined || changed
	changed = request.Model != dispatch.upstreamModel || request.Stream != ctx.ExpectStreamingResponse || changed
	request.Model = dispatch.upstreamModel
	request.Stream = ctx.ExpectStreamingResponse
	decisionChanged, err := r.applyDispatchDecision(request, dispatch, ctx)
	if err != nil {
		return false, err
	}
	changed = decisionChanged || changed
	paramsChanged, err := r.applyDispatchRequestParams(request, ctx)
	return paramsChanged || changed, err
}

func (r *OpenAIRouter) applyDispatchDecision(
	request *llmprotocol.Request,
	dispatch *providerDispatch,
	ctx *RequestContext,
) (bool, error) {
	if dispatch.decisionName == "" {
		return false, nil
	}
	changed := false
	if dispatch.targetFormat != llmprotocol.OpenAIChatV1 {
		changed = r.applySemanticReasoningMode(
			request, dispatch.logicalModel, dispatch.targetFormat, dispatch.useReasoning, ctx.VSRSelectedDecision,
		)
	}
	injected, err := r.addSemanticSystemPromptIfConfigured(
		request, dispatch.decisionName, dispatch.logicalModel, ctx,
	)
	return changed || injected, err
}

func (r *OpenAIRouter) applyDispatchRequestParams(
	request *llmprotocol.Request,
	ctx *RequestContext,
) (bool, error) {
	if ctx.VSRSelectedDecision != nil && ctx.VSRSelectedDecision.GetRequestParamsConfig() != nil {
		return r.applySemanticRequestParams(
			ctx.VSRSelectedDecision, request, ctx.Routing.RecipeName(),
		)
	}
	return false, nil
}

func wireFormatForModel(apiFormat string) (llmprotocol.WireFormat, error) {
	switch strings.ToLower(strings.TrimSpace(apiFormat)) {
	case "", config.APIFormatOpenAI, "openai.chat", string(llmprotocol.OpenAIChatV1):
		return llmprotocol.OpenAIChatV1, nil
	case config.APIFormatAnthropic, "anthropic.messages", string(llmprotocol.AnthropicMessagesV1):
		return llmprotocol.AnthropicMessagesV1, nil
	case config.APIFormatResponses, "openai.responses", string(llmprotocol.OpenAIResponsesV1):
		return llmprotocol.OpenAIResponsesV1, nil
	case config.APIFormatImages, "openai.images", string(llmprotocol.OpenAIImagesV1):
		return llmprotocol.OpenAIImagesV1, nil
	default:
		return "", fmt.Errorf("unsupported API format %q", apiFormat)
	}
}

func (r *OpenAIRouter) buildProviderDispatchResponse(
	dispatch *providerDispatch,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	if dispatch == nil {
		return r.createErrorResponse(500, "Internal routing error. Contact your administrator.")
	}
	state := &routeHeaderState{
		setHeaders: r.startUpstreamSpanAndInjectHeaders(
			dispatch.logicalModel, dispatch.backendAddress, ctx,
		),
		removeHeaders: []string{"content-length"},
		profile:       dispatch.profile,
	}
	// Provider metadata is applied before credentials so an operator-supplied
	// extra header can never replace the credential selected for this request.
	appendProfileHeaders(&state.setHeaders, dispatch.profile)
	if errorResponse := r.appendProviderCredential(
		state, dispatch.logicalModel, dispatch.backendName, ctx,
	); errorResponse != nil {
		return errorResponse
	}
	appendRoutingHeaders(&state.setHeaders, dispatch.logicalModel)
	r.appendReliabilityHeaders(&state.setHeaders, dispatch.logicalModel)
	setProviderRequestPath(&state.setHeaders, dispatch.profile, dispatch.targetFormat)
	r.applyDecisionHeaderMutations(state, ctx)
	return buildRequestBodyContinueResponse(state, nil, false)
}

// finalizeProviderDispatchResponse serializes the request only after every
// semantic plugin has run. This prevents late tool-selection mutations from
// being lost and keeps provider wire concerns at one boundary.
func (r *OpenAIRouter) finalizeProviderDispatchResponse(
	dispatch *providerDispatch,
	response *ext_proc.ProcessingResponse,
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	if dispatch == nil || response == nil {
		return nil, status.Error(codes.Internal, "provider dispatch is unavailable")
	}
	captureRequestDemand(
		ctx,
		requestDemandStageProviderBound,
		ctx.SemanticRequest,
		dispatch.logicalModel,
	)
	body, err := r.encodeDispatchRequest(ctx)
	if err != nil {
		metrics.RecordRequestError(dispatch.logicalModel, "serialization_error")
		return nil, dispatchWireError(err, ctx, "encode provider request")
	}
	body, err = r.adaptProviderRequest(body, dispatch, ctx)
	if err != nil {
		metrics.RecordRequestError(dispatch.logicalModel, "provider_adapter_error")
		return nil, dispatchWireError(err, ctx, "adapt provider request")
	}
	common := response.GetRequestBody().GetResponse()
	if common == nil {
		return nil, status.Error(codes.Internal, "provider dispatch response is unavailable")
	}
	if common.HeaderMutation == nil {
		common.HeaderMutation = &ext_proc.HeaderMutation{}
	}
	appendContentLengthHeader(&common.HeaderMutation.SetHeaders, len(body))
	common.BodyMutation = &ext_proc.BodyMutation{
		Mutation: &ext_proc.BodyMutation_Body{Body: body},
	}
	logging.ComponentDebugEvent("extproc", "provider_dispatch_encoded", map[string]interface{}{
		"request_id":  ctx.RequestID,
		"model":       dispatch.logicalModel,
		"wire_format": dispatch.targetFormat,
		"body_bytes":  len(body),
	})
	return response, nil
}

// processBodyRoutingError answers every ProtocolError it recognizes with HTTP
// 400, so only client-owned categories may reach it unwrapped. status.Errorf
// formats with Sprintf, which flattens the error and hides it from errors.As,
// keeping server-owned categories on the internal path where they belong.
func dispatchWireError(err error, ctx *RequestContext, reason string) error {
	var protocolError *llmprotocol.ProtocolError
	if errors.As(err, &protocolError) && isClientProtocolError(protocolError.Category) {
		if ctx != nil {
			ctx.ImmediateProtocolError = protocolError
		}
		return err
	}
	return status.Errorf(codes.Internal, "%s: %v", reason, err)
}

// Categories a caller can fix by changing the request. Everything else,
// including ErrorInternal and the upstream categories, is a server fault.
func isClientProtocolError(category llmprotocol.ErrorCategory) bool {
	return category == llmprotocol.ErrorInvalidRequest ||
		category == llmprotocol.ErrorUnsupportedFeature
}

func (r *OpenAIRouter) startUpstreamSpanAndInjectHeaders(
	model string,
	endpoint string,
	ctx *RequestContext,
) []*core.HeaderValueOption {
	spanContext, upstreamSpan := tracing.StartSpan(
		ctx.TraceContext, tracing.SpanUpstreamRequest, trace.WithSpanKind(trace.SpanKindClient),
	)
	ctx.TraceContext = spanContext
	ctx.UpstreamSpan = upstreamSpan
	tracing.SetSpanAttributes(upstreamSpan,
		attribute.String(tracing.AttrModelName, model),
		attribute.String(tracing.AttrEndpointAddress, endpoint),
	)
	traceHeaders := tracing.InjectTraceContextToSlice(spanContext)
	result := make([]*core.HeaderValueOption, 0, len(traceHeaders))
	for _, header := range traceHeaders {
		result = append(result, &core.HeaderValueOption{Header: &core.HeaderValue{
			Key: header[0], RawValue: []byte(header[1]),
		}})
	}
	return result
}

func resolveProviderAuth(profile *config.ProviderProfile) (authz.LLMProvider, modelcatalog.ProviderAuth, error) {
	if profile == nil {
		return authz.ProviderOpenAI, modelcatalog.ProviderAuth{
			Strategy: "bearer", Header: "Authorization", Prefix: "Bearer",
		}, nil
	}
	providerType, err := profile.ProviderType()
	if err != nil {
		return "", modelcatalog.ProviderAuth{}, fmt.Errorf("resolve provider auth: %w", err)
	}
	providerAuth, err := profile.ResolveAuth()
	if err != nil {
		return "", modelcatalog.ProviderAuth{}, fmt.Errorf("resolve provider auth header: %w", err)
	}
	return authz.LLMProvider(providerType), providerAuth, nil
}

func (r *OpenAIRouter) appendProviderCredential(
	state *routeHeaderState,
	model string,
	backendName string,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	provider, providerAuth, err := resolveProviderAuth(state.profile)
	if err != nil {
		return r.createErrorResponse(500, "Internal routing error. Contact your administrator.")
	}
	if providerAuth.Strategy == "none" {
		if r.CredentialResolver != nil {
			state.removeHeaders = append(state.removeHeaders, r.CredentialResolver.HeadersToStrip()...)
		}
		return nil
	}
	if r.CredentialResolver == nil {
		return r.createErrorResponse(500, "Provider credentials are unavailable.")
	}
	state.removeHeaders = append(state.removeHeaders, r.CredentialResolver.HeadersToStrip()...)
	accessKey, err := r.CredentialResolver.KeyForProvider(provider, model, ctx.Headers)
	if err != nil {
		logging.ComponentErrorEvent("extproc", "credential_resolution_failed", map[string]interface{}{
			"request_id": ctx.RequestID, "model": model, "backend": backendName,
		})
		return r.createErrorResponse(401, "Authentication failed. Check your API key configuration.")
	}
	if accessKey == "" {
		return nil
	}
	value := accessKey
	if providerAuth.Prefix != "" {
		value = providerAuth.Prefix + " " + accessKey
	}
	state.setHeaders = append(state.setHeaders, overwriteRequestHeader(providerAuth.Header, value))
	return nil
}

func appendProfileHeaders(headersOut *[]*core.HeaderValueOption, profile *config.ProviderProfile) {
	if profile == nil {
		return
	}
	for key, value := range profile.ExtraHeaders {
		*headersOut = append(*headersOut, overwriteRequestHeader(key, value))
	}
}

func overwriteRequestHeader(key, value string) *core.HeaderValueOption {
	return &core.HeaderValueOption{
		Header:       &core.HeaderValue{Key: key, RawValue: []byte(value)},
		AppendAction: core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD,
	}
}

func setProviderRequestPath(
	headersOut *[]*core.HeaderValueOption,
	profile *config.ProviderProfile,
	format llmprotocol.WireFormat,
) {
	requestPath := requestWirePath(format)
	if profile != nil {
		if configured, err := profile.ResolveCreatePath(requestWireProtocol(format)); err == nil && configured != "" {
			requestPath = configured
		}
	}
	*headersOut = append(*headersOut, &core.HeaderValueOption{Header: &core.HeaderValue{
		Key: ":path", RawValue: []byte(requestPath),
	}})
}

func appendRoutingHeaders(headersOut *[]*core.HeaderValueOption, model string) {
	if model == "" {
		return
	}
	*headersOut = append(*headersOut, &core.HeaderValueOption{Header: &core.HeaderValue{
		Key: headers.SelectedModel, RawValue: []byte(model),
	}})
}

func (r *OpenAIRouter) appendReliabilityHeaders(headersOut *[]*core.HeaderValueOption, model string) {
	if r == nil || r.Config == nil || model == "" {
		return
	}
	modelParams, ok := r.Config.ModelConfig[model]
	if !ok {
		return
	}
	rel := modelParams.Reliability
	if rel.RequestTimeout != "" {
		if dur, err := time.ParseDuration(rel.RequestTimeout); err == nil {
			timeoutMs := dur.Milliseconds()
			*headersOut = append(*headersOut, &core.HeaderValueOption{Header: &core.HeaderValue{
				Key:      "x-envoy-upstream-rq-timeout-ms",
				RawValue: []byte(strconv.FormatInt(timeoutMs, 10)),
			}})
		}
	}
}

func appendContentLengthHeader(headersOut *[]*core.HeaderValueOption, bodyLength int) {
	*headersOut = append(*headersOut, &core.HeaderValueOption{Header: &core.HeaderValue{
		Key: "content-length", RawValue: []byte(fmt.Sprintf("%d", bodyLength)),
	}})
}

func (r *OpenAIRouter) applyDecisionHeaderMutations(state *routeHeaderState, ctx *RequestContext) {
	if ctx == nil || ctx.VSRSelectedDecision == nil {
		return
	}
	setHeaders, removeHeaders := r.buildHeaderMutations(ctx.VSRSelectedDecision)
	state.setHeaders = append(state.setHeaders, setHeaders...)
	state.removeHeaders = append(state.removeHeaders, removeHeaders...)
}

func buildRequestBodyContinueResponse(
	state *routeHeaderState,
	bodyMutation *ext_proc.BodyMutation,
	clearRouteCache bool,
) *ext_proc.ProcessingResponse {
	return &ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_RequestBody{
		RequestBody: &ext_proc.BodyResponse{Response: &ext_proc.CommonResponse{
			Status: ext_proc.CommonResponse_CONTINUE, ClearRouteCache: clearRouteCache,
			HeaderMutation: &ext_proc.HeaderMutation{
				SetHeaders: state.setHeaders, RemoveHeaders: state.removeHeaders,
			},
			BodyMutation: bodyMutation,
		}},
	}}
}

func (r *OpenAIRouter) getModelParams() map[string]config.ModelParams {
	if r == nil || r.Config == nil {
		return nil
	}
	return r.Config.ModelConfig
}
