package extproc

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"strings"

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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
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

const preparedDispatchReceiptVersion = 1

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
	if err := r.prepareDispatchContextOverflow(ctx, request, dispatch.logicalModel); err != nil {
		return nil, err
	}
	if err := r.prepareAutomaticDispatch(ctx, request, dispatch); err != nil {
		return nil, err
	}
	// Selection already compared capable candidates. Late mutations may still
	// invalidate the result, but cannot restart routing or bypass its policies.
	if protocolErr := r.rejectDispatchCapabilityMismatch(request, dispatch, ctx); protocolErr != nil {
		return nil, protocolErr
	}
	ctx.TargetFormat = dispatch.targetFormat
	// Bind response policy where the backend is selected.
	ctx.ResponseVendor = resolveResponseVendor(dispatch.profile)
	ctx.SemanticRequest = request
	// Per-model accounting keys off the validated concrete dispatch model.
	ctx.RequestModel = dispatch.logicalModel

	ctx.VSRSelectedModel = dispatch.logicalModel
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

// rejectDispatchCapabilityMismatch applies both wire fidelity and declared
// model task constraints to the primary dispatch, using the same qualification
// as fallback candidates. A wire's ability to encode a task does not establish
// that the selected model can execute it.
func (r *OpenAIRouter) rejectDispatchCapabilityMismatch(
	request *llmprotocol.Request,
	dispatch *providerDispatch,
	ctx *RequestContext,
) error {
	projected, err := r.projectAnthropicRequestForBackend(*request, dispatch.logicalModel, dispatch.targetFormat)
	if err != nil {
		var protocolError *llmprotocol.ProtocolError
		if errors.As(err, &protocolError) && ctx != nil {
			ctx.ImmediateProtocolError = protocolError
		}
		return err
	}
	if err := r.validateDispatchRequirements(&projected, dispatch, ctx); err != nil {
		return err
	}
	if err := r.providerCapabilityMismatch(dispatch.logicalModel, dispatch.targetFormat, llmprotocol.RequiredCapabilities(projected)); err != nil {
		var protocolError *llmprotocol.ProtocolError
		if errors.As(err, &protocolError) && ctx != nil {
			ctx.ImmediateProtocolError = protocolError
		}
		return err
	}
	return nil
}

// declaredModelCapabilities projects recognized model facts without allowing
// descriptive catalog labels to erase their constraints. A model with no
// recognized declaration retains the unannotated compatibility behavior.
func (r *OpenAIRouter) declaredModelCapabilities(model string) (llmprotocol.CapabilitySet, bool) {
	if r == nil || r.Config == nil {
		return llmprotocol.CapabilitySet{}, false
	}
	params, ok := r.Config.ModelConfig[model]
	if !ok || len(params.Capabilities) == 0 {
		return llmprotocol.CapabilitySet{}, false
	}
	return llmprotocol.ModelCapabilities(params.Capabilities)
}

func (r *OpenAIRouter) providerCapabilityMismatch(model string, format llmprotocol.WireFormat, required llmprotocol.CapabilitySet) error {
	available, ok := r.codecCapabilitiesForFormat(format)
	if !ok {
		return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_capability", fmt.Sprintf("model %q has no codec for %q", model, format), nil)
	}
	if err := llmprotocol.RequireCapabilities(format, available, required); err != nil {
		return err
	}
	if declared, annotated := r.declaredModelCapabilities(model); annotated && !declared.Contains(required.TaskCapabilities()) {
		return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_capability",
			fmt.Sprintf("model %q does not declare the required tasks: %s", model, strings.Join(required.TaskCapabilities().Names(), ", ")), nil)
	}
	return nil
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
	changed, err := r.prepareModelInput(request, ctx)
	if err != nil {
		return false, err
	}
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
			request, dispatch.logicalModel, dispatch.targetFormat, dispatch.useReasoning, ctx.decisionForCandidate(dispatch.logicalModel),
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
		setHeaders:    r.startUpstreamSpanAndInjectHeaders(dispatch, ctx),
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
	setProviderRequestPath(&state.setHeaders, dispatch.profile, dispatch.targetFormat)
	r.applyDecisionHeaderMutations(state, ctx)
	// Body-stage model and path mutations can change the Envoy route selected
	// during headers. Apply the same cache policy to every provider dispatch,
	// including internal multi-model calls and unchanged logical model names.
	return buildRequestBodyContinueResponse(state, nil, r.shouldClearRouteCache())
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
	if err := selectionRequestContext(ctx).Err(); err != nil {
		return nil, err
	}
	if ctx != nil {
		ctx.preparedDispatchReceipt = nil
	}
	if response.GetImmediateResponse() != nil {
		return response, nil
	}
	if ctx != nil && ctx.SemanticRequest != nil {
		if err := r.prepareAutomaticDispatch(ctx, ctx.SemanticRequest, dispatch); err != nil {
			return nil, err
		}
		if err := r.rejectDispatchCapabilityMismatch(ctx.SemanticRequest, dispatch, ctx); err != nil {
			return nil, err
		}
		if r.shouldAttemptFallback(ctx) {
			snapshot, err := cloneSemanticRequestForReplay(ctx.SemanticRequest)
			if err != nil {
				return nil, status.Errorf(codes.Internal, "capture fallback request: %v", err)
			}
			ctx.FallbackRequest = snapshot
		}
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
	if err := commitAgenticSessionDecision(ctx); err != nil {
		return nil, err
	}
	bindPreparedDispatchArtifact(common, ctx, body, dispatch.targetFormat)
	logging.ComponentDebugEvent("extproc", "provider_dispatch_encoded", map[string]interface{}{
		"request_id":  ctx.RequestID,
		"model":       dispatch.logicalModel,
		"wire_format": dispatch.targetFormat,
		"body_bytes":  len(body),
	})
	return response, nil
}

// bindPreparedDispatchArtifact couples the payload-free primary-dispatch
// receipt to the exact final byte slice returned to Envoy. The caller has
// already completed codec encoding and provider adaptation; this helper
// performs no transformation. Existing Replay records, including Looper's
// per-attempt records, are intentionally outside this no-updater contract.
func bindPreparedDispatchArtifact(
	common *ext_proc.CommonResponse,
	ctx *RequestContext,
	body []byte,
	format llmprotocol.WireFormat,
) {
	common.BodyMutation = &ext_proc.BodyMutation{
		Mutation: &ext_proc.BodyMutation_Body{Body: body},
	}
	if !shouldStartRouterReplay(ctx) {
		return
	}
	digest := sha256.Sum256(body)
	ctx.preparedDispatchReceipt = &routerreplay.PreparedDispatchReceipt{
		Version: preparedDispatchReceiptVersion, WireFormat: string(format),
		SHA256: fmt.Sprintf("%x", digest), ByteLength: len(body),
	}
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
	dispatch *providerDispatch,
	ctx *RequestContext,
) []*core.HeaderValueOption {
	spanContext, upstreamSpan := tracing.StartSpan(
		ctx.TraceContext, tracing.SpanUpstreamRequest, trace.WithSpanKind(trace.SpanKindClient),
		trace.WithAttributes(genAIRequestAttributes(dispatch)...),
	)
	ctx.UpstreamSpan = upstreamSpan
	tracing.SetSpanAttributes(upstreamSpan,
		attribute.String(tracing.AttrModelName, dispatch.logicalModel),
		attribute.String(tracing.AttrEndpointAddress, dispatch.backendAddress),
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
