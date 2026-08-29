package extproc

import (
	"fmt"
	"net/url"
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
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
//
//nolint:cyclop // Dispatch validates each required routing boundary before Envoy forwarding.
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
	changed, err := r.materializeResponseObjectContext(request, ctx)
	if err != nil {
		return nil, err
	}
	upstreamModel := r.Config.ResolveExternalModelID(logicalModel, backendName)
	changed = request.Model != upstreamModel || request.Stream != ctx.ExpectStreamingResponse || changed
	request.Model = upstreamModel
	request.Stream = ctx.ExpectStreamingResponse
	if decisionName != "" {
		if targetFormat != llmprotocol.OpenAIChatV1 {
			changed = r.applySemanticReasoningMode(request, logicalModel, targetFormat, useReasoning, ctx.VSRSelectedDecision) || changed
		}
		injected, injectErr := r.addSemanticSystemPromptIfConfigured(
			request, decisionName, logicalModel, ctx,
		)
		if injectErr != nil {
			return nil, injectErr
		}
		changed = injected || changed
	}
	if ctx.VSRSelectedDecision != nil && ctx.VSRSelectedDecision.GetRequestParamsConfig() != nil {
		paramsChanged, paramsErr := r.applySemanticRequestParams(
			ctx.VSRSelectedDecision, request, ctx.Routing.RecipeName(),
		)
		if paramsErr != nil {
			return nil, paramsErr
		}
		changed = paramsChanged || changed
	}
	if changed {
		request.Generation++
	}
	ctx.TargetFormat = targetFormat
	ctx.SemanticRequest = request
	logging.ComponentDebugEvent("extproc", "provider_dispatch_prepared", map[string]interface{}{
		"request_id":  ctx.RequestID,
		"model":       logicalModel,
		"backend":     backendName,
		"wire_format": targetFormat,
	})
	return &providerDispatch{
		logicalModel: logicalModel, upstreamModel: upstreamModel,
		backendAddress: backendAddress, backendName: backendName,
		profile: profile, targetFormat: targetFormat,
		decisionName: decisionName, useReasoning: useReasoning,
	}, nil
}

func wireFormatForModel(apiFormat string) (llmprotocol.WireFormat, error) {
	switch strings.ToLower(strings.TrimSpace(apiFormat)) {
	case "", config.APIFormatOpenAI, "openai.chat", string(llmprotocol.OpenAIChatV1):
		return llmprotocol.OpenAIChatV1, nil
	case config.APIFormatAnthropic, "anthropic.messages", string(llmprotocol.AnthropicMessagesV1):
		return llmprotocol.AnthropicMessagesV1, nil
	case config.APIFormatResponses, "openai.responses", string(llmprotocol.OpenAIResponsesV1):
		return llmprotocol.OpenAIResponsesV1, nil
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
	if errorResponse := r.appendProviderCredential(
		state, dispatch.logicalModel, dispatch.backendName, ctx,
	); errorResponse != nil {
		return errorResponse
	}
	appendProfileHeaders(&state.setHeaders, dispatch.profile)
	appendRoutingHeaders(&state.setHeaders, dispatch.logicalModel)
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
	body, err := r.encodeDispatchRequest(ctx)
	if err != nil {
		metrics.RecordRequestError(dispatch.logicalModel, "serialization_error")
		return nil, status.Errorf(codes.Internal, "encode provider request: %v", err)
	}
	body, err = r.adaptProviderRequest(body, dispatch, ctx)
	if err != nil {
		metrics.RecordRequestError(dispatch.logicalModel, "provider_adapter_error")
		return nil, status.Errorf(codes.Internal, "adapt provider request: %v", err)
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

func resolveProviderAuth(profile *config.ProviderProfile) (authz.LLMProvider, string, string, error) {
	if profile == nil {
		return authz.ProviderOpenAI, "Authorization", "Bearer", nil
	}
	providerType, err := profile.ProviderType()
	if err != nil {
		return "", "", "", fmt.Errorf("resolve provider auth: %w", err)
	}
	header, prefix, err := profile.ResolveAuthHeader()
	if err != nil {
		return "", "", "", fmt.Errorf("resolve provider auth header: %w", err)
	}
	return authz.LLMProvider(providerType), header, prefix, nil
}

func (r *OpenAIRouter) appendProviderCredential(
	state *routeHeaderState,
	model string,
	backendName string,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	provider, authHeader, authPrefix, err := resolveProviderAuth(state.profile)
	if err != nil {
		return r.createErrorResponse(500, "Internal routing error. Contact your administrator.")
	}
	if r.CredentialResolver == nil {
		return r.createErrorResponse(500, "Provider credentials are unavailable.")
	}
	accessKey, err := r.CredentialResolver.KeyForProvider(provider, model, ctx.Headers)
	if err != nil {
		logging.ComponentErrorEvent("extproc", "credential_resolution_failed", map[string]interface{}{
			"request_id": ctx.RequestID, "model": model, "backend": backendName,
		})
		return r.createErrorResponse(401, "Authentication failed. Check your API key configuration.")
	}
	state.removeHeaders = append(state.removeHeaders, r.CredentialResolver.HeadersToStrip()...)
	if accessKey == "" {
		return nil
	}
	value := accessKey
	if authPrefix != "" {
		value = authPrefix + " " + accessKey
	}
	state.setHeaders = append(state.setHeaders, &core.HeaderValueOption{Header: &core.HeaderValue{
		Key: authHeader, RawValue: []byte(value),
	}})
	return nil
}

func appendProfileHeaders(headersOut *[]*core.HeaderValueOption, profile *config.ProviderProfile) {
	if profile == nil {
		return
	}
	for key, value := range profile.ExtraHeaders {
		*headersOut = append(*headersOut, &core.HeaderValueOption{Header: &core.HeaderValue{
			Key: key, RawValue: []byte(value),
		}})
	}
}

func setProviderRequestPath(
	headersOut *[]*core.HeaderValueOption,
	profile *config.ProviderProfile,
	format llmprotocol.WireFormat,
) {
	requestPath := requestWirePath(format)
	if profile != nil && format == llmprotocol.OpenAIChatV1 {
		if configured, err := profile.ResolveChatPath(); err == nil && configured != "" {
			requestPath = configured
		}
	} else if profile != nil {
		requestPath = providerProtocolPath(profile.BaseURL, requestPath)
	}
	*headersOut = append(*headersOut, &core.HeaderValueOption{Header: &core.HeaderValue{
		Key: ":path", RawValue: []byte(requestPath),
	}})
}

// providerProtocolPath preserves a provider's base path while allowing the
// selected model protocol to own the endpoint suffix. ChatPath remains a
// chat-completions override and must not redirect Responses or Messages
// dispatches back to the chat endpoint.
func providerProtocolPath(baseURL, protocolPath string) string {
	parsed, err := url.Parse(baseURL)
	if err != nil {
		return protocolPath
	}
	basePath := strings.TrimRight(parsed.Path, "/")
	if basePath == "" || basePath == "/" || strings.HasPrefix(protocolPath, basePath+"/") {
		return protocolPath
	}
	if strings.HasSuffix(basePath, "/v1") && strings.HasPrefix(protocolPath, "/v1/") {
		return basePath + strings.TrimPrefix(protocolPath, "/v1")
	}
	return basePath + protocolPath
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
