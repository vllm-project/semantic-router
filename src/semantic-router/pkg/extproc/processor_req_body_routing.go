package extproc

import (
	"fmt"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
)

type routeHeaderState struct {
	setHeaders    []*core.HeaderValueOption
	removeHeaders []string
	profile       *config.ProviderProfile
}

func (r *OpenAIRouter) selectEndpointForModel(ctx *RequestContext, model string) (string, string, error) {
	endpointAddress, endpointName, endpointFound, err := r.Config.SelectBestEndpointWithDetailsForModel(model)
	if err != nil {
		return "", "", fmt.Errorf("endpoint resolution for model %q: %w", model, err)
	}
	if endpointFound {
		logging.Infof("Selected endpoint address: %s (name: %s) for model: %s", endpointAddress, endpointName, model)
	}

	ctx.SelectedEndpoint = endpointAddress
	metrics.IncrementModelActiveRequests(model)

	return endpointAddress, endpointName, nil
}

// resolveModelNameForEndpoint resolves the model name alias to the real model name
// that the backend endpoint expects, using external_model_ids configuration.
func (r *OpenAIRouter) resolveModelNameForEndpoint(modelName string, endpointName string) string {
	if r.Config == nil {
		return modelName
	}
	resolved := r.Config.ResolveExternalModelID(modelName, endpointName)
	if resolved != modelName {
		logging.Infof("Resolved model name: %s -> %s (endpoint: %s)", modelName, resolved, endpointName)
	}
	return resolved
}

func (r *OpenAIRouter) modifyRequestBodyForAutoRouting(
	openAIRequest *openai.ChatCompletionNewParams,
	matchedModel string,
	decisionName string,
	useReasoning bool,
	ctx *RequestContext,
) ([]byte, error) {
	openAIRequest.Model = matchedModel

	modifiedBody, err := serializeOpenAIRequestWithStream(openAIRequest, ctx.ExpectStreamingResponse)
	if err != nil {
		logging.Errorf("Error serializing modified request: %v", err)
		metrics.RecordRequestError(matchedModel, "serialization_error")
		return nil, status.Errorf(codes.Internal, "error serializing modified request: %v", err)
	}

	if decisionName != "" {
		modifiedBody, err = r.setReasoningModeToRequestBody(modifiedBody, useReasoning, decisionName)
		if err != nil {
			logging.Errorf("Error setting reasoning mode %v to request: %v", useReasoning, err)
			metrics.RecordRequestError(matchedModel, "serialization_error")
			return nil, status.Errorf(codes.Internal, "error setting reasoning mode: %v", err)
		}

		modifiedBody, err = r.addSystemPromptIfConfigured(modifiedBody, decisionName, matchedModel, ctx)
		if err != nil {
			return nil, err
		}
	}

	if ctx.MemoryContext != "" {
		modifiedBody, err = injectMemoryMessages(modifiedBody, ctx.MemoryContext)
		if err != nil {
			logging.Warnf("Memory: Failed to inject memory context: %v", err)
		}
	}

	return modifiedBody, nil
}

// startUpstreamSpanAndInjectHeaders starts an upstream request span and returns trace context headers.
func (r *OpenAIRouter) startUpstreamSpanAndInjectHeaders(model string, endpoint string, ctx *RequestContext) []*core.HeaderValueOption {
	var traceContextHeaders []*core.HeaderValueOption

	spanCtx, upstreamSpan := tracing.StartSpan(ctx.TraceContext, tracing.SpanUpstreamRequest,
		trace.WithSpanKind(trace.SpanKindClient))
	ctx.TraceContext = spanCtx
	ctx.UpstreamSpan = upstreamSpan

	tracing.SetSpanAttributes(upstreamSpan,
		attribute.String(tracing.AttrModelName, model),
		attribute.String(tracing.AttrEndpointAddress, endpoint))

	traceHeaders := tracing.InjectTraceContextToSlice(spanCtx)
	for _, th := range traceHeaders {
		traceContextHeaders = append(traceContextHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      th[0],
				RawValue: []byte(th[1]),
			},
		})
	}

	return traceContextHeaders
}

// resolveProviderAuth determines the LLMProvider, auth header name, and auth prefix from a provider profile.
func resolveProviderAuth(profile *config.ProviderProfile) (authz.LLMProvider, string, string, error) {
	if profile == nil {
		return authz.ProviderOpenAI, "Authorization", "Bearer", nil
	}
	providerType, err := profile.ProviderType()
	if err != nil {
		return "", "", "", fmt.Errorf("resolving provider auth: %w", err)
	}
	llmProvider := authz.LLMProvider(providerType)
	authHeader, authPrefix, err := profile.ResolveAuthHeader()
	if err != nil {
		return "", "", "", fmt.Errorf("resolving auth header: %w", err)
	}
	return llmProvider, authHeader, authPrefix, nil
}

// createRoutingResponse creates a routing response with request header/body mutations.
func (r *OpenAIRouter) createRoutingResponse(
	model string,
	endpoint string,
	endpointName string,
	modifiedBody []byte,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	bodyMutation := &ext_proc.BodyMutation{
		Mutation: &ext_proc.BodyMutation_Body{
			Body: modifiedBody,
		},
	}
	logging.Infof("createRoutingResponse: modifiedBody length=%d, model=%s", len(modifiedBody), model)

	contentLength := len(modifiedBody)
	state, errorResponse := r.buildRouteHeaderState(model, endpoint, endpointName, ctx, &contentLength)
	if errorResponse != nil {
		return errorResponse
	}
	if _, errorResponse := r.applyRoutingPathHeader(state, endpointName, ctx, false); errorResponse != nil {
		return errorResponse
	}
	r.applyDecisionHeaderMutations(state, ctx)

	return buildRequestBodyContinueResponse(state, bodyMutation, false)
}

// createSpecifiedModelResponse creates a response for specified-model routing.
func (r *OpenAIRouter) createSpecifiedModelResponse(
	model string,
	upstreamModel string,
	endpoint string,
	endpointName string,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	state, errorResponse := r.buildRouteHeaderState(model, endpoint, endpointName, ctx, nil)
	if errorResponse != nil {
		return errorResponse
	}

	needsBodyMutation := upstreamModel != model
	if needsBodyMutation {
		logging.Infof("Model name rewriting: %s -> %s in request body", model, upstreamModel)
	}

	// RAG/memory injection modifies ctx.OriginalRequestBody. Without a body
	// mutation Envoy forwards the original (pre-injection) body, silently
	// dropping the personalized context. Force the mutation so the modified
	// body actually reaches the upstream model.
	if !needsBodyMutation && (ctx.RAGRetrievedContext != "" || ctx.MemoryContext != "") {
		logging.Infof("Forcing body mutation: personalized context injected (RAG=%d, memory=%d chars) for specified model %s",
			len(ctx.RAGRetrievedContext), len(ctx.MemoryContext), model)
		needsBodyMutation = true
	}

	pathMutatesBody, errorResponse := r.applyRoutingPathHeader(state, endpointName, ctx, true)
	if errorResponse != nil {
		return errorResponse
	}

	bodyMutation := r.buildSpecifiedModelBodyMutation(model, upstreamModel, needsBodyMutation || pathMutatesBody, state, ctx)
	return buildRequestBodyContinueResponse(state, bodyMutation, false)
}

func (r *OpenAIRouter) buildRouteHeaderState(
	model string,
	endpoint string,
	endpointName string,
	ctx *RequestContext,
	contentLength *int,
) (*routeHeaderState, *ext_proc.ProcessingResponse) {
	state := &routeHeaderState{
		setHeaders:    []*core.HeaderValueOption{},
		removeHeaders: []string{},
	}
	if contentLength != nil {
		state.removeHeaders = append(state.removeHeaders, "content-length")
		if *contentLength > 0 {
			appendContentLengthHeader(&state.setHeaders, *contentLength)
		}
	}

	state.setHeaders = append(state.setHeaders, r.startUpstreamSpanAndInjectHeaders(model, endpoint, ctx)...)

	profile, profileErr := r.Config.GetProviderProfileForEndpoint(endpointName)
	if profileErr != nil {
		logging.Errorf("Provider profile resolution failed for endpoint %s: %v", endpointName, profileErr)
		return nil, r.createErrorResponse(500, "Internal routing error. Contact your administrator.")
	}
	state.profile = profile

	if errorResponse := r.appendCredentialHeaders(state, model, endpointName, ctx); errorResponse != nil {
		return nil, errorResponse
	}
	state.removeHeaders = append(state.removeHeaders, r.CredentialResolver.HeadersToStrip()...)
	appendProfileHeaders(&state.setHeaders, profile)
	appendRoutingHeaders(&state.setHeaders, model, endpoint)

	return state, nil
}

func (r *OpenAIRouter) appendCredentialHeaders(
	state *routeHeaderState,
	model string,
	endpointName string,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	llmProvider, authHeader, authPrefix, authErr := resolveProviderAuth(state.profile)
	if authErr != nil {
		logging.Errorf("Provider auth resolution failed for endpoint %s: %v", endpointName, authErr)
		return r.createErrorResponse(500, "Internal routing error. Contact your administrator.")
	}

	accessKey, credErr := r.CredentialResolver.KeyForProvider(llmProvider, model, ctx.Headers)
	if credErr != nil {
		logging.Errorf("Credential resolution failed for model %s: %v", model, credErr)
		return r.createErrorResponse(401, "Authentication failed. Check your API key configuration.")
	}
	if accessKey == "" {
		logging.Debugf("No API key for %s model %q (fail_open=true) — preserving original auth header", llmProvider, model)
		return nil
	}

	value := accessKey
	if authPrefix != "" {
		value = authPrefix + " " + accessKey
	}
	state.setHeaders = append(state.setHeaders, &core.HeaderValueOption{
		Header: &core.HeaderValue{
			Key:      authHeader,
			RawValue: []byte(value),
		},
	})
	logging.Infof("Added %s header for model %s (provider=%s)", authHeader, model, llmProvider)
	return nil
}

func appendProfileHeaders(setHeaders *[]*core.HeaderValueOption, profile *config.ProviderProfile) {
	if profile == nil {
		return
	}
	for key, value := range profile.ExtraHeaders {
		*setHeaders = append(*setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{Key: key, RawValue: []byte(value)},
		})
	}
}

func appendRoutingHeaders(setHeaders *[]*core.HeaderValueOption, model string, endpoint string) {
	if endpoint != "" {
		*setHeaders = append(*setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.GatewayDestinationEndpoint,
				RawValue: []byte(endpoint),
			},
		})
	}
	if model != "" {
		*setHeaders = append(*setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.SelectedModel,
				RawValue: []byte(model),
			},
		})
	}
}

func appendContentLengthHeader(setHeaders *[]*core.HeaderValueOption, bodyLength int) {
	*setHeaders = append(*setHeaders, &core.HeaderValueOption{
		Header: &core.HeaderValue{
			Key:      "content-length",
			RawValue: []byte(fmt.Sprintf("%d", bodyLength)),
		},
	})
}

func (r *OpenAIRouter) applyRoutingPathHeader(
	state *routeHeaderState,
	endpointName string,
	ctx *RequestContext,
	specifiedModel bool,
) (bool, *ext_proc.ProcessingResponse) {
	if ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.IsResponseAPIRequest {
		state.setHeaders = append(state.setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      ":path",
				RawValue: []byte("/v1/chat/completions"),
			},
		})
		if specifiedModel {
			logging.Infof("Response API: Rewriting path to /v1/chat/completions (specified model)")
		} else {
			logging.Infof("Response API: Rewriting path to /v1/chat/completions")
		}
		return specifiedModel, nil
	}
	if state.profile == nil {
		return false, nil
	}

	chatPath, pathErr := state.profile.ResolveChatPath()
	if pathErr != nil {
		logging.Errorf("Chat path resolution failed for endpoint %s: %v", endpointName, pathErr)
		return false, r.createErrorResponse(500, "Internal routing error. Contact your administrator.")
	}
	if chatPath != "" {
		state.setHeaders = append(state.setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      ":path",
				RawValue: []byte(chatPath),
			},
		})
		if specifiedModel {
			logging.Infof("Provider profile: Rewriting path to %s (specified model)", chatPath)
		} else {
			logging.Infof("Provider profile: Rewriting path to %s", chatPath)
		}
	}
	return false, nil
}

func (r *OpenAIRouter) applyDecisionHeaderMutations(state *routeHeaderState, ctx *RequestContext) {
	if ctx.VSRSelectedDecision == nil {
		return
	}

	pluginSetHeaders, pluginRemoveHeaders := r.buildHeaderMutations(ctx.VSRSelectedDecision)
	if len(pluginSetHeaders) > 0 {
		state.setHeaders = append(state.setHeaders, pluginSetHeaders...)
		logging.Infof("Applied %d header mutations from decision %s", len(pluginSetHeaders), ctx.VSRSelectedDecision.Name)
	}
	if len(pluginRemoveHeaders) > 0 {
		state.removeHeaders = append(state.removeHeaders, pluginRemoveHeaders...)
		logging.Infof("Applied %d header deletions from decision %s", len(pluginRemoveHeaders), ctx.VSRSelectedDecision.Name)
	}
}

func (r *OpenAIRouter) buildSpecifiedModelBodyMutation(
	model string,
	upstreamModel string,
	needsBodyMutation bool,
	state *routeHeaderState,
	ctx *RequestContext,
) *ext_proc.BodyMutation {
	if !needsBodyMutation {
		return nil
	}

	state.removeHeaders = append(state.removeHeaders, "content-length")
	bodyBytes := getBodyMutationSource(ctx)
	if upstreamModel != model && len(bodyBytes) > 0 {
		rewritten, err := rewriteModelInBody(bodyBytes, upstreamModel)
		if err != nil {
			logging.Warnf("Failed to rewrite model in body: %v, sending original body", err)
		} else {
			bodyBytes = rewritten
		}
	}
	if len(bodyBytes) == 0 {
		return nil
	}

	appendContentLengthHeader(&state.setHeaders, len(bodyBytes))
	return &ext_proc.BodyMutation{
		Mutation: &ext_proc.BodyMutation_Body{
			Body: bodyBytes,
		},
	}
}

func getBodyMutationSource(ctx *RequestContext) []byte {
	if ctx != nil && ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.IsResponseAPIRequest && len(ctx.ResponseAPICtx.TranslatedBody) > 0 {
		return ctx.ResponseAPICtx.TranslatedBody
	}
	if ctx != nil && len(ctx.OriginalRequestBody) > 0 {
		return ctx.OriginalRequestBody
	}
	return nil
}

func buildRequestBodyContinueResponse(
	state *routeHeaderState,
	bodyMutation *ext_proc.BodyMutation,
	clearRouteCache bool,
) *ext_proc.ProcessingResponse {
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					Status:          ext_proc.CommonResponse_CONTINUE,
					ClearRouteCache: clearRouteCache,
					HeaderMutation: &ext_proc.HeaderMutation{
						SetHeaders:    state.setHeaders,
						RemoveHeaders: state.removeHeaders,
					},
					BodyMutation: bodyMutation,
				},
			},
		},
	}
}

// rewriteModelInBody rewrites the "model" field in a JSON request body.
func rewriteModelInBody(body []byte, newModel string) ([]byte, error) {
	return rewriteModelInBodyFast(body, newModel)
}

// getModelAccessKey retrieves the access_key for a given model from the config.
func (r *OpenAIRouter) getModelAccessKey(modelName string) string {
	if r.Config == nil || r.Config.ModelConfig == nil {
		return ""
	}

	modelConfig, ok := r.Config.ModelConfig[modelName]
	if !ok {
		return ""
	}

	return modelConfig.AccessKey
}

// getModelParams returns model params for looper/model helpers.
func (r *OpenAIRouter) getModelParams() map[string]config.ModelParams {
	if r.Config == nil || r.Config.ModelConfig == nil {
		return nil
	}
	return r.Config.ModelConfig
}
