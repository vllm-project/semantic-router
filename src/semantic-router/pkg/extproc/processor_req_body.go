package extproc

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/openai/openai-go"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/anthropic"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ratelimit"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// handleRequestBody processes the request body
func (r *OpenAIRouter) handleRequestBody(v *ext_proc.ProcessingRequest_RequestBody, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	// Record start time for model routing
	ctx.ProcessingStartTime = time.Now()
	// Save the original request body
	ctx.OriginalRequestBody = v.RequestBody.GetBody()

	// Handle Response API translation if this is a /v1/responses request
	requestBody := ctx.OriginalRequestBody
	if ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.IsResponseAPIRequest && r.ResponseAPIFilter != nil {
		respCtx, translatedBody, err := r.ResponseAPIFilter.TranslateRequest(ctx.TraceContext, requestBody)
		if err != nil {
			logging.Errorf("Response API translation error: %v", err)
			return r.createErrorResponse(400, "Invalid Response API request: "+err.Error()), nil
		}
		if respCtx != nil && translatedBody != nil {
			// Update context with full Response API context
			ctx.ResponseAPICtx = respCtx
			requestBody = translatedBody
			logging.Infof("Response API: Translated to Chat Completions format")
		} else {
			// Translation returned nil - this means the request is missing required fields (e.g., 'input')
			// Return error since the request was sent to /v1/responses but is not valid Response API format
			logging.Errorf("Response API: Request to /v1/responses missing required 'input' field")
			return r.createErrorResponse(400, "Invalid Response API request: 'input' field is required. Use 'input' instead of 'messages' for Response API."), nil
		}
	}

	// Extract stream parameter from original request and update ExpectStreamingResponse if needed
	hasStreamParam := extractStreamParam(requestBody)
	if hasStreamParam {
		logging.Infof("Original request contains stream parameter: true")
		ctx.ExpectStreamingResponse = true // Set this if stream param is found
	}

	// Parse the OpenAI request using SDK types
	openAIRequest, err := parseOpenAIRequest(requestBody)
	if err != nil {
		logging.Errorf("Error parsing OpenAI request: %v", err)
		// Attempt to determine model for labeling (may be unknown here)
		metrics.RecordRequestError(ctx.RequestModel, "parse_error")
		// Count this request as well, with unknown model if necessary
		metrics.RecordModelRequest(ctx.RequestModel)
		return nil, status.Errorf(codes.InvalidArgument, "invalid request body: %v", err)
	}

	// Store the original model
	originalModel := openAIRequest.Model

	// Set the model on context early so error metrics can label it
	if ctx.RequestModel == "" {
		ctx.RequestModel = originalModel
	}

	// Check if this is a looper internal request - if so, execute decision plugins
	// (lookup decision by name and apply configured plugins)
	if r.isLooperRequest(ctx) {
		logging.Infof("[Looper] Internal request detected, executing decision plugins for model: %s", originalModel)
		return r.handleLooperInternalRequestWithPlugins(originalModel, ctx)
	}

	// Get content from messages
	userContent, nonUserMessages := extractUserAndNonUserContent(openAIRequest)

	// Store user content for later use in hallucination detection
	ctx.UserContent = userContent

	// Perform decision evaluation and model selection once at the beginning
	// Use decision-based routing if decisions are configured, otherwise fall back to category-based
	// This also evaluates fact-check signal as part of the signal evaluation
	decisionName, _, reasoningDecision, selectedModel, authzErr := r.performDecisionEvaluation(originalModel, userContent, nonUserMessages, ctx)
	if authzErr != nil {
		// Authz failure is a hard error — return 403 Forbidden.
		// This happens when role_bindings are configured but the x-authz-user-id header is missing.
		logging.Errorf("[Request Body] Authz evaluation failed: %v", authzErr)
		return r.createErrorResponse(403, authzErr.Error()), nil
	}

	// Record the initial request to this model (count all requests)
	metrics.RecordModelRequest(selectedModel)

	// Perform security checks with decision-specific settings
	if response, shouldReturn := r.performJailbreaks(ctx, userContent, nonUserMessages, decisionName); shouldReturn {
		// Record blocked request to replay before returning
		r.startRouterReplay(ctx, originalModel, selectedModel, decisionName)
		r.updateRouterReplayStatus(ctx, 403, false) // 403 Forbidden for jailbreak block
		return response, nil
	}

	// Perform PII detection and policy check (if PII policy is enabled for the decision)
	piiResponse := r.performPIIDetection(ctx, userContent, nonUserMessages, decisionName)
	if piiResponse != nil {
		// Record blocked request to replay before returning
		r.startRouterReplay(ctx, originalModel, selectedModel, decisionName)
		r.updateRouterReplayStatus(ctx, 403, false) // 403 Forbidden for PII block
		// PII policy violation - return error response
		return piiResponse, nil
	}

	// Rate limit check — after security checks, before cache/RAG/routing.
	// This prevents rate-limited users from consuming cache or RAG resources.
	if r.RateLimiter != nil {
		rlCtx := r.buildRateLimitContext(ctx, selectedModel)
		decision, err := r.RateLimiter.Check(rlCtx)
		if err != nil {
			logging.Errorf("[Request Body] Rate limit check error: %v", err)
			return r.createRateLimitResponse(decision), nil
		}
		if decision != nil && !decision.Allowed {
			logging.Infof("[Request Body] Rate limited: user=%s model=%s provider=%s remaining=%d",
				rlCtx.UserID, rlCtx.Model, decision.Provider, decision.Remaining)
			return r.createRateLimitResponse(decision), nil
		}
		// Store context on RequestContext for post-response reporting
		ctx.RateLimitCtx = &rlCtx
	}

	// Handle caching with decision-specific settings
	if response, shouldReturn := r.handleCaching(ctx, decisionName); shouldReturn {
		logging.Infof("handleCaching returned a response, returning immediately")
		return response, nil
	}
	logging.Infof("handleCaching returned no cached response, continuing to model routing")

	// Execute RAG plugin if enabled (after cache check, before other plugins)
	// RAG plugin retrieves context and injects it into the request
	if ragErr := r.executeRAGPlugin(ctx, decisionName); ragErr != nil {
		// If RAG fails with on_failure=block, return error response
		return r.createErrorResponse(503, fmt.Sprintf("RAG retrieval failed: %v", ragErr)), nil
	}

	// If the Responses API request includes an explicit image_generation tool and the
	// modality classifier did not already detect a modality, use the tool presence as a
	// strong signal. With text content present we classify as BOTH; otherwise DIFFUSION.
	if ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.HasImageGenerationTool &&
		(ctx.ModalityClassification == nil || ctx.ModalityClassification.Modality == "" || ctx.ModalityClassification.Modality == ModalityAR) {
		modality := ModalityDiffusion
		if userContent != "" {
			modality = ModalityBoth
		}
		ctx.ModalityClassification = &ModalityClassificationResult{
			Modality:   modality,
			Confidence: 1.0,
			Method:     "image_generation_tool",
		}
		logging.Infof("[ModalityRouter] Explicit image_generation tool detected — forcing modality=%s", modality)
	}

	// Modality routing: if the decision matched a modality signal (DIFFUSION/BOTH),
	// execute image generation. This is driven by the modality signal in the decision engine
	// or by the explicit image_generation tool detection above.
	if resp, err := r.handleModalityFromDecision(ctx, openAIRequest); err != nil {
		logging.Errorf("[ModalityRouter] Error: %v", err)
		return r.createErrorResponse(503, fmt.Sprintf("Modality routing failed: %v", err)), nil
	} else if resp != nil {
		return resp, nil // DIFFUSION/BOTH short-circuit — image already generated
	}

	// Handle memory retrieval (if enabled)
	// Memory retrieval happens after cache check to avoid unnecessary work on cache hits
	// and before model routing to inject memories into LLM context
	requestBody, memErr := r.handleMemoryRetrieval(ctx, userContent, requestBody, openAIRequest)
	if memErr != nil {
		logging.Warnf("Memory retrieval failed: %v, continuing without memory", memErr)
		// Graceful degradation: continue without memory if retrieval fails
	}
	// Update the translated body with injected memories for Response API
	if ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.IsResponseAPIRequest && len(requestBody) > 0 {
		ctx.ResponseAPICtx.TranslatedBody = requestBody
	}

	// Handle model selection and routing with pre-computed classification results and selected model
	return r.handleModelRouting(openAIRequest, originalModel, decisionName, reasoningDecision, selectedModel, ctx)
}

// handleModelRouting handles model selection and routing logic
// decisionName, reasoningDecision, and selectedModel are pre-computed from ProcessRequest
func (r *OpenAIRouter) handleModelRouting(openAIRequest *openai.ChatCompletionNewParams, originalModel string, decisionName string, reasoningDecision entropy.ReasoningDecision, selectedModel string, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	response := &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					Status: ext_proc.CommonResponse_CONTINUE,
				},
			},
		},
	}

	isAutoModel := r.Config != nil && r.Config.IsAutoModelName(originalModel)

	targetModel := originalModel
	if isAutoModel && selectedModel != "" {
		targetModel = selectedModel
	}

	// Anthropic model routing
	if r.Config.GetModelAPIFormat(targetModel) == config.APIFormatAnthropic {
		return r.handleAnthropicRouting(openAIRequest, originalModel, targetModel, decisionName, ctx)
	}

	// OpenAI-compatible routing
	switch {
	case !isAutoModel:
		return r.handleSpecifiedModelRouting(openAIRequest, originalModel, ctx)
	case r.shouldUseLooper(ctx.VSRSelectedDecision):
		logging.Infof("Using Looper for decision %s with algorithm %s",
			ctx.VSRSelectedDecision.Name, ctx.VSRSelectedDecision.Algorithm.Type)
		return r.handleLooperExecution(ctx.TraceContext, openAIRequest, ctx.VSRSelectedDecision, ctx)
	case selectedModel != "":
		return r.handleAutoModelRouting(openAIRequest, originalModel, decisionName, reasoningDecision, selectedModel, ctx, response)
	default:
		// Auto model without selection - no routing needed
		ctx.RequestModel = originalModel
		return response, nil
	}
}

// handleAnthropicRouting handles routing to Anthropic Claude API via Envoy.
// Transforms the request body from OpenAI format to Anthropic format and sets
// appropriate headers for Envoy to route to the Anthropic cluster.
func (r *OpenAIRouter) handleAnthropicRouting(openAIRequest *openai.ChatCompletionNewParams, originalModel string, targetModel string, decisionName string, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	logging.Infof("Routing to Anthropic API via Envoy for model: %s (original: %s)", targetModel, originalModel)

	// Reject streaming requests (not yet supported for Anthropic backend)
	if ctx.ExpectStreamingResponse {
		logging.Warnf("Streaming not supported for Anthropic backend, rejecting request for model: %s", targetModel)
		return r.createErrorResponse(400, "Streaming is not supported for Anthropic models. Please set stream=false in your request."), nil
	}

	// Resolve API key for Anthropic via credential chain (ext_authz headers → static config)
	accessKey, err := r.CredentialResolver.KeyForProvider(authz.ProviderAnthropic, targetModel, ctx.Headers)
	if err != nil {
		return r.createErrorResponse(401, fmt.Sprintf("Credential resolution failed for model %s: %v", targetModel, err)), nil
	}
	if accessKey == "" {
		// fail_open=true path: no key but allowed through — warn operator
		logging.Warnf("No API key for Anthropic model %q (fail_open=true) — request will use empty key", targetModel)
	}

	// Update model in request to target model
	openAIRequest.Model = targetModel

	// Transform request body from OpenAI format to Anthropic format
	anthropicBody, err := anthropic.ToAnthropicRequestBody(openAIRequest)
	if err != nil {
		logging.Errorf("Failed to transform request to Anthropic format: %v", err)
		return r.createErrorResponse(500, fmt.Sprintf("Request transformation error: %v", err)), nil
	}

	// Track VSR decision information
	ctx.RequestModel = targetModel
	ctx.VSRSelectedModel = targetModel
	ctx.APIFormat = config.APIFormatAnthropic // Mark for response transformation
	if decisionName != "" {
		ctx.VSRSelectedDecision = r.Config.GetDecisionByName(decisionName)
	}

	// Build header mutations using anthropic package helpers
	anthropicHeaders := anthropic.BuildRequestHeaders(accessKey, len(anthropicBody))
	setHeaders := make([]*core.HeaderValueOption, 0, len(anthropicHeaders)+2)
	for _, h := range anthropicHeaders {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      h.Key,
				RawValue: []byte(h.Value),
			},
		})
	}

	// Add x-selected-model for Envoy routing
	setHeaders = append(setHeaders, &core.HeaderValueOption{
		Header: &core.HeaderValue{
			Key:      headers.SelectedModel,
			RawValue: []byte(targetModel),
		},
	})

	// Start upstream span and inject trace context headers
	traceContextHeaders := r.startUpstreamSpanAndInjectHeaders(targetModel, "api.anthropic.com", ctx)
	setHeaders = append(setHeaders, traceContextHeaders...)

	// Record routing latency
	r.recordRoutingLatency(ctx)

	logging.Infof("Transformed request for Anthropic API, body size: %d bytes", len(anthropicBody))

	// Strip ext_authz / Authorino injected headers before forwarding upstream (prevent key leakage)
	removeHeaders := append(anthropic.HeadersToRemove(), r.CredentialResolver.HeadersToStrip()...)

	// Return response with body and header mutations - let Envoy route to Anthropic
	// ClearRouteCache forces Envoy to re-evaluate routing after we set x-selected-model header
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					Status:          ext_proc.CommonResponse_CONTINUE,
					ClearRouteCache: true,
					HeaderMutation: &ext_proc.HeaderMutation{
						SetHeaders:    setHeaders,
						RemoveHeaders: removeHeaders,
					},
					BodyMutation: &ext_proc.BodyMutation{
						Mutation: &ext_proc.BodyMutation_Body{
							Body: anthropicBody,
						},
					},
				},
			},
		},
	}, nil
}

// handleAutoModelRouting handles routing for auto model selection
func (r *OpenAIRouter) handleAutoModelRouting(openAIRequest *openai.ChatCompletionNewParams, originalModel string, decisionName string, reasoningDecision entropy.ReasoningDecision, selectedModel string, ctx *RequestContext, response *ext_proc.ProcessingResponse) (*ext_proc.ProcessingResponse, error) {
	logging.Infof("Using Auto Model Selection (model=%s), decision=%s, selected=%s",
		originalModel, decisionName, selectedModel)

	matchedModel := selectedModel

	if matchedModel == originalModel || matchedModel == "" {
		// No model change needed
		ctx.RequestModel = originalModel
		return response, nil
	}

	// Record routing decision with tracing
	r.recordRoutingDecision(ctx, decisionName, originalModel, matchedModel, reasoningDecision)

	// Track VSR decision information
	// categoryName is already set in ctx.VSRSelectedCategory by performDecisionEvaluation
	r.trackVSRDecision(ctx, ctx.VSRSelectedCategory, decisionName, matchedModel, reasoningDecision.UseReasoning)

	// Track model routing metrics
	metrics.RecordModelRouting(originalModel, matchedModel)

	// Select endpoint for the matched model
	selectedEndpoint, selectedEndpointName, endpointErr := r.selectEndpointForModel(ctx, matchedModel)
	if endpointErr != nil {
		return nil, fmt.Errorf("auto routing: %w", endpointErr)
	}

	// Resolve model name alias to real model name for the selected endpoint
	// e.g., "qwen14b-rack1" -> "Qwen/Qwen2.5-14B-Instruct"
	upstreamModel := r.resolveModelNameForEndpoint(matchedModel, selectedEndpointName)

	// Modify request body with resolved model name, reasoning mode, and system prompt
	modifiedBody, err := r.modifyRequestBodyForAutoRouting(openAIRequest, upstreamModel, decisionName, reasoningDecision.UseReasoning, ctx)
	if err != nil {
		return nil, err
	}

	// Create response with mutations (use original alias for headers/tracing, upstream model in body)
	response = r.createRoutingResponse(matchedModel, selectedEndpoint, selectedEndpointName, modifiedBody, ctx)

	// Log routing decision
	r.logRoutingDecision(ctx, "auto_routing", originalModel, matchedModel, decisionName, reasoningDecision.UseReasoning)

	// Handle route cache clearing
	if r.shouldClearRouteCache() {
		r.setClearRouteCache(response)
	}

	// Save the actual model for token tracking
	ctx.RequestModel = matchedModel

	// Capture router replay information if enabled
	r.startRouterReplay(ctx, originalModel, matchedModel, decisionName)

	// Handle tool selection
	r.handleToolSelectionForRequest(openAIRequest, response, ctx)

	// Record routing latency
	r.recordRoutingLatency(ctx)

	return response, nil
}

// handleSpecifiedModelRouting handles routing for explicitly specified models
func (r *OpenAIRouter) handleSpecifiedModelRouting(openAIRequest *openai.ChatCompletionNewParams, originalModel string, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	logging.Infof("Using specified model: %s", originalModel)

	// Track VSR decision information for non-auto models
	ctx.VSRSelectedModel = originalModel
	ctx.VSRReasoningMode = "off" // Non-auto models don't use reasoning mode by default
	// PII policy check already done in performPIIDetection
	// Memory injection already happened in handleMemoryRetrieval (before routing diverged)

	// Select endpoint for the specified model
	selectedEndpoint, selectedEndpointName, endpointErr := r.selectEndpointForModel(ctx, originalModel)
	if endpointErr != nil {
		return nil, fmt.Errorf("specified model routing: %w", endpointErr)
	}

	// Resolve model name alias to real model name for the selected endpoint
	upstreamModel := r.resolveModelNameForEndpoint(originalModel, selectedEndpointName)

	// Create response with headers (and body mutation if model name changed)
	response := r.createSpecifiedModelResponse(originalModel, upstreamModel, selectedEndpoint, selectedEndpointName, ctx)

	// Handle route cache clearing
	if r.shouldClearRouteCache() {
		r.setClearRouteCache(response)
	}

	// Log routing decision
	r.logRoutingDecision(ctx, "model_specified", originalModel, originalModel, "", false)

	// Save the actual model for token tracking
	ctx.RequestModel = originalModel

	// Handle tool selection
	r.handleToolSelectionForRequest(openAIRequest, response, ctx)

	// Record routing latency
	r.recordRoutingLatency(ctx)

	return response, nil
}

// selectEndpointForModel selects the best endpoint for the given model.
// Returns the endpoint address:port, the endpoint name, and any error.
// Backend selection is now part of the model layer (upstream request span)
func (r *OpenAIRouter) selectEndpointForModel(ctx *RequestContext, model string) (string, string, error) {
	endpointAddress, endpointName, endpointFound, err := r.Config.SelectBestEndpointWithDetailsForModel(model)
	if err != nil {
		return "", "", fmt.Errorf("endpoint resolution for model %q: %w", model, err)
	}
	if endpointFound {
		logging.Infof("Selected endpoint address: %s (name: %s) for model: %s", endpointAddress, endpointName, model)
	}

	// Store the selected endpoint in context (for routing/logging purposes)
	ctx.SelectedEndpoint = endpointAddress

	// Increment active request count for queue depth estimation (model-level)
	metrics.IncrementModelActiveRequests(model)

	return endpointAddress, endpointName, nil
}

// resolveModelNameForEndpoint resolves the model name alias to the real model name
// that the backend endpoint expects, using external_model_ids configuration.
// For example, "qwen14b-rack1" -> "Qwen/Qwen2.5-14B-Instruct" for a vllm endpoint.
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

// modifyRequestBodyForAutoRouting modifies the request body for auto routing
func (r *OpenAIRouter) modifyRequestBodyForAutoRouting(openAIRequest *openai.ChatCompletionNewParams, matchedModel string, decisionName string, useReasoning bool, ctx *RequestContext) ([]byte, error) {
	// Modify the model in the request
	openAIRequest.Model = matchedModel

	// Serialize the modified request
	modifiedBody, err := serializeOpenAIRequestWithStream(openAIRequest, ctx.ExpectStreamingResponse)
	if err != nil {
		logging.Errorf("Error serializing modified request: %v", err)
		metrics.RecordRequestError(matchedModel, "serialization_error")
		return nil, status.Errorf(codes.Internal, "error serializing modified request: %v", err)
	}

	// Only apply decision-specific modifications if a decision was matched
	if decisionName != "" {
		// Set reasoning mode
		modifiedBody, err = r.setReasoningModeToRequestBody(modifiedBody, useReasoning, decisionName)
		if err != nil {
			logging.Errorf("Error setting reasoning mode %v to request: %v", useReasoning, err)
			metrics.RecordRequestError(matchedModel, "serialization_error")
			return nil, status.Errorf(codes.Internal, "error setting reasoning mode: %v", err)
		}

		// Add decision-specific system prompt if configured
		modifiedBody, err = r.addSystemPromptIfConfigured(modifiedBody, decisionName, matchedModel, ctx)
		if err != nil {
			return nil, err
		}
	}

	// Inject memory context AFTER system prompt (so it appends, not gets overwritten)
	// NOTE: Memory injection must happen regardless of whether a decision was matched
	if ctx.MemoryContext != "" {
		modifiedBody, err = injectSystemMessage(modifiedBody, ctx.MemoryContext)
		if err != nil {
			logging.Warnf("Memory: Failed to inject memory context: %v", err)
			// Graceful degradation: continue without memory injection
		}
	}

	return modifiedBody, nil
}

// startUpstreamSpanAndInjectHeaders starts an upstream request span and returns trace context headers.
// The span will be ended when response headers arrive in handleResponseHeaders.
func (r *OpenAIRouter) startUpstreamSpanAndInjectHeaders(model string, endpoint string, ctx *RequestContext) []*core.HeaderValueOption {
	var traceContextHeaders []*core.HeaderValueOption

	// Start upstream request span (will be ended when response headers arrive)
	spanCtx, upstreamSpan := tracing.StartSpan(ctx.TraceContext, tracing.SpanUpstreamRequest,
		trace.WithSpanKind(trace.SpanKindClient))
	ctx.TraceContext = spanCtx
	ctx.UpstreamSpan = upstreamSpan

	// Set span attributes for upstream request
	tracing.SetSpanAttributes(upstreamSpan,
		attribute.String(tracing.AttrModelName, model),
		attribute.String(tracing.AttrEndpointAddress, endpoint))

	// Inject W3C trace context headers for distributed tracing to vLLM
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

// resolveProviderAuth determines the LLMProvider, auth header name, and auth prefix
// from a provider profile.
//
// When profile is nil (legacy endpoint without provider_profile), the caller MUST
// have already determined LLMProvider from the endpoint's Type field or from the
// pre-existing openai-compatible convention.
//
// When profile is non-nil, all three values are derived from the profile's type.
// Returns error if the profile type is unrecognised.
func resolveProviderAuth(profile *config.ProviderProfile) (authz.LLMProvider, string, string, error) {
	if profile == nil {
		// Legacy endpoint (no provider_profile set).
		// Use ProviderOpenAI — this is the only provider type that existed
		// before provider_profiles were introduced. The auth header format
		// comes from the same openai convention that was previously hardcoded
		// in this function's callers.
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

// createRoutingResponse creates a routing response with mutations.
// endpointName is the name of the selected VLLMEndpoint (used to look up provider profile).
func (r *OpenAIRouter) createRoutingResponse(model string, endpoint string, endpointName string, modifiedBody []byte, ctx *RequestContext) *ext_proc.ProcessingResponse {
	bodyMutation := &ext_proc.BodyMutation{
		Mutation: &ext_proc.BodyMutation_Body{
			Body: modifiedBody,
		},
	}

	setHeaders := []*core.HeaderValueOption{}
	removeHeaders := []string{"content-length"} // Always remove old content-length when body is modified

	// Add new content-length header for the modified body
	if len(modifiedBody) > 0 {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      "content-length",
				RawValue: []byte(fmt.Sprintf("%d", len(modifiedBody))),
			},
		})
	}

	logging.Infof("createRoutingResponse: modifiedBody length=%d, model=%s", len(modifiedBody), model)

	// Start upstream span and inject trace context headers
	traceContextHeaders := r.startUpstreamSpanAndInjectHeaders(model, endpoint, ctx)
	setHeaders = append(setHeaders, traceContextHeaders...)

	// Resolve provider type and auth format from endpoint's provider profile
	profile, profileErr := r.Config.GetProviderProfileForEndpoint(endpointName)
	if profileErr != nil {
		return r.createErrorResponse(500, fmt.Sprintf("Provider profile resolution failed for endpoint %s: %v", endpointName, profileErr))
	}
	llmProvider, authHeader, authPrefix, authErr := resolveProviderAuth(profile)
	if authErr != nil {
		return r.createErrorResponse(500, fmt.Sprintf("Provider auth resolution failed for endpoint %s: %v", endpointName, authErr))
	}

	// Resolve API key via credential chain (ext_authz headers → static config)
	accessKey, credErr := r.CredentialResolver.KeyForProvider(llmProvider, model, ctx.Headers)
	if credErr != nil {
		return r.createErrorResponse(401, fmt.Sprintf("Credential resolution failed for model %s: %v", model, credErr))
	}
	if accessKey != "" {
		value := accessKey
		if authPrefix != "" {
			value = authPrefix + " " + accessKey
		}
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      authHeader,
				RawValue: []byte(value),
			},
		})
		logging.Infof("Added %s header for model %s (provider=%s)", authHeader, model, llmProvider)
	} else {
		// fail_open=true path: no key but allowed through
		logging.Warnf("No API key for %s model %q (fail_open=true) — forwarding without auth header", llmProvider, model)
	}

	// Add explicit extra headers from provider profile config
	if profile != nil {
		for k, v := range profile.ExtraHeaders {
			setHeaders = append(setHeaders, &core.HeaderValueOption{
				Header: &core.HeaderValue{Key: k, RawValue: []byte(v)},
			})
		}
	}

	// Strip ext_authz injected headers before forwarding upstream (prevent key leakage)
	removeHeaders = append(removeHeaders, r.CredentialResolver.HeadersToStrip()...)

	// Add standard routing headers
	if endpoint != "" {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.GatewayDestinationEndpoint,
				RawValue: []byte(endpoint),
			},
		})
	}
	if model != "" {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.SelectedModel,
				RawValue: []byte(model),
			},
		})
	}

	// Set :path from provider profile, or use Response API override
	if ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.IsResponseAPIRequest {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      ":path",
				RawValue: []byte("/v1/chat/completions"),
			},
		})
		logging.Infof("Response API: Rewriting path to /v1/chat/completions")
	} else if profile != nil {
		chatPath, pathErr := profile.ResolveChatPath()
		if pathErr != nil {
			return r.createErrorResponse(500, fmt.Sprintf("Chat path resolution failed for endpoint %s: %v", endpointName, pathErr))
		}
		if chatPath != "" {
			setHeaders = append(setHeaders, &core.HeaderValueOption{
				Header: &core.HeaderValue{
					Key:      ":path",
					RawValue: []byte(chatPath),
				},
			})
			logging.Infof("Provider profile: Rewriting path to %s", chatPath)
		}
	}

	// Apply header mutations from decision's header_mutation plugin
	if ctx.VSRSelectedDecision != nil {
		pluginSetHeaders, pluginRemoveHeaders := r.buildHeaderMutations(ctx.VSRSelectedDecision)
		if len(pluginSetHeaders) > 0 {
			setHeaders = append(setHeaders, pluginSetHeaders...)
			logging.Infof("Applied %d header mutations from decision %s", len(pluginSetHeaders), ctx.VSRSelectedDecision.Name)
		}
		if len(pluginRemoveHeaders) > 0 {
			removeHeaders = append(removeHeaders, pluginRemoveHeaders...)
			logging.Infof("Applied %d header deletions from decision %s", len(pluginRemoveHeaders), ctx.VSRSelectedDecision.Name)
		}
	}

	headerMutation := &ext_proc.HeaderMutation{
		RemoveHeaders: removeHeaders,
		SetHeaders:    setHeaders,
	}

	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					Status:         ext_proc.CommonResponse_CONTINUE,
					HeaderMutation: headerMutation,
					BodyMutation:   bodyMutation,
				},
			},
		},
	}
}

// createSpecifiedModelResponse creates a response for specified model routing.
// model is the internal alias (used for headers/tracing), upstreamModel is the real model
// name that the backend endpoint expects (resolved via external_model_ids).
// endpointName is the name of the selected VLLMEndpoint (used to look up provider profile).
func (r *OpenAIRouter) createSpecifiedModelResponse(model string, upstreamModel string, endpoint string, endpointName string, ctx *RequestContext) *ext_proc.ProcessingResponse {
	setHeaders := []*core.HeaderValueOption{}
	removeHeaders := []string{}

	// Start upstream span and inject trace context headers
	traceContextHeaders := r.startUpstreamSpanAndInjectHeaders(model, endpoint, ctx)
	setHeaders = append(setHeaders, traceContextHeaders...)

	// Resolve provider type and auth format from endpoint's provider profile
	profile, profileErr := r.Config.GetProviderProfileForEndpoint(endpointName)
	if profileErr != nil {
		return r.createErrorResponse(500, fmt.Sprintf("Provider profile resolution failed for endpoint %s: %v", endpointName, profileErr))
	}
	llmProvider, authHeader, authPrefix, authErr := resolveProviderAuth(profile)
	if authErr != nil {
		return r.createErrorResponse(500, fmt.Sprintf("Provider auth resolution failed for endpoint %s: %v", endpointName, authErr))
	}

	// Resolve API key via credential chain (ext_authz headers → static config)
	accessKey, credErr := r.CredentialResolver.KeyForProvider(llmProvider, model, ctx.Headers)
	if credErr != nil {
		return r.createErrorResponse(401, fmt.Sprintf("Credential resolution failed for model %s: %v", model, credErr))
	}
	if accessKey != "" {
		value := accessKey
		if authPrefix != "" {
			value = authPrefix + " " + accessKey
		}
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      authHeader,
				RawValue: []byte(value),
			},
		})
		logging.Infof("Added %s header for model %s (provider=%s)", authHeader, model, llmProvider)
	} else {
		// fail_open=true path: no key but allowed through
		logging.Warnf("No API key for %s model %q (fail_open=true) — forwarding without auth header", llmProvider, model)
	}

	// Add explicit extra headers from provider profile config
	if profile != nil {
		for k, v := range profile.ExtraHeaders {
			setHeaders = append(setHeaders, &core.HeaderValueOption{
				Header: &core.HeaderValue{Key: k, RawValue: []byte(v)},
			})
		}
	}

	// Strip ext_authz injected headers before forwarding upstream (prevent key leakage)
	removeHeaders = append(removeHeaders, r.CredentialResolver.HeadersToStrip()...)

	if endpoint != "" {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.GatewayDestinationEndpoint,
				RawValue: []byte(endpoint),
			},
		})
	}
	// Set x-selected-model header for non-auto models
	setHeaders = append(setHeaders, &core.HeaderValueOption{
		Header: &core.HeaderValue{
			Key:      headers.SelectedModel,
			RawValue: []byte(model),
		},
	})

	// Determine if we need to mutate the request body
	var bodyMutation *ext_proc.BodyMutation
	needsBodyMutation := false

	// Model name rewriting: if the upstream model name differs from the alias,
	// we need to rewrite the "model" field in the request body
	if upstreamModel != model {
		needsBodyMutation = true
		logging.Infof("Model name rewriting: %s -> %s in request body", model, upstreamModel)
	}

	// Set :path from provider profile, or use Response API override
	if ctx != nil && ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.IsResponseAPIRequest {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      ":path",
				RawValue: []byte("/v1/chat/completions"),
			},
		})
		needsBodyMutation = true
		logging.Infof("Response API: Rewriting path to /v1/chat/completions (specified model)")
	} else if profile != nil {
		chatPath, pathErr := profile.ResolveChatPath()
		if pathErr != nil {
			return r.createErrorResponse(500, fmt.Sprintf("Chat path resolution failed for endpoint %s: %v", endpointName, pathErr))
		}
		if chatPath != "" {
			setHeaders = append(setHeaders, &core.HeaderValueOption{
				Header: &core.HeaderValue{
					Key:      ":path",
					RawValue: []byte(chatPath),
				},
			})
			logging.Infof("Provider profile: Rewriting path to %s (specified model)", chatPath)
		}
	}

	if needsBodyMutation {
		removeHeaders = append(removeHeaders, "content-length")

		// Start with the original request body or Response API translated body
		var bodyBytes []byte
		if ctx != nil && ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.IsResponseAPIRequest && len(ctx.ResponseAPICtx.TranslatedBody) > 0 {
			bodyBytes = ctx.ResponseAPICtx.TranslatedBody
		} else if ctx != nil && len(ctx.OriginalRequestBody) > 0 {
			bodyBytes = ctx.OriginalRequestBody
		}

		// Rewrite model name in body if needed
		if upstreamModel != model && len(bodyBytes) > 0 {
			rewritten, err := rewriteModelInBody(bodyBytes, upstreamModel)
			if err != nil {
				logging.Warnf("Failed to rewrite model in body: %v, sending original body", err)
			} else {
				bodyBytes = rewritten
			}
		}

		if len(bodyBytes) > 0 {
			// Update content-length for the modified body
			setHeaders = append(setHeaders, &core.HeaderValueOption{
				Header: &core.HeaderValue{
					Key:      "content-length",
					RawValue: []byte(fmt.Sprintf("%d", len(bodyBytes))),
				},
			})
			bodyMutation = &ext_proc.BodyMutation{
				Mutation: &ext_proc.BodyMutation_Body{
					Body: bodyBytes,
				},
			}
		}
	}

	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					Status: ext_proc.CommonResponse_CONTINUE,
					HeaderMutation: &ext_proc.HeaderMutation{
						SetHeaders:    setHeaders,
						RemoveHeaders: removeHeaders,
					},
					BodyMutation: bodyMutation,
				},
			},
		},
	}
}

// rewriteModelInBody rewrites the "model" field in a JSON request body.
// Uses a generic map approach to preserve all other fields.
func rewriteModelInBody(body []byte, newModel string) ([]byte, error) {
	var requestMap map[string]json.RawMessage
	if err := json.Unmarshal(body, &requestMap); err != nil {
		return nil, fmt.Errorf("failed to unmarshal request body: %w", err)
	}

	modelJSON, err := json.Marshal(newModel)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal new model name: %w", err)
	}
	requestMap["model"] = json.RawMessage(modelJSON)

	rewritten, err := json.Marshal(requestMap)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal rewritten request: %w", err)
	}

	return rewritten, nil
}

// getModelAccessKey retrieves the access_key for a given model from the config
// Returns empty string if model not found or access_key not configured
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

// getModelParams returns a map of model names to their ModelParams
// This is used by looper to access model-specific configuration like access_key and param_size
func (r *OpenAIRouter) getModelParams() map[string]config.ModelParams {
	if r.Config == nil || r.Config.ModelConfig == nil {
		return nil
	}
	return r.Config.ModelConfig
}

// handleMemoryRetrieval retrieves relevant memories and injects them into the request.
// Per-decision plugin config takes precedence over global config.
func (r *OpenAIRouter) handleMemoryRetrieval(
	ctx *RequestContext,
	userContent string,
	requestBody []byte,
	openAIRequest *openai.ChatCompletionNewParams,
) ([]byte, error) {
	var memoryPluginConfig *config.MemoryPluginConfig
	if ctx.VSRSelectedDecision != nil {
		memoryPluginConfig = ctx.VSRSelectedDecision.GetMemoryConfig()
	}

	memoryEnabled := r.Config.Memory.Enabled
	if memoryPluginConfig != nil {
		memoryEnabled = memoryPluginConfig.Enabled
		if !memoryEnabled {
			logging.Debugf("Memory: Disabled by per-decision plugin config for decision '%s'", ctx.VSRSelectedDecisionName)
			return requestBody, nil
		}
	} else if !memoryEnabled {
		logging.Debugf("Memory: Disabled in global config, skipping retrieval")
		return requestBody, nil
	}

	// Get memory store from router
	store := r.getMemoryStore()
	if store == nil || !store.IsEnabled() {
		logging.Debugf("Memory: Store not available or disabled, skipping retrieval")
		return requestBody, nil
	}

	// TODO: Remove demo logs after POC
	logging.Infof("╔══════════════════════════════════════════════════════════════════╗")
	logging.Infof("║                    MEMORY RETRIEVAL FLOW                         ║")
	logging.Infof("╠══════════════════════════════════════════════════════════════════╣")
	logging.Infof("║ User Query: %s", truncateForLog(userContent, 50))
	if memoryPluginConfig != nil {
		logging.Infof("║ Config Source: per-decision plugin (decision: %s)", ctx.VSRSelectedDecisionName)
	} else {
		logging.Infof("║ Config Source: global config")
	}

	// Step 1: Memory decision - should we search?
	if !ShouldSearchMemory(ctx, userContent) {
		logging.Infof("║ Decision: ❌ SKIP (query type not suitable for memory search)")
		logging.Infof("╚══════════════════════════════════════════════════════════════════╝")
		return requestBody, nil
	}
	logging.Infof("║ Decision: ✅ SEARCH (query may benefit from memory)")

	// Step 2: Extract conversation history from request body
	// Use the existing ExtractConversationHistory function which works with raw JSON
	var messagesJSON []byte
	if openAIRequest.Messages != nil {
		messagesJSON, _ = json.Marshal(openAIRequest.Messages)
	}

	history, err := ExtractConversationHistory(messagesJSON)
	if err != nil {
		logging.Warnf("Memory: Failed to extract conversation history: %v", err)
		// Continue with empty history
		history = []ConversationMessage{}
	}

	// Step 3: Build search query (with context/rewriting if memory_rewrite external model is configured)
	searchQuery, err := BuildSearchQuery(
		ctx.TraceContext,
		history,
		userContent,
		r.Config,
	)
	if err != nil {
		logging.Warnf("Memory: Query rewriting failed, using original query: %v", err)
		searchQuery = userContent
	}

	// Step 4: Get user ID from Response API context or request
	userID := r.getUserIDFromContext(ctx)
	if userID == "" {
		logging.Infof("║ User ID: ❌ NOT FOUND (skipping memory search)")
		logging.Infof("╚══════════════════════════════════════════════════════════════════╝")
		return requestBody, nil
	}
	logging.Infof("║ User ID: %s", userID)

	// Step 5: Search Milvus (per-decision settings override global defaults)
	retrieveLimit := r.Config.Memory.DefaultRetrievalLimit
	retrieveThreshold := r.Config.Memory.DefaultSimilarityThreshold

	if memoryPluginConfig != nil {
		if memoryPluginConfig.RetrievalLimit != nil {
			retrieveLimit = *memoryPluginConfig.RetrievalLimit
		}
		if memoryPluginConfig.SimilarityThreshold != nil {
			retrieveThreshold = *memoryPluginConfig.SimilarityThreshold
		}
	}

	retrieveOpts := memory.RetrieveOptions{
		Query:     searchQuery,
		UserID:    userID,
		Limit:     retrieveLimit,
		Threshold: retrieveThreshold,
	}

	// Apply defaults if not configured
	if retrieveOpts.Limit <= 0 {
		retrieveOpts.Limit = 5
	}
	if retrieveOpts.Threshold <= 0 {
		retrieveOpts.Threshold = 0.6
	}

	memories, err := store.Retrieve(ctx.TraceContext, retrieveOpts)
	if err != nil {
		return requestBody, fmt.Errorf("memory retrieval failed: %w", err)
	}

	if len(memories) == 0 {
		logging.Infof("║ Search Result: 📭 No memories found above threshold")
		logging.Infof("╚══════════════════════════════════════════════════════════════════╝")
		return requestBody, nil
	}

	logging.Infof("║ Search Result: 📬 Found %d memories!", len(memories))
	for i, mem := range memories {
		if mem.Memory != nil {
			logging.Infof("║   %d. [%s] (score: %.2f) %s", i+1, mem.Memory.Type, mem.Score, mem.Memory.Content) // Full content for demo
		}
	}
	logging.Infof("╚══════════════════════════════════════════════════════════════════╝")

	// Step 6: Format memory context and inject into request body
	ctx.MemoryContext = FormatMemoriesAsContext(memories)

	// Step 7: Inject memory into request body as system message
	// This happens here (before routing diverges) so it works for BOTH auto and specified models
	if ctx.MemoryContext != "" {
		injectedBody, err := injectSystemMessage(requestBody, ctx.MemoryContext)
		if err != nil {
			logging.Warnf("Memory: Failed to inject memory context: %v", err)
			// Graceful degradation: continue without injection
			return requestBody, nil
		}
		logging.Infof("Memory: Injected %d memories into request", len(memories))
		return injectedBody, nil
	}

	return requestBody, nil
}

// getMemoryStore returns the memory store instance.
func (r *OpenAIRouter) getMemoryStore() *memory.MilvusStore {
	// Return the actual memory store from router
	return r.MemoryStore
}

// getUserIDFromContext extracts user ID from the trusted auth header (x-authz-user-id).
// Falls back to untrusted metadata["user_id"] only for development/testing without an auth layer.
func (r *OpenAIRouter) getUserIDFromContext(ctx *RequestContext) string {
	return extractUserID(ctx)
}

// buildRateLimitContext constructs a ratelimit.Context from the request context.
func (r *OpenAIRouter) buildRateLimitContext(ctx *RequestContext, selectedModel string) ratelimit.Context {
	userID := ctx.Headers[r.Config.Authz.Identity.GetUserIDHeader()]
	groupsStr := ctx.Headers[r.Config.Authz.Identity.GetUserGroupsHeader()]
	var groups []string
	if groupsStr != "" {
		for _, g := range strings.Split(groupsStr, ",") {
			g = strings.TrimSpace(g)
			if g != "" {
				groups = append(groups, g)
			}
		}
	}

	return ratelimit.Context{
		UserID:     userID,
		Groups:     groups,
		Model:      selectedModel,
		Headers:    ctx.Headers,
		TokenCount: ctx.VSRContextTokenCount,
	}
}

// createRateLimitResponse builds a 429 Too Many Requests response with
// standard rate limit headers.
func (r *OpenAIRouter) createRateLimitResponse(decision *ratelimit.Decision) *ext_proc.ProcessingResponse {
	retryAfterSec := "60"
	if decision != nil && decision.RetryAfter > 0 {
		retryAfterSec = fmt.Sprintf("%d", int(decision.RetryAfter.Seconds()))
	}

	body := []byte(fmt.Sprintf(`{"error":{"message":"Rate limit exceeded. Retry after %s seconds.","type":"rate_limit_error","code":429}}`, retryAfterSec))

	respHeaders := []*core.HeaderValueOption{
		{Header: &core.HeaderValue{Key: "content-type", RawValue: []byte("application/json")}},
		{Header: &core.HeaderValue{Key: "retry-after", RawValue: []byte(retryAfterSec)}},
	}

	if decision != nil {
		respHeaders = append(respHeaders,
			&core.HeaderValueOption{Header: &core.HeaderValue{
				Key: "x-ratelimit-limit", RawValue: []byte(fmt.Sprintf("%d", decision.Limit)),
			}},
			&core.HeaderValueOption{Header: &core.HeaderValue{
				Key: "x-ratelimit-remaining", RawValue: []byte(fmt.Sprintf("%d", decision.Remaining)),
			}},
		)
		if !decision.ResetAt.IsZero() {
			respHeaders = append(respHeaders, &core.HeaderValueOption{Header: &core.HeaderValue{
				Key: "x-ratelimit-reset", RawValue: []byte(fmt.Sprintf("%d", decision.ResetAt.Unix())),
			}})
		}
	}

	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &ext_proc.ImmediateResponse{
				Status: &typev3.HttpStatus{Code: typev3.StatusCode_TooManyRequests},
				Headers: &ext_proc.HeaderMutation{
					SetHeaders: respHeaders,
				},
				Body: body,
			},
		},
	}
}
