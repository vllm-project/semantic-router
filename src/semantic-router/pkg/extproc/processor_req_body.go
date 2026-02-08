package extproc

import (
	"encoding/json"
	"fmt"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/anthropic"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
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
	decisionName, _, reasoningDecision, selectedModel := r.performDecisionEvaluation(originalModel, userContent, nonUserMessages, ctx)

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

	// Get API key for the model
	accessKey := r.Config.GetModelAccessKey(targetModel)
	if accessKey == "" {
		logging.Errorf("No access_key configured for Anthropic model: %s", targetModel)
		return r.createErrorResponse(500, fmt.Sprintf("No API key configured for model: %s", targetModel)), nil
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
						RemoveHeaders: anthropic.HeadersToRemove(),
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
	selectedEndpoint := r.selectEndpointForModel(ctx, matchedModel)

	// Modify request body with new model, reasoning mode, and system prompt
	modifiedBody, err := r.modifyRequestBodyForAutoRouting(openAIRequest, matchedModel, decisionName, reasoningDecision.UseReasoning, ctx)
	if err != nil {
		return nil, err
	}

	// Create response with mutations
	response = r.createRoutingResponse(matchedModel, selectedEndpoint, modifiedBody, ctx)

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
	selectedEndpoint := r.selectEndpointForModel(ctx, originalModel)

	// Create response with headers
	response := r.createSpecifiedModelResponse(originalModel, selectedEndpoint, ctx)

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

// selectEndpointForModel selects the best endpoint for the given model
// Backend selection is now part of the model layer (upstream request span)
func (r *OpenAIRouter) selectEndpointForModel(ctx *RequestContext, model string) string {
	endpointAddress, endpointFound := r.Config.SelectBestEndpointAddressForModel(model)
	if endpointFound {
		logging.Infof("Selected endpoint address: %s for model: %s", endpointAddress, model)
	}

	// Store the selected endpoint in context (for routing/logging purposes)
	ctx.SelectedEndpoint = endpointAddress

	// Increment active request count for queue depth estimation (model-level)
	metrics.IncrementModelActiveRequests(model)

	return endpointAddress
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

// createRoutingResponse creates a routing response with mutations
func (r *OpenAIRouter) createRoutingResponse(model string, endpoint string, modifiedBody []byte, ctx *RequestContext) *ext_proc.ProcessingResponse {
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

	// Add Authorization header if model has access_key configured
	if accessKey := r.getModelAccessKey(model); accessKey != "" {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      "Authorization",
				RawValue: []byte(fmt.Sprintf("Bearer %s", accessKey)),
			},
		})
		logging.Infof("Added Authorization header for model %s", model)
	}

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

	// For Response API requests, modify :path to /v1/chat/completions
	if ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.IsResponseAPIRequest {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      ":path",
				RawValue: []byte("/v1/chat/completions"),
			},
		})
		logging.Infof("Response API: Rewriting path to /v1/chat/completions")
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

// createSpecifiedModelResponse creates a response for specified model routing
func (r *OpenAIRouter) createSpecifiedModelResponse(model string, endpoint string, ctx *RequestContext) *ext_proc.ProcessingResponse {
	setHeaders := []*core.HeaderValueOption{}
	removeHeaders := []string{}

	// Start upstream span and inject trace context headers
	traceContextHeaders := r.startUpstreamSpanAndInjectHeaders(model, endpoint, ctx)
	setHeaders = append(setHeaders, traceContextHeaders...)

	// Add Authorization header if model has access_key configured
	if accessKey := r.getModelAccessKey(model); accessKey != "" {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      "Authorization",
				RawValue: []byte(fmt.Sprintf("Bearer %s", accessKey)),
			},
		})
		logging.Infof("Added Authorization header for model %s", model)
	}

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

	// For Response API requests, modify :path to /v1/chat/completions and use translated body
	var bodyMutation *ext_proc.BodyMutation
	if ctx != nil && ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.IsResponseAPIRequest {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      ":path",
				RawValue: []byte("/v1/chat/completions"),
			},
		})
		removeHeaders = append(removeHeaders, "content-length")

		// Use the translated body from Response API context
		if len(ctx.ResponseAPICtx.TranslatedBody) > 0 {
			bodyMutation = &ext_proc.BodyMutation{
				Mutation: &ext_proc.BodyMutation_Body{
					Body: ctx.ResponseAPICtx.TranslatedBody,
				},
			}
		}
		logging.Infof("Response API: Rewriting path to /v1/chat/completions (specified model)")
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

// getUserIDFromContext extracts user ID from Response API context or request.
func (r *OpenAIRouter) getUserIDFromContext(ctx *RequestContext) string {
	// Check Response API context first
	// userID is provided via metadata.user_id (OpenAI API spec-compliant)
	if ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.OriginalRequest != nil {
		if ctx.ResponseAPICtx.OriginalRequest.Metadata != nil {
			if userID, ok := ctx.ResponseAPICtx.OriginalRequest.Metadata["user_id"]; ok {
				return userID
			}
		}
	}

	return ""
}
