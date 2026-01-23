package extproc

import (
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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// handleRequestBody processes the request body
func (r *OpenAIRouter) handleRequestBody(v *ext_proc.ProcessingRequest_RequestBody, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	logging.Infof("Processing request body: %s", string(v.RequestBody.GetBody()))
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

	// Check if this is a looper internal request - if so, skip all plugin processing
	// and route directly to the specified model (looper already did decision evaluation)
	if r.isLooperRequest(ctx) {
		logging.Infof("[Looper] Internal request detected, skipping plugin processing, routing to model: %s", originalModel)
		ctx.RequestModel = originalModel
		ctx.VSRSelectedModel = originalModel
		return r.handleLooperInternalRequest(originalModel, ctx)
	}

	// Get content from messages
	userContent, nonUserMessages := extractUserAndNonUserContent(openAIRequest)

	// Store user content for later use in hallucination detection
	ctx.UserContent = userContent

	// Perform decision evaluation and model selection once at the beginning
	// Use decision-based routing if decisions are configured, otherwise fall back to category-based
	// This also evaluates fact-check signal as part of the signal evaluation
	decisionName, classificationConfidence, reasoningDecision, selectedModel := r.performDecisionEvaluation(originalModel, userContent, nonUserMessages, ctx)

	// Record the initial request to this model (count all requests)
	metrics.RecordModelRequest(selectedModel)

	// Perform security checks with decision-specific settings
	if response, shouldReturn := r.performJailbreaks(ctx, userContent, nonUserMessages, decisionName); shouldReturn {
		return response, nil
	}

	// Perform PII detection and policy check (if PII policy is enabled for the decision)
	piiResponse := r.performPIIDetection(ctx, userContent, nonUserMessages, decisionName)
	if piiResponse != nil {
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
	if err := r.executeRAGPlugin(ctx, decisionName); err != nil {
		// If RAG fails with on_failure=block, return error response
		return r.createErrorResponse(503, fmt.Sprintf("RAG retrieval failed: %v", err)), nil
	}

	// Handle model selection and routing with pre-computed classification results and selected model
	return r.handleModelRouting(openAIRequest, originalModel, decisionName, classificationConfidence, reasoningDecision, selectedModel, ctx)
}

// handleModelRouting handles model selection and routing logic
// decisionName, classificationConfidence, reasoningDecision, and selectedModel are pre-computed from ProcessRequest
func (r *OpenAIRouter) handleModelRouting(openAIRequest *openai.ChatCompletionNewParams, originalModel string, decisionName string, classificationConfidence float64, reasoningDecision entropy.ReasoningDecision, selectedModel string, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
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
	r.logRoutingDecision(ctx, "auto_routing", originalModel, matchedModel, decisionName, reasoningDecision.UseReasoning, selectedEndpoint)

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

	// Select endpoint for the specified model
	selectedEndpoint := r.selectEndpointForModel(ctx, originalModel)

	// Create response with headers
	response := r.createSpecifiedModelResponse(originalModel, selectedEndpoint, ctx)

	// Handle route cache clearing
	if r.shouldClearRouteCache() {
		r.setClearRouteCache(response)
	}

	// Log routing decision
	r.logRoutingDecision(ctx, "model_specified", originalModel, originalModel, "", false, selectedEndpoint)

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

	if decisionName == "" {
		return modifiedBody, nil
	}
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
