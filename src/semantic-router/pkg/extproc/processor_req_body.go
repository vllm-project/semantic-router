package extproc

import (
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"
	"go.opentelemetry.io/otel/attribute"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

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

	// Extract stream parameter from original request and update ExpectStreamingResponse if needed
	hasStreamParam := extractStreamParam(ctx.OriginalRequestBody)
	if hasStreamParam {
		logging.Infof("Original request contains stream parameter: true")
		ctx.ExpectStreamingResponse = true // Set this if stream param is found
	}

	// Parse the OpenAI request using SDK types
	openAIRequest, err := parseOpenAIRequest(ctx.OriginalRequestBody)
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

	// Set model on span
	if ctx.TraceContext != nil {
		_, span := tracing.StartSpan(ctx.TraceContext, "parse_request")
		tracing.SetSpanAttributes(span,
			attribute.String(tracing.AttrOriginalModel, originalModel))
		span.End()
	}

	// Record the initial request to this model (count all requests)
	metrics.RecordModelRequest(originalModel)
	// Also set the model on context early so error metrics can label it
	if ctx.RequestModel == "" {
		ctx.RequestModel = originalModel
	}

	// Get content from messages
	userContent, nonUserMessages := extractUserAndNonUserContent(openAIRequest)

	// Perform decision evaluation and model selection once at the beginning
	// Use decision-based routing if decisions are configured, otherwise fall back to category-based
	decisionName, classificationConfidence, reasoningDecision, selectedModel := r.performDecisionEvaluationAndModelSelection(originalModel, userContent, nonUserMessages, ctx)

	// Perform security checks with decision-specific settings
	if response, shouldReturn := r.performSecurityChecks(ctx, userContent, nonUserMessages, decisionName); shouldReturn {
		return response, nil
	}

	// Perform PII detection and policy check (if PII policy is enabled for the decision)
	piiResponse := r.performPIIDetection(ctx, userContent, nonUserMessages, decisionName)
	if piiResponse != nil {
		// PII policy violation - return error response
		return piiResponse, nil
	}

	// Handle caching with decision-specific settings
	logging.Infof("About to call handleCaching - decisionName=%s, cacheEnabled=%v", decisionName, r.Config.SemanticCache.Enabled)
	if response, shouldReturn := r.handleCaching(ctx, decisionName); shouldReturn {
		logging.Infof("handleCaching returned a response, returning immediately")
		return response, nil
	}
	logging.Infof("handleCaching returned no cached response, continuing to model routing")

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

	if isAutoModel && selectedModel != "" {
		return r.handleAutoModelRouting(openAIRequest, originalModel, decisionName, reasoningDecision, selectedModel, ctx, response)
	} else if !isAutoModel {
		return r.handleSpecifiedModelRouting(openAIRequest, originalModel, ctx)
	}

	// No routing needed, return default response
	ctx.RequestModel = originalModel
	return response, nil
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
	// categoryName is already set in ctx.VSRSelectedCategory by performDecisionEvaluationAndModelSelection
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
	response := r.createSpecifiedModelResponse(originalModel, selectedEndpoint)

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
func (r *OpenAIRouter) selectEndpointForModel(ctx *RequestContext, model string) string {
	backendCtx, backendSpan := tracing.StartSpan(ctx.TraceContext, tracing.SpanBackendSelection)

	endpointAddress, endpointFound := r.Config.SelectBestEndpointAddressForModel(model)
	if endpointFound {
		logging.Infof("Selected endpoint address: %s for model: %s", endpointAddress, model)

		endpoints := r.Config.GetEndpointsForModel(model)
		if len(endpoints) > 0 {
			tracing.SetSpanAttributes(backendSpan,
				attribute.String(tracing.AttrEndpointName, endpoints[0].Name),
				attribute.String(tracing.AttrEndpointAddress, endpointAddress))
		}
	} else {
		logging.Warnf("No endpoint found for model %s, using fallback", model)
	}

	backendSpan.End()
	ctx.TraceContext = backendCtx

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

// createRoutingResponse creates a routing response with mutations
func (r *OpenAIRouter) createRoutingResponse(model string, endpoint string, modifiedBody []byte, ctx *RequestContext) *ext_proc.ProcessingResponse {
	bodyMutation := &ext_proc.BodyMutation{
		Mutation: &ext_proc.BodyMutation_Body{
			Body: modifiedBody,
		},
	}

	setHeaders := []*core.HeaderValueOption{}
	removeHeaders := []string{"content-length"}

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
func (r *OpenAIRouter) createSpecifiedModelResponse(model string, endpoint string) *ext_proc.ProcessingResponse {
	setHeaders := []*core.HeaderValueOption{}
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

	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					Status: ext_proc.CommonResponse_CONTINUE,
					HeaderMutation: &ext_proc.HeaderMutation{
						SetHeaders: setHeaders,
					},
				},
			},
		},
	}
}
