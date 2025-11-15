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

	// Perform classification and model selection once at the beginning
	categoryName, classificationConfidence, reasoningDecision, selectedModel := r.performClassificationAndModelSelection(originalModel, userContent, nonUserMessages)

	// Perform security checks with category-specific settings
	if response, shouldReturn := r.performSecurityChecks(ctx, userContent, nonUserMessages, categoryName); shouldReturn {
		return response, nil
	}

	// Perform PII detection and policy check (if PII policy is enabled for the category)
	// For auto models: this may modify selectedModel if the initially selected model violates PII policy
	// For non-auto models: this checks if the specified model passes PII policy
	isAutoModel := r.Config != nil && r.Config.IsAutoModelName(originalModel)
	modelToCheck := selectedModel
	if !isAutoModel || selectedModel == "" {
		// For non-auto models or when no model was selected, check the original model
		modelToCheck = originalModel
	}

	allowedModel, piiResponse := r.performPIIDetection(ctx, userContent, nonUserMessages, categoryName, modelToCheck, isAutoModel)
	if piiResponse != nil {
		// PII policy violation - return error response
		return piiResponse, nil
	}
	// Use the allowed model (may be different from selectedModel if PII policy required a change)
	if allowedModel != "" {
		selectedModel = allowedModel
	}

	// Handle caching with category-specific settings
	if response, shouldReturn := r.handleCaching(ctx, categoryName); shouldReturn {
		return response, nil
	}

	// Handle model selection and routing with pre-computed classification results and selected model
	return r.handleModelRouting(openAIRequest, originalModel, categoryName, classificationConfidence, reasoningDecision, selectedModel, ctx)
}

// handleModelRouting handles model selection and routing logic
// categoryName, classificationConfidence, reasoningDecision, and selectedModel are pre-computed from ProcessRequest
func (r *OpenAIRouter) handleModelRouting(openAIRequest *openai.ChatCompletionNewParams, originalModel string, categoryName string, classificationConfidence float64, reasoningDecision entropy.ReasoningDecision, selectedModel string, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
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
		return r.handleAutoModelRouting(openAIRequest, originalModel, categoryName, reasoningDecision, selectedModel, ctx, response)
	} else if !isAutoModel {
		return r.handleSpecifiedModelRouting(openAIRequest, originalModel, ctx)
	}

	// No routing needed, return default response
	ctx.RequestModel = originalModel
	return response, nil
}

// handleAutoModelRouting handles routing for auto model selection
func (r *OpenAIRouter) handleAutoModelRouting(openAIRequest *openai.ChatCompletionNewParams, originalModel string, categoryName string, reasoningDecision entropy.ReasoningDecision, selectedModel string, ctx *RequestContext, response *ext_proc.ProcessingResponse) (*ext_proc.ProcessingResponse, error) {
	logging.Infof("Using Auto Model Selection (model=%s), category=%s, selected=%s",
		originalModel, categoryName, selectedModel)

	matchedModel := selectedModel

	if matchedModel == originalModel || matchedModel == "" {
		// No model change needed
		ctx.RequestModel = originalModel
		return response, nil
	}

	// Record routing decision with tracing
	r.recordRoutingDecision(ctx, categoryName, originalModel, matchedModel, reasoningDecision)

	// Track VSR decision information
	r.trackVSRDecision(ctx, categoryName, matchedModel, reasoningDecision.UseReasoning)

	// Track model routing metrics
	metrics.RecordModelRouting(originalModel, matchedModel)

	// Select endpoint for the matched model
	selectedEndpoint := r.selectEndpointForModel(ctx, matchedModel)

	// Modify request body with new model, reasoning mode, and system prompt
	modifiedBody, err := r.modifyRequestBodyForAutoRouting(openAIRequest, matchedModel, categoryName, reasoningDecision.UseReasoning, ctx)
	if err != nil {
		return nil, err
	}

	// Create response with mutations
	response = r.createRoutingResponse(matchedModel, selectedEndpoint, modifiedBody)

	// Log routing decision
	r.logRoutingDecision(ctx, "auto_routing", originalModel, matchedModel, categoryName, reasoningDecision.UseReasoning, selectedEndpoint)

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

	return endpointAddress
}

// modifyRequestBodyForAutoRouting modifies the request body for auto routing
func (r *OpenAIRouter) modifyRequestBodyForAutoRouting(openAIRequest *openai.ChatCompletionNewParams, matchedModel string, categoryName string, useReasoning bool, ctx *RequestContext) ([]byte, error) {
	// Modify the model in the request
	openAIRequest.Model = matchedModel

	// Serialize the modified request
	modifiedBody, err := serializeOpenAIRequestWithStream(openAIRequest, ctx.ExpectStreamingResponse)
	if err != nil {
		logging.Errorf("Error serializing modified request: %v", err)
		metrics.RecordRequestError(matchedModel, "serialization_error")
		return nil, status.Errorf(codes.Internal, "error serializing modified request: %v", err)
	}

	if categoryName == "" {
		return modifiedBody, nil
	}
	// Set reasoning mode
	modifiedBody, err = r.setReasoningModeToRequestBody(modifiedBody, useReasoning, categoryName)
	if err != nil {
		logging.Errorf("Error setting reasoning mode %v to request: %v", useReasoning, err)
		metrics.RecordRequestError(matchedModel, "serialization_error")
		return nil, status.Errorf(codes.Internal, "error setting reasoning mode: %v", err)
	}

	// Add category-specific system prompt if configured
	modifiedBody, err = r.addSystemPromptIfConfigured(modifiedBody, categoryName, matchedModel, ctx)
	if err != nil {
		return nil, err
	}

	return modifiedBody, nil
}

// createRoutingResponse creates a routing response with mutations
func (r *OpenAIRouter) createRoutingResponse(model string, endpoint string, modifiedBody []byte) *ext_proc.ProcessingResponse {
	bodyMutation := &ext_proc.BodyMutation{
		Mutation: &ext_proc.BodyMutation_Body{
			Body: modifiedBody,
		},
	}

	setHeaders := []*core.HeaderValueOption{}
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

	headerMutation := &ext_proc.HeaderMutation{
		RemoveHeaders: []string{"content-length"},
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
