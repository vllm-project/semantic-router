package extproc

import (
	"fmt"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

type requestDecisionState struct {
	decisionName      string
	reasoningDecision entropy.ReasoningDecision
	selectedModel     string
}

// handleRequestBody processes the request body.
//
// The hot path uses gjson-based field extraction (extractContentFast) to avoid
// the expensive json.Unmarshal into the full OpenAI SDK struct. The SDK struct
// is parsed lazily — only when body mutations are actually needed (modality
// routing, memory injection, model routing). Requests that hit fast_response,
// rate limiting, or cache never pay the full parse cost.
func (r *OpenAIRouter) handleRequestBody(v *ext_proc.ProcessingRequest_RequestBody, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	ctx.ProcessingStartTime = time.Now()
	ctx.OriginalRequestBody = v.RequestBody.GetBody()

	requestBody, earlyResponse := r.translateResponseAPIRequest(ctx.OriginalRequestBody, ctx)
	if earlyResponse != nil {
		return earlyResponse, nil
	}
	if validationResp := r.validateRequestBody(requestBody, ctx); validationResp != nil {
		return validationResp, nil
	}

	fast, err := r.extractFastRequestState(requestBody, ctx)
	if validationResp := r.validationResponseFromRequestError(err); validationResp != nil {
		return validationResp, nil
	}
	if err != nil {
		return nil, err
	}

	originalModel := fast.Model
	if ctx.RequestModel == "" {
		ctx.RequestModel = originalModel
	}
	if r.isLooperRequest(ctx) {
		logging.ComponentDebugEvent("extproc", "looper_internal_request_detected", map[string]interface{}{
			"request_id": ctx.RequestID,
			"model":      originalModel,
		})
		return r.handleLooperInternalRequestWithPlugins(originalModel, ctx)
	}

	ctx.UserContent = fast.UserContent
	ctx.RequestImageURL = fast.FirstImageURL

	decisionState, earlyResponse := r.runRequestPreRoutingStages(originalModel, fast, ctx)
	if earlyResponse != nil {
		return earlyResponse, nil
	}

	openAIRequest, earlyResponse, err := r.prepareRequestForModelRouting(requestBody, fast.UserContent, ctx)
	if earlyResponse != nil {
		return earlyResponse, nil
	}
	if validationResp := r.validationResponseFromRequestError(err); validationResp != nil {
		return validationResp, nil
	}
	if err != nil {
		return nil, err
	}

	return r.handleModelRouting(
		openAIRequest,
		originalModel,
		decisionState.decisionName,
		decisionState.reasoningDecision,
		decisionState.selectedModel,
		ctx,
	)
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
		return r.handleSpecifiedModelRouting(openAIRequest, originalModel, decisionName, ctx)
	case r.shouldUseLooper(ctx.VSRSelectedDecision):
		logging.ComponentDebugEvent("extproc", "looper_execution_selected", map[string]interface{}{
			"request_id": ctx.RequestID,
			"decision":   ctx.VSRSelectedDecision.Name,
			"algorithm":  ctx.VSRSelectedDecision.Algorithm.Type,
		})
		return r.handleLooperExecution(ctx.TraceContext, openAIRequest, ctx.VSRSelectedDecision, ctx)
	case selectedModel != "":
		return r.handleAutoModelRouting(openAIRequest, originalModel, decisionName, reasoningDecision, selectedModel, ctx, response)
	default:
		// Auto model without selection - no routing needed
		ctx.RequestModel = originalModel
		return response, nil
	}
}

// handleAutoModelRouting handles routing for auto model selection
func (r *OpenAIRouter) handleAutoModelRouting(openAIRequest *openai.ChatCompletionNewParams, originalModel string, decisionName string, reasoningDecision entropy.ReasoningDecision, selectedModel string, ctx *RequestContext, response *ext_proc.ProcessingResponse) (*ext_proc.ProcessingResponse, error) {
	logging.ComponentDebugEvent("extproc", "auto_model_routing_selected", map[string]interface{}{
		"request_id":     ctx.RequestID,
		"original_model": originalModel,
		"decision":       decisionName,
		"selected_model": selectedModel,
	})

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
func (r *OpenAIRouter) handleSpecifiedModelRouting(openAIRequest *openai.ChatCompletionNewParams, originalModel string, decisionName string, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	logging.ComponentDebugEvent("extproc", "specified_model_routing_selected", map[string]interface{}{
		"request_id": ctx.RequestID,
		"model":      originalModel,
	})

	// Track VSR decision information for non-auto models
	ctx.VSRSelectedDecisionName = decisionName
	ctx.VSRSelectedModel = originalModel
	ctx.VSRReasoningMode = "off" // Non-auto models don't use reasoning mode by default
	// Security checks (jailbreak/PII) are handled at the signal level via fast_response plugin
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
	r.logRoutingDecision(ctx, "model_specified", originalModel, originalModel, decisionName, false)

	// Save the actual model for token tracking
	ctx.RequestModel = originalModel

	// Capture router replay information if enabled even when the client pins a model.
	r.startRouterReplay(ctx, originalModel, originalModel, decisionName)

	// Handle tool selection
	r.handleToolSelectionForRequest(openAIRequest, response, ctx)

	// Record routing latency
	r.recordRoutingLatency(ctx)

	return response, nil
}

// selectEndpointForModel selects the best endpoint for the given model.
// Returns the endpoint address:port, the endpoint name, and any error.
// Backend selection is now part of the model layer (upstream request span)
