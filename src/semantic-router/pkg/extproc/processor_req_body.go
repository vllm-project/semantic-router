package extproc

import (
	"fmt"
	"net/http"
	"strings"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/inflight"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
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
// The ingress codec decodes the wire request once into the neutral protocol
// contract used by routing, plugins, backend dispatch, and accounting.
func (r *OpenAIRouter) handleRequestBody(
	v *ext_proc.ProcessingRequest_RequestBody,
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	ctx.ProcessingStartTime = time.Now()
	requestBody := v.RequestBody.GetBody()
	request, earlyResponse := r.prepareProtocolRequest(requestBody, ctx)
	if earlyResponse != nil {
		return earlyResponse, nil
	}
	snapshot, err := r.extractRequestSignalSnapshot(ctx)
	if validationResp := r.validationResponseFromRequestError(err); validationResp != nil {
		return validationResp, nil
	}
	if err != nil {
		return nil, err
	}

	originalModel := strings.TrimSpace(snapshot.Model)
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
	ctx.UserContent = snapshot.UserContent
	ctx.RequestImageURL = snapshot.FirstImageURL

	decisionState, earlyResponse := r.runRequestPreRoutingStages(originalModel, snapshot, ctx)
	if earlyResponse != nil {
		return earlyResponse, nil
	}
	request, earlyResponse, err = r.prepareRequestForModelRouting(request, snapshot.UserContent, ctx)
	if earlyResponse != nil {
		return earlyResponse, nil
	}
	if validationResp := r.validationResponseFromRequestError(err); validationResp != nil {
		return validationResp, nil
	}
	if err != nil {
		return nil, err
	}
	return r.handleModelRoutingWithPersonalizedCache(
		request,
		originalModel,
		decisionState,
		ctx,
	)
}

func (r *OpenAIRouter) handleModelRoutingWithPersonalizedCache(
	request *llmprotocol.Request,
	originalModel string,
	decisionState requestDecisionState,
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	if response, hit := r.lookupPersonalizedExactCache(ctx, decisionState.decisionName, decisionState.selectedModel); hit {
		inflight.End(decisionState.selectedModel, ctx.InflightToken)
		ctx.InflightToken = 0
		return response, nil
	}
	return r.handleModelRouting(
		request,
		originalModel,
		decisionState.decisionName,
		decisionState.reasoningDecision,
		decisionState.selectedModel,
		ctx,
	)
}

// handleModelRouting handles model selection and routing logic
// decisionName, reasoningDecision, and selectedModel are pre-computed from ProcessRequest
func (r *OpenAIRouter) handleModelRouting(request *llmprotocol.Request, originalModel string, decisionName string, reasoningDecision entropy.ReasoningDecision, selectedModel string, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	if changed, err := r.applyPreDispatchToolsPolicy(ctx); err != nil {
		return nil, err
	} else if changed {
		logging.ComponentDebugEvent("extproc", "pre_dispatch_tools_policy_applied", map[string]interface{}{
			"request_id": ctx.RequestID,
			"decision":   decisionName,
		})
	}
	isEntrypoint := ctx.Routing.SelectedRecipe() != nil
	executesLooper := r.routeExecutesLooper(ctx)
	if !isEntrypoint && !executesLooper {
		if unavailable := r.unavailableModelResponse(originalModel, ctx); unavailable != nil {
			return unavailable, nil
		}
	}
	response := &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					Status: ext_proc.CommonResponse_CONTINUE,
				},
			},
		},
	}

	if !isEntrypoint {
		return r.handleSpecifiedModelRouting(request, originalModel, decisionName, ctx)
	}
	return r.handleEntrypointRouting(
		request,
		originalModel,
		decisionName,
		reasoningDecision,
		selectedModel,
		ctx,
		response,
	)
}

func (r *OpenAIRouter) routeExecutesLooper(ctx *RequestContext) bool {
	if r == nil || r.Config == nil {
		return false
	}
	return ctx != nil && r.shouldUseLooper(ctx.VSRSelectedDecision)
}

func (r *OpenAIRouter) handleEntrypointRouting(
	request *llmprotocol.Request,
	originalModel string,
	decisionName string,
	reasoningDecision entropy.ReasoningDecision,
	selectedModel string,
	ctx *RequestContext,
	response *ext_proc.ProcessingResponse,
) (*ext_proc.ProcessingResponse, error) {
	if r.shouldUseLooper(ctx.VSRSelectedDecision) {
		logging.ComponentDebugEvent("extproc", "looper_execution_selected", map[string]interface{}{
			"request_id": ctx.RequestID,
			"decision":   ctx.VSRSelectedDecision.Name,
			"algorithm":  ctx.VSRSelectedDecision.Algorithm.Type,
		})
		// Looper execution uses the same selected-decision contract as a
		// single-Model Entrypoint path.
		ctx.VSRSelectedDecisionName = ctx.VSRSelectedDecision.Name
		return r.handleLooperExecution(ctx.TraceContext, request, ctx.VSRSelectedDecision, ctx)
	}
	if selectedModel != "" {
		return r.handleEntrypointModelRouting(request, originalModel, decisionName, reasoningDecision, selectedModel, ctx)
	}

	logging.ComponentWarnEvent("extproc", "entrypoint_routing_no_selection", map[string]interface{}{
		"request_id": ctx.RequestID,
	})
	metrics.RecordRequestError(originalModel, "no_model_selected")
	return r.createErrorResponse(http.StatusBadRequest, "unable to route request: the Entrypoint selected no model"), nil
}

// handleEntrypointModelRouting dispatches the Model selected by a Recipe.
func (r *OpenAIRouter) handleEntrypointModelRouting(request *llmprotocol.Request, originalModel string, decisionName string, reasoningDecision entropy.ReasoningDecision, selectedModel string, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	logging.ComponentDebugEvent("extproc", "entrypoint_model_routing_selected", map[string]interface{}{
		"request_id":     ctx.RequestID,
		"original_model": originalModel,
		"decision":       decisionName,
		"selected_model": selectedModel,
	})

	matchedModel := selectedModel

	if matchedModel == originalModel || matchedModel == "" {
		ctx.RequestModel = originalModel
		dispatch, err := r.prepareProviderDispatch(
			request, originalModel, decisionName, reasoningDecision.UseReasoning, ctx,
		)
		if err != nil {
			return r.imageFileDispatchFailure(err, ctx)
		}
		response := r.buildProviderDispatchResponse(dispatch, ctx)
		r.handleToolSelectionForRequest(request, response, ctx)
		return r.finalizeProviderDispatchResponse(dispatch, response, ctx)
	}

	// Record routing decision with tracing
	r.recordRoutingDecision(ctx, decisionName, originalModel, matchedModel, reasoningDecision)

	// Track VSR decision information
	// categoryName is already set in ctx.VSRSelectedCategory by performDecisionEvaluation
	r.trackVSRDecision(ctx, ctx.VSRSelectedCategory, decisionName, matchedModel, reasoningDecision.UseReasoning)

	// Track model routing metrics
	metrics.RecordModelRouting(originalModel, matchedModel)

	dispatch, err := r.prepareProviderDispatch(
		request, matchedModel, decisionName, reasoningDecision.UseReasoning, ctx,
	)
	if err != nil {
		return r.imageFileDispatchFailure(err, ctx)
	}

	response := r.buildProviderDispatchResponse(dispatch, ctx)

	// Log routing decision
	r.logRoutingDecision(ctx, "entrypoint_routing", originalModel, matchedModel, decisionName, reasoningDecision.UseReasoning)

	// Handle route cache clearing
	if r.shouldClearRouteCache() {
		r.setClearRouteCache(response)
	}

	// Save the actual model for token tracking
	ctx.RequestModel = matchedModel

	// Capture router replay information if enabled
	r.startRouterReplay(ctx, originalModel, matchedModel, decisionName)

	// Handle tool selection
	r.handleToolSelectionForRequest(request, response, ctx)
	response, err = r.finalizeProviderDispatchResponse(dispatch, response, ctx)
	if err != nil {
		return nil, err
	}

	// Record routing latency
	r.recordRoutingLatency(ctx)

	return response, nil
}

// handleSpecifiedModelRouting handles routing for explicitly specified models
func (r *OpenAIRouter) handleSpecifiedModelRouting(request *llmprotocol.Request, originalModel string, decisionName string, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	logging.ComponentDebugEvent("extproc", "specified_model_routing_selected", map[string]interface{}{
		"request_id": ctx.RequestID,
		"model":      originalModel,
	})

	// Reject models that are not configured. Without this guard an unknown
	// model is forwarded with no resolvable backend credential and surfaces as
	// a misleading upstream "401 No api key" instead of a clear client error.
	if unavailable := r.unavailableModelResponse(originalModel, ctx); unavailable != nil {
		return unavailable, nil
	}

	// Concrete backend models bypass every recipe-local signal, decision, and
	// plugin. They still use shared provider/backend infrastructure.
	ctx.VSRSelectedDecisionName = decisionName
	ctx.VSRSelectedModel = originalModel
	ctx.VSRReasoningMode = "off" // Concrete Models do not use Recipe reasoning mode by default.

	dispatch, err := r.prepareProviderDispatch(request, originalModel, "", false, ctx)
	if err != nil {
		return r.imageFileDispatchFailure(err, ctx)
	}
	response := r.buildProviderDispatchResponse(dispatch, ctx)

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
	r.handleToolSelectionForRequest(request, response, ctx)
	response, err = r.finalizeProviderDispatchResponse(dispatch, response, ctx)
	if err != nil {
		return nil, err
	}

	// Record routing latency
	r.recordRoutingLatency(ctx)

	return response, nil
}

func (r *OpenAIRouter) unavailableModelResponse(
	model string,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	if r != nil && r.Config != nil {
		if len(r.Config.GetEndpointsForModel(model)) > 0 {
			return nil
		}
	}
	requestID := ""
	if ctx != nil {
		requestID = ctx.RequestID
	}
	logging.ComponentWarnEvent("extproc", "specified_model_not_found", map[string]interface{}{
		"request_id": requestID,
		"model":      model,
	})
	metrics.RecordRequestError(model, "model_not_found")
	return r.createErrorResponse(http.StatusBadRequest, fmt.Sprintf("model %q is not available", model))
}
