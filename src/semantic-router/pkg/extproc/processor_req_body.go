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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
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
) (response *ext_proc.ProcessingResponse, err error) {
	defer func() {
		if !r.managedInferenceAccessEnabled() || ctx == nil || ctx.InferenceAccess == nil {
			return
		}
		if err != nil {
			if settleErr := r.settleImmediateInference(ctx, nil, "request_processing_failed"); settleErr != nil {
				logging.Errorf("failed to settle inference after request processing error: %v", settleErr)
			}
			return
		}
		if response == nil || response.GetImmediateResponse() == nil {
			return
		}
		if settleErr := r.settleImmediateInference(ctx, response, "request_short_circuit"); settleErr != nil {
			logging.Errorf("failed to settle immediate inference response: %v", settleErr)
			response = r.createInferenceAccessError(quotaruntime.AdmissionUnavailable, nil)
		}
	}()
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
	if accessResponse := r.authorizeInferenceTarget(ctx.TraceContext, ctx, originalModel); accessResponse != nil {
		return accessResponse, nil
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
	isEntrypoint := r.requestModelIsEntrypoint(originalModel)
	executesLooper := r.routeExecutesLooper(ctx)
	if !isEntrypoint && !executesLooper {
		if unavailable := r.unavailableModelResponse(originalModel, ctx); unavailable != nil {
			return unavailable, nil
		}
	}
	if !executesLooper {
		dispatchModel := selectedModel
		if !r.requestModelIsEntrypoint(originalModel) {
			dispatchModel = originalModel
		} else if dispatchModel == "" {
			dispatchModel = r.Config.DefaultModel
		}
		if dispatchModel != "" {
			if err := r.beginPrimaryInferenceDispatch(ctx.TraceContext, ctx, dispatchModel); err != nil {
				return r.createInferenceAccessError(quotaruntime.AdmissionUnavailable, nil), nil
			}
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
		// No model change is needed, but route-local request plugins may still
		// have changed the provider-bound body.
		ctx.RequestModel = originalModel
		body, err := r.backendDispatchBody(request, originalModel, ctx)
		if err != nil {
			return nil, err
		}
		response := r.buildBackendDispatchResponse(originalModel, body, ctx)
		r.handleToolSelectionForRequest(request, response, ctx)
		return r.finalizeBackendDispatchResponse(originalModel, response, ctx), nil
	}

	// Record routing decision with tracing
	r.recordRoutingDecision(ctx, decisionName, originalModel, matchedModel, reasoningDecision)

	// Track VSR decision information
	// categoryName is already set in ctx.VSRSelectedCategory by performDecisionEvaluation
	r.trackVSRDecision(ctx, ctx.VSRSelectedCategory, decisionName, matchedModel, reasoningDecision.UseReasoning)

	// Track model routing metrics
	metrics.RecordModelRouting(originalModel, matchedModel)

	// ExtProc mutates only the canonical request and logical model identity.
	// Backend address, credentials, wire model ID, protocol translation,
	// retries, and fallback are owned by BackendInvoker's immutable snapshot.
	modifiedBody, err := r.modifyRequestBodyForEntrypointRouting(
		request,
		matchedModel,
		decisionName,
		reasoningDecision.UseReasoning,
		ctx,
	)
	if err != nil {
		return nil, err
	}

	response := r.buildBackendDispatchResponse(matchedModel, modifiedBody, ctx)

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
	response = r.finalizeBackendDispatchResponse(matchedModel, response, ctx)

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

	body, bodyErr := r.backendDispatchBody(request, originalModel, ctx)
	if bodyErr != nil {
		return nil, bodyErr
	}
	response := r.buildBackendDispatchResponse(originalModel, body, ctx)

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
	response = r.finalizeBackendDispatchResponse(originalModel, response, ctx)

	// Record routing latency
	r.recordRoutingLatency(ctx)

	return response, nil
}

func (r *OpenAIRouter) unavailableModelResponse(
	model string,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	if r != nil && r.Config != nil {
		if params, found := r.Config.ModelConfig[model]; found &&
			params.ResourceID != "" && params.ResourceRevision > 0 {
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
