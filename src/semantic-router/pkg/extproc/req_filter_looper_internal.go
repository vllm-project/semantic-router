package extproc

import (
	"fmt"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// getReasoningInfoFromDecision extracts reasoning configuration from a decision for a specific model.
func (r *OpenAIRouter) getReasoningInfoFromDecision(
	decision *config.Decision,
	modelName string,
) (bool, string) {
	for _, ref := range decision.ModelRefs {
		if ref.Model == modelName || ref.LoRAName == modelName {
			if ref.UseReasoning == nil {
				break
			}

			reasoningEffort := ref.ReasoningEffort
			if reasoningEffort == "" {
				reasoningEffort = "medium"
			}
			logging.ComponentDebugEvent("extproc", "looper_reasoning_config_found", map[string]interface{}{
				"model":             modelName,
				"source":            "model_ref",
				"reasoning_enabled": *ref.UseReasoning,
				"reasoning_effort":  reasoningEffort,
			})
			return *ref.UseReasoning, reasoningEffort
		}
	}

	if r.Config == nil || r.Config.ModelConfig == nil {
		return false, ""
	}

	params, ok := r.Config.ModelConfig[modelName]
	if !ok || params.ReasoningFamily == "" {
		return false, ""
	}

	reasoningEffort := "medium"
	if r.Config.DefaultReasoningEffort != "" {
		reasoningEffort = r.Config.DefaultReasoningEffort
	}
	logging.ComponentDebugEvent("extproc", "looper_reasoning_config_found", map[string]interface{}{
		"model":            modelName,
		"source":           "model_params",
		"reasoning_family": params.ReasoningFamily,
		"reasoning_effort": reasoningEffort,
	})
	return true, reasoningEffort
}

// modifyRequestBodyForLooper modifies the request body for looper internal requests.
func (r *OpenAIRouter) modifyRequestBodyForLooper(
	request *llmprotocol.Request,
	modelName string,
	decisionName string,
	useReasoning bool,
	ctx *RequestContext,
) error {
	if request == nil {
		return fmt.Errorf("neutral looper request is unavailable")
	}
	changed := request.Model != modelName || request.Stream != ctx.ExpectStreamingResponse
	request.Model = modelName
	request.Stream = ctx.ExpectStreamingResponse

	if decisionName != "" {
		targetFormat, err := wireFormatForModel(r.Config.GetModelAPIFormat(modelName))
		if err != nil {
			return fmt.Errorf("resolve looper target format: %w", err)
		}
		changed = r.applySemanticReasoningMode(request, modelName, targetFormat, useReasoning, ctx.VSRSelectedDecision) || changed
		injected, err := r.addSemanticSystemPromptIfConfigured(request, decisionName, modelName, ctx)
		if err != nil {
			return fmt.Errorf("apply looper system instruction: %w", err)
		}
		changed = injected || changed
	}
	if changed {
		request.Generation++
	}
	return nil
}

// buildLooperBackendDispatchResponse sends internal Looper hops through the
// same provider boundary as ordinary requests.
func (r *OpenAIRouter) buildLooperBackendDispatchResponse(
	modelName string,
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	dispatch, err := r.prepareProviderDispatch(ctx.SemanticRequest, modelName, "", false, ctx)
	if err != nil {
		return nil, err
	}
	response := r.buildProviderDispatchResponse(dispatch, ctx)
	common := response.GetRequestBody().GetResponse()
	if common == nil {
		return r.createErrorResponse(500, "Failed to build provider dispatch"), nil
	}
	if common.HeaderMutation == nil {
		common.HeaderMutation = &ext_proc.HeaderMutation{}
	}
	common.HeaderMutation.RemoveHeaders = append(
		common.HeaderMutation.RemoveHeaders,
		looperInternalHeadersForRemoval()...,
	)
	setHeaderValue(common.HeaderMutation, headers.VSRSelectedModel, modelName)
	return r.finalizeProviderDispatchResponse(dispatch, response, ctx)
}

// handleLooperInternalRequest handles requests from looper to extproc.
func (r *OpenAIRouter) handleLooperInternalRequest(
	modelName string,
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	logging.ComponentDebugEvent("extproc", "looper_request_handled", map[string]interface{}{
		"request_id": ctx.RequestID,
		"model":      modelName,
	})

	if ctx.SemanticRequest == nil {
		return r.createErrorResponse(400, "Invalid inference request"), nil
	}
	ctx.SemanticRequest.Model = modelName
	ctx.SemanticRequest.Generation++
	ctx.VSRSelectedModel = modelName
	ctx.RequestModel = modelName
	return r.buildLooperBackendDispatchResponse(modelName, ctx)
}

// handleLooperInternalRequestWithPlugins handles looper internal requests with plugin execution.
func (r *OpenAIRouter) handleLooperInternalRequestWithPlugins(
	modelName string,
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	r.hydrateLooperRoutingContext(ctx)
	decisionName := headerValueCI(ctx, headers.VSRLooperDecision)
	decision, fallback := r.resolveLooperDecision(modelName, decisionName, ctx)
	if fallback != nil {
		return fallback, nil
	}

	r.prepareLooperInternalContext(decisionName, decision, modelName, ctx)
	useReasoning, reasoningEffort := r.getReasoningInfoFromDecision(decision, modelName)
	applyLooperReasoningContext(ctx, modelName, useReasoning, reasoningEffort)

	request, err := r.parseLooperRequestForPlugins(ctx)
	if err != nil {
		return r.createErrorResponse(400, "Invalid request body"), nil
	}

	if response := r.runLooperInternalPlugins(ctx, decisionName); response != nil {
		return response, nil
	}
	compressionErr := r.applySemanticContextCompression(ctx, request)
	if compressionErr != nil {
		return r.createErrorResponse(
			500,
			"Context compression failed under fail_closed policy",
		), nil
	}
	err = r.modifyRequestBodyForLooper(
		request,
		modelName,
		decisionName,
		useReasoning,
		ctx,
	)
	if err != nil {
		logging.ComponentErrorEvent("extproc", "looper_request_modify_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"model":      modelName,
			"decision":   decisionName,
			"error":      err.Error(),
		})
		return r.createErrorResponse(500, "Failed to process looper request"), nil
	}

	r.startLooperInternalReplay(ctx, modelName, decisionName)
	return r.buildLooperBackendDispatchResponse(modelName, ctx)
}

func (r *OpenAIRouter) resolveLooperDecision(
	modelName string,
	decisionName string,
	ctx *RequestContext,
) (*config.Decision, *ext_proc.ProcessingResponse) {
	if decisionName == "" {
		logging.ComponentWarnEvent("extproc", "looper_decision_missing", map[string]interface{}{
			"request_id": ctx.RequestID,
			"model":      modelName,
			"fallback":   "simple_routing",
		})
		response, _ := r.handleLooperInternalRequest(modelName, ctx)
		return nil, response
	}

	logging.ComponentDebugEvent("extproc", "looper_request_processing", map[string]interface{}{
		"request_id": ctx.RequestID,
		"model":      modelName,
		"decision":   decisionName,
	})

	decision := r.looperDecisionForRoutingContext(ctx, decisionName)
	if decision != nil {
		return decision, nil
	}

	logging.ComponentWarnEvent("extproc", "looper_decision_not_found", map[string]interface{}{
		"request_id": ctx.RequestID,
		"model":      modelName,
		"decision":   decisionName,
		"fallback":   "simple_routing",
	})
	response, _ := r.handleLooperInternalRequest(modelName, ctx)
	return nil, response
}

func (r *OpenAIRouter) prepareLooperInternalContext(
	decisionName string,
	decision *config.Decision,
	modelName string,
	ctx *RequestContext,
) {
	ctx.VSRSelectedDecision = decision
	ctx.VSRSelectedDecisionName = decisionName
	ctx.VSRSelectedModel = modelName
	ctx.RequestModel = modelName

	if replayCfg := r.Config.EffectiveRouterReplayConfig(decision); replayCfg != nil {
		cfgCopy := *replayCfg
		ctx.RouterReplayPluginConfig = &cfgCopy
		logging.ComponentDebugEvent("extproc", "looper_router_replay_enabled", map[string]interface{}{
			"request_id": ctx.RequestID,
			"decision":   decisionName,
		})
	}
}

func applyLooperReasoningContext(
	ctx *RequestContext,
	modelName string,
	useReasoning bool,
	reasoningEffort string,
) {
	if useReasoning {
		ctx.VSRReasoningMode = "on"
		logging.ComponentDebugEvent("extproc", "looper_reasoning_enabled", map[string]interface{}{
			"request_id":       ctx.RequestID,
			"model":            modelName,
			"reasoning_effort": reasoningEffort,
		})
		return
	}

	ctx.VSRReasoningMode = "off"
}

func (r *OpenAIRouter) parseLooperRequestForPlugins(
	ctx *RequestContext,
) (*llmprotocol.Request, error) {
	if ctx == nil || ctx.SemanticRequest == nil {
		err := fmt.Errorf("neutral looper request is unavailable")
		requestID := ""
		if ctx != nil {
			requestID = ctx.RequestID
		}
		logging.ComponentErrorEvent("extproc", "looper_request_parse_failed", map[string]interface{}{
			"request_id": requestID,
			"error":      err.Error(),
		})
		return nil, err
	}
	ctx.UserContent = extractSemanticRequestSignals(ctx.SemanticRequest).UserContent
	return ctx.SemanticRequest, nil
}

func (r *OpenAIRouter) runLooperInternalPlugins(
	ctx *RequestContext,
	decisionName string,
) *ext_proc.ProcessingResponse {
	if response := r.handleFastResponse(ctx, decisionName); response != nil {
		return response
	}

	if response, shouldReturn := r.handleCaching(ctx, decisionName); shouldReturn {
		return response
	}

	if err := r.executeRAGPlugin(ctx, decisionName); err != nil {
		return r.createErrorResponse(503, fmt.Sprintf("RAG failed: %v", err))
	}

	return nil
}

func (r *OpenAIRouter) startLooperInternalReplay(
	ctx *RequestContext,
	modelName string,
	decisionName string,
) {
	ctx.RouterReplayID = ""
	r.startRouterReplay(ctx, "ReMoM", modelName, decisionName)
}
