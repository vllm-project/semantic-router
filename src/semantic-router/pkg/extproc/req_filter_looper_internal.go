package extproc

import (
	"fmt"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// findDecisionByName finds a decision by name in the router configuration.
func (r *OpenAIRouter) findDecisionByName(name string) *config.Decision {
	if r.Config == nil || r.Config.Decisions == nil {
		return nil
	}

	for i := range r.Config.Decisions {
		if r.Config.Decisions[i].Name == name {
			return &r.Config.Decisions[i]
		}
	}

	return nil
}

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
			logging.Infof(
				"[Looper] Found reasoning config in ModelRef: use=%v, effort=%s",
				*ref.UseReasoning,
				reasoningEffort,
			)
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
	logging.Infof(
		"[Looper] Found reasoning_family in ModelParams: %s, effort=%s",
		params.ReasoningFamily,
		reasoningEffort,
	)
	return true, reasoningEffort
}

// modifyRequestBodyForLooper modifies the request body for looper internal requests.
func (r *OpenAIRouter) modifyRequestBodyForLooper(
	openAIRequest *openai.ChatCompletionNewParams,
	modelName string,
	decisionName string,
	useReasoning bool,
	ctx *RequestContext,
) ([]byte, error) {
	openAIRequest.Model = modelName

	modifiedBody, err := serializeOpenAIRequestWithStream(
		openAIRequest,
		ctx.ExpectStreamingResponse,
	)
	if err != nil {
		logging.Errorf("[Looper] Error serializing modified request: %v", err)
		return nil, fmt.Errorf("error serializing modified request: %w", err)
	}

	if decisionName == "" {
		return modifiedBody, nil
	}

	modifiedBody, err = r.setReasoningModeToRequestBody(
		modifiedBody,
		useReasoning,
		decisionName,
	)
	if err != nil {
		logging.Errorf(
			"[Looper] Error setting reasoning mode %v to request: %v",
			useReasoning,
			err,
		)
		return nil, fmt.Errorf("error setting reasoning mode: %w", err)
	}

	modifiedBody, err = r.addSystemPromptIfConfigured(
		modifiedBody,
		decisionName,
		modelName,
		ctx,
	)
	if err != nil {
		logging.Errorf("[Looper] Error adding system prompt: %v", err)
		return nil, fmt.Errorf("error adding system prompt: %w", err)
	}

	return modifiedBody, nil
}

// buildHeaderMutationsForLooper builds header mutations for looper internal requests.
func (r *OpenAIRouter) buildHeaderMutationsForLooper(
	decision *config.Decision,
	modelName string,
) ([]*core.HeaderValueOption, []string) {
	setHeaders := []*core.HeaderValueOption{
		newHeaderValueOption(headers.VSRSelectedModel, modelName),
	}
	removeHeaders := []string{"content-length"}

	if accessKey := r.getModelAccessKey(modelName); accessKey != "" {
		setHeaders = append(
			setHeaders,
			newHeaderValueOption("Authorization", fmt.Sprintf("Bearer %s", accessKey)),
		)
		logging.Infof("[Looper] Added Authorization header for model %s", modelName)
	}

	if decision == nil {
		return setHeaders, removeHeaders
	}

	pluginSetHeaders, pluginRemoveHeaders := r.buildHeaderMutations(decision)
	if len(pluginSetHeaders) > 0 {
		setHeaders = append(setHeaders, pluginSetHeaders...)
		logging.Infof(
			"[Looper] Applied %d header mutations from decision %s",
			len(pluginSetHeaders),
			decision.Name,
		)
	}
	if len(pluginRemoveHeaders) > 0 {
		removeHeaders = append(removeHeaders, pluginRemoveHeaders...)
		logging.Infof(
			"[Looper] Applied %d header deletions from decision %s",
			len(pluginRemoveHeaders),
			decision.Name,
		)
	}

	return setHeaders, removeHeaders
}

// handleLooperInternalRequest handles requests from looper to extproc.
func (r *OpenAIRouter) handleLooperInternalRequest(
	modelName string,
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	logging.Infof("[Looper] Handling internal request for model: %s", modelName)

	modifiedBody, err := rewriteRequestModel(ctx.OriginalRequestBody, modelName)
	if err != nil {
		logging.Errorf("[Looper] Failed to rewrite request body: %v", err)
		return r.createErrorResponse(500, "Failed to process looper request: "+err.Error()), nil
	}

	setHeaders := []*core.HeaderValueOption{newHeaderValueOption(headers.VSRSelectedModel, modelName)}
	return buildLooperContinueResponse(modifiedBody, setHeaders, nil), nil
}

// handleLooperInternalRequestWithPlugins handles looper internal requests with plugin execution.
func (r *OpenAIRouter) handleLooperInternalRequestWithPlugins(
	modelName string,
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	decisionName := ctx.Headers[headers.VSRLooperDecision]
	decision, fallback := r.resolveLooperDecision(modelName, decisionName, ctx)
	if fallback != nil {
		return fallback, nil
	}

	r.prepareLooperInternalContext(decisionName, decision, modelName, ctx)
	useReasoning, reasoningEffort := r.getReasoningInfoFromDecision(decision, modelName)
	applyLooperReasoningContext(ctx, modelName, useReasoning, reasoningEffort)

	openAIRequest, err := r.parseLooperRequestForPlugins(ctx)
	if err != nil {
		return r.createErrorResponse(400, "Invalid request body"), nil
	}

	if response := r.runLooperInternalPlugins(ctx, decisionName); response != nil {
		return response, nil
	}

	modifiedBody, err := r.modifyRequestBodyForLooper(
		openAIRequest,
		modelName,
		decisionName,
		useReasoning,
		ctx,
	)
	if err != nil {
		logging.Errorf("[Looper] Failed to modify request body: %v", err)
		return r.createErrorResponse(500, "Failed to process looper request"), nil
	}

	setHeaders, removeHeaders := r.buildHeaderMutationsForLooper(decision, modelName)
	r.startLooperInternalReplay(ctx, modelName, decisionName)
	return buildLooperContinueResponse(modifiedBody, setHeaders, removeHeaders), nil
}

func (r *OpenAIRouter) resolveLooperDecision(
	modelName string,
	decisionName string,
	ctx *RequestContext,
) (*config.Decision, *ext_proc.ProcessingResponse) {
	if decisionName == "" {
		logging.Warnf("[Looper] No decision name in looper request, falling back to simple routing")
		response, _ := r.handleLooperInternalRequest(modelName, ctx)
		return nil, response
	}

	logging.Infof(
		"[Looper] Processing internal request for model: %s, decision: %s",
		modelName,
		decisionName,
	)

	decision := r.findDecisionByName(decisionName)
	if decision != nil {
		return decision, nil
	}

	logging.Warnf("[Looper] Decision %s not found, falling back to simple routing", decisionName)
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

	if replayCfg := decision.GetRouterReplayConfig(); replayCfg != nil && replayCfg.Enabled {
		cfgCopy := *replayCfg
		ctx.RouterReplayPluginConfig = &cfgCopy
		logging.Debugf("[Looper] Router replay enabled for decision %s", decisionName)
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
		logging.Infof(
			"[Looper] Reasoning enabled for model %s, effort: %s",
			modelName,
			reasoningEffort,
		)
		return
	}

	ctx.VSRReasoningMode = "off"
}

func (r *OpenAIRouter) parseLooperRequestForPlugins(
	ctx *RequestContext,
) (*openai.ChatCompletionNewParams, error) {
	openAIRequest, err := parseOpenAIRequest(ctx.OriginalRequestBody)
	if err != nil {
		logging.Errorf("[Looper] Failed to parse request body: %v", err)
		return nil, err
	}

	userContent, _ := extractUserAndNonUserContent(openAIRequest)
	ctx.UserContent = userContent
	return openAIRequest, nil
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

func buildLooperContinueResponse(
	modifiedBody []byte,
	setHeaders []*core.HeaderValueOption,
	removeHeaders []string,
) *ext_proc.ProcessingResponse {
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					Status: ext_proc.CommonResponse_CONTINUE,
					HeaderMutation: &ext_proc.HeaderMutation{
						SetHeaders:    setHeaders,
						RemoveHeaders: removeHeaders,
					},
					BodyMutation: &ext_proc.BodyMutation{
						Mutation: &ext_proc.BodyMutation_Body{Body: modifiedBody},
					},
					ClearRouteCache: true,
				},
			},
		},
	}
}
