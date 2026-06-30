package extproc

import (
	"encoding/json"
	"fmt"
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const directFusionDecisionName = "fusion_direct"

func (r *OpenAIRouter) handleDirectFusionExecution(
	openAIRequest *openai.ChatCompletionNewParams,
	originalModel string,
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	if r.Config == nil || !r.Config.Looper.IsEnabled() {
		return r.createErrorResponse(500, "Fusion execution requires global.integrations.looper.endpoint"), nil
	}

	ctx.RequestModel = originalModel
	decision, status, err := r.resolveDirectFusionDecision(ctx)
	if err != nil {
		return r.createErrorResponse(status, err.Error()), nil
	}
	ctx.VSRSelectedDecision = decision
	ctx.VSRSelectedDecisionName = decision.Name
	ctx.VSRSelectionMethod = "fusion"

	return r.handleLooperExecution(ctx.TraceContext, openAIRequest, decision, ctx)
}

func (r *OpenAIRouter) resolveDirectFusionDecision(ctx *RequestContext) (*config.Decision, int, error) {
	if isFusionDecision(ctx.VSRSelectedDecision) {
		return ctx.VSRSelectedDecision, 0, nil
	}
	if ctx.VSRSelectedDecision != nil {
		return nil, 400, fmt.Errorf("no eligible Fusion decision matched for model %q", ctx.RequestModel)
	}

	plugin, err := parseFusionRequestConfig(ctx.OriginalRequestBody)
	if err != nil {
		return nil, 400, fmt.Errorf("invalid Fusion plugin: %w", err)
	}
	if plugin != nil && fusionRequestOverrideEnabled(plugin) && len(plugin.AnalysisModels) > 0 {
		return buildRequestOnlyFusionDecision(), 0, nil
	}
	if decision := r.defaultLooperDecisionByAlgorithm("fusion"); decision != nil {
		return decision, 0, nil
	}

	return nil, 400, fmt.Errorf("no eligible Fusion decision matched for model %q and no request plugins[].id=fusion analysis_models override was provided", ctx.RequestModel)
}

func isFusionDecision(decision *config.Decision) bool {
	return decision != nil && decision.Algorithm != nil && decision.Algorithm.Type == "fusion"
}

func (r *OpenAIRouter) decisionCandidatesForRequestModel(modelName string) []config.Decision {
	if r == nil || r.Config == nil {
		return nil
	}
	candidates := make([]config.Decision, 0)
	if r.Config.IsReMoMModelName(modelName) {
		for _, decision := range r.Config.Decisions {
			if isReMoMDecision(&decision) {
				candidates = append(candidates, decision)
			}
		}
		return candidates
	}
	if r.Config.IsFusionModelName(modelName) {
		for _, decision := range r.Config.Decisions {
			if isFusionDecision(&decision) {
				candidates = append(candidates, decision)
			}
		}
		return candidates
	}
	if r.Config.IsFlowModelName(modelName) {
		for _, decision := range r.Config.Decisions {
			if isFlowDecision(&decision) {
				candidates = append(candidates, decision)
			}
		}
		return candidates
	}
	return nil
}

func (r *OpenAIRouter) defaultLooperDecisionByAlgorithm(algorithmType string) *config.Decision {
	if r == nil || r.Config == nil {
		return nil
	}
	var selected *config.Decision
	for i := range r.Config.Decisions {
		decision := &r.Config.Decisions[i]
		if decision.Algorithm == nil || decision.Algorithm.Type != algorithmType {
			continue
		}
		if !hasLooperModelInputs(decision) {
			continue
		}
		if selected == nil || decision.Priority > selected.Priority {
			selected = decision
		}
	}
	return selected
}

func buildRequestOnlyFusionDecision() *config.Decision {
	return &config.Decision{
		Name:      directFusionDecisionName,
		Algorithm: &config.AlgorithmConfig{Type: "fusion"},
	}
}

func fusionRequestOverrideEnabled(req *config.FusionRequestConfig) bool {
	return req.Enabled == nil || *req.Enabled
}

func parseFusionRequestConfig(body []byte) (*config.FusionRequestConfig, error) {
	var envelope struct {
		Plugins []json.RawMessage `json:"plugins"`
	}
	if len(body) == 0 {
		return nil, nil
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		return nil, fmt.Errorf("parse fusion request plugins: %w", err)
	}
	for _, raw := range envelope.Plugins {
		var probe struct {
			ID string `json:"id"`
		}
		if err := json.Unmarshal(raw, &probe); err != nil {
			return nil, fmt.Errorf("parse fusion plugin id: %w", err)
		}
		if !isFusionRequestPluginID(probe.ID) {
			continue
		}
		var fusion config.FusionRequestConfig
		if err := json.Unmarshal(raw, &fusion); err != nil {
			return nil, fmt.Errorf("parse fusion plugin: %w", err)
		}
		if err := fusion.Validate(); err != nil {
			return nil, err
		}
		return &fusion, nil
	}
	return nil, nil
}

func isFusionRequestPluginID(id string) bool {
	switch strings.TrimSpace(id) {
	case "fusion", "openrouter:fusion":
		return true
	default:
		return false
	}
}
