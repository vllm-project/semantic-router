package looper

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

var _ StreamingLooper = (*FusionLooper)(nil)

// PrepareStreamingSynthesis executes intermediate candidate analysis panels
// (buffered) and constructs the final synthesis request payload without
// executing the final upstream HTTP call directly.
func (l *FusionLooper) PrepareStreamingSynthesis(
	ctx context.Context,
	req *Request,
) (*StreamingSynthesisPlan, error) {
	if req == nil || req.OriginalRequest == nil {
		return nil, fmt.Errorf("fusion streaming synthesis requires a valid request and original params")
	}

	cfg := l.resolveFusionExecutionConfig(req)
	if len(cfg.AnalysisModels) == 0 {
		return nil, fmt.Errorf("fusion analysis_models cannot be empty")
	}
	if cfg.Model == "" {
		cfg.Model = cfg.AnalysisModels[0]
	}
	if err := validateFusionExecutionConfig(cfg); err != nil {
		return nil, err
	}
	if err := l.validateFusionModels(cfg); err != nil {
		return nil, err
	}

	logging.ComponentEvent("looper", "fusion_streaming_synthesis_prepare_started", map[string]interface{}{
		"decision":        req.DecisionName,
		"judge_model":     cfg.Model,
		"analysis_models": len(cfg.AnalysisModels),
		"analysis_mode":   cfg.AnalysisMode,
	})

	panel, err := l.executeFusionPanel(ctx, req, cfg)
	if err != nil {
		return l.applyStreamingQuorumFailurePolicy(ctx, req, cfg, panel, err)
	}
	if len(panel.failedModels) > 0 {
		logging.ComponentWarnEvent("looper", "fusion_panel_partial", map[string]interface{}{
			"decision":  req.DecisionName,
			"responses": len(panel.responses),
			"failures":  len(panel.failedModels),
		})
	}

	groundedPanel, groundingScores, groundingMode, err := l.applyGrounding(ctx, req, cfg, panel.responses)
	if err != nil {
		return nil, err
	}

	panelAttempts := len(panel.dispatchedModels)
	var (
		prompt            string
		intermediateUsage = panel.usage
		analysis          *FusionAnalysis
		iterations        = panelAttempts
		modelsUsed        []string
	)

	switch cfg.AnalysisMode {
	case config.FusionAnalysisModeSeparate:
		var analysisResp *ModelResponse
		analysis, analysisResp = l.runFusionAnalysis(
			ctx, req, cfg, groundedPanel, groundingScores, panelAttempts+1)
		if analysisResp != nil {
			intermediateUsage = intermediateUsage.Add(analysisResp)
		}
		iterations++
		modelsUsed = orderedFusionModelsUsed(panel.dispatchedModels, cfg.Model)

		original := extractOriginalContent(req.OriginalRequest)
		outputContract := requestOutputContract(req.OriginalRequest, req.OutputContract)
		prompt = buildFusionFinalPrompt(cfg, original, outputContract, groundedPanel, analysis)
		if notes := groundingSynthesisNotes(groundingScores, cfg.GroundingPolicy); notes != "" {
			prompt = prompt + "\n\n" + notes
		}

	case config.FusionAnalysisModeOneCall, config.FusionAnalysisModeNone:
		modelsUsed = orderedFusionModelsUsed(panel.dispatchedModels, "")
		original := extractOriginalContent(req.OriginalRequest)
		outputContract := requestOutputContract(req.OriginalRequest, req.OutputContract)
		prompt = buildFusionModeFinalPrompt(cfg, original, outputContract, groundedPanel)
		if notes := groundingSynthesisNotes(groundingScores, cfg.GroundingPolicy); notes != "" {
			prompt = prompt + "\n\n" + notes
		}

	default:
		return nil, fmt.Errorf("unsupported fusion analysis_mode %q", cfg.AnalysisMode)
	}

	finalReq := appendFusionStageMessage(req.OriginalRequest, prompt)
	finalReq.Model = cfg.Model
	if cfg.Temperature != nil {
		finalReq.Temperature = openai.Float(*cfg.Temperature)
	}
	if cfg.MaxCompletionTokens > 0 {
		finalReq.MaxTokens = (openai.ChatCompletionNewParams{}).MaxTokens
		finalReq.MaxCompletionTokens = openai.Int(int64(cfg.MaxCompletionTokens))
	}

	trace := buildFusionTrace(cfg, groundedPanel, panel.failedModels, analysis, groundingMode, groundingScores)

	return &StreamingSynthesisPlan{
		SynthesisParams:   finalReq,
		TargetModel:       cfg.Model,
		IntermediateUsage: intermediateUsage,
		Trace:             trace,
		ModelsUsed:        modelsUsed,
		Iterations:        iterations,
		QuorumOutcome:     nil,
	}, nil
}

func (l *FusionLooper) applyStreamingQuorumFailurePolicy(
	ctx context.Context,
	req *Request,
	cfg fusionExecutionConfig,
	panel fusionPanelOutcome,
	panelErr error,
) (*StreamingSynthesisPlan, error) {
	var quorumErr *FusionQuorumError
	if !errors.As(panelErr, &quorumErr) {
		return nil, panelErr
	}
	evidence := quorumErr.Evidence()
	policy := string(resolvedQuorumFailurePolicy(cfg))

	if ctxErr := fusionCallerBudgetErr(ctx); ctxErr != nil {
		l.logQuorumDecision(req, cfg, evidence, FusionQuorumCancelled, "")
		return nil, quorumErr.WithPolicyOutcome(policy, "", FusionQuorumCancelled, evidence.Usage)
	}

	if cfg.QuorumFailurePolicy != config.FusionQuorumFailurePolicyFallback {
		l.logQuorumDecision(req, cfg, evidence, FusionQuorumFailed, "")
		return nil, quorumErr.WithPolicyOutcome(policy, "", FusionQuorumFailed, evidence.Usage)
	}

	target := cfg.QuorumFallbackTarget
	if target == "" {
		l.logQuorumDecision(req, cfg, evidence, FusionQuorumFailed, "")
		return nil, fmt.Errorf("fusion quorum fallback requested without a target: %w",
			quorumErr.WithPolicyOutcome(policy, "", FusionQuorumFailed, evidence.Usage))
	}

	fallbackReq := req.OriginalRequest
	if outputContract := requestOutputContract(req.OriginalRequest, req.OutputContract); outputContract != "" {
		fallbackReq = appendFusionStageMessage(req.OriginalRequest, outputContract)
	} else if fallbackReq != nil {
		data, err := json.Marshal(fallbackReq)
		if err != nil {
			return nil, fmt.Errorf("marshal fallback request: %w", err)
		}
		var cloned openai.ChatCompletionNewParams
		if err := json.Unmarshal(data, &cloned); err != nil {
			return nil, fmt.Errorf("unmarshal fallback request: %w", err)
		}
		fallbackReq = &cloned
	}

	if fallbackReq != nil {
		fallbackReq.Model = target
		if cfg.Temperature != nil {
			fallbackReq.Temperature = openai.Float(*cfg.Temperature)
		}
		if cfg.MaxCompletionTokens > 0 {
			fallbackReq.MaxTokens = (openai.ChatCompletionNewParams{}).MaxTokens
			fallbackReq.MaxCompletionTokens = openai.Int(int64(cfg.MaxCompletionTokens))
		}
	}

	modelsUsed := orderedFusionModelsUsed(panel.dispatchedModels, "")
	iterations := len(panel.dispatchedModels)
	evidence.Usage = panel.usage

	l.logQuorumDecision(req, cfg, evidence, FusionQuorumFallbackReady, target)
	quorumOutcome := newFusionQuorumOutcome(evidence, policy, target, FusionQuorumFallbackReady)

	return &StreamingSynthesisPlan{
		SynthesisParams:   fallbackReq,
		TargetModel:       target,
		IntermediateUsage: panel.usage,
		Trace:             nil,
		ModelsUsed:        modelsUsed,
		Iterations:        iterations,
		QuorumOutcome:     quorumOutcome,
	}, nil
}
