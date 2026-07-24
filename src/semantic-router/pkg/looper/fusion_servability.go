package looper

import (
	"context"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// This file holds Fusion "servability" concerns kept out of the core Execute
// flow in fusion.go: per-request/stage metrics, panel-agreement early-exit, and
// adaptive escalation (single-model fast path for easy queries).

// prepareFusionExecution resolves and validates the execution config for a
// fusion request, defaulting the judge model to the first analysis model.
func (l *FusionLooper) prepareFusionExecution(req *Request) (fusionExecutionConfig, error) {
	cfg := l.resolveFusionExecutionConfig(req)
	if len(cfg.AnalysisModels) == 0 {
		return cfg, fmt.Errorf("fusion analysis_models cannot be empty")
	}
	if cfg.Model == "" {
		cfg.Model = cfg.AnalysisModels[0]
	}
	if err := validateFusionExecutionConfig(cfg); err != nil {
		return cfg, err
	}
	if err := l.validateFusionModels(cfg); err != nil {
		return cfg, err
	}
	return cfg, nil
}

// recordFusionOutcome records the per-request count and duration for a fusion
// execution, classifying status from the (named-return) error pointer.
func recordFusionOutcome(decision string, start time.Time, errp *error) {
	status := "success"
	if errp != nil && *errp != nil {
		status = "error"
	}
	metrics.RecordFusionRequest(decision, status)
	metrics.RecordFusionRequestDuration(decision, time.Since(start).Seconds())
}

// recordFusionPanelMetrics records the success/failure outcome of each panel
// model call.
func recordFusionPanelMetrics(panelResponses []*ModelResponse, failedModels []FusionFailedModel) {
	for _, r := range panelResponses {
		metrics.RecordFusionPanelModel(r.Model, "success")
	}
	for _, f := range failedModels {
		metrics.RecordFusionPanelModel(f.Model, "failed")
	}
}

// recordFusionGroundingMetrics emits the per-response groundedness score
// distribution and the count of responses dropped by the grounding stage.
func recordFusionGroundingMetrics(cfg fusionExecutionConfig, referenceMode string, scores []groundingScore) {
	if len(scores) == 0 {
		return
	}
	dropped := 0
	for _, s := range scores {
		metrics.RecordFusionGroundingScore(referenceMode, cfg.GroundingPolicy, s.Score)
		if s.Dropped {
			dropped++
		}
	}
	metrics.RecordFusionGroundingDropped(cfg.GroundingPolicy, dropped)
}

// finalizeFusionResponse records token usage and formats the looper response
// (JSON or streaming). iterations is panel + synthesis, plus analysis unless the
// early-exit path skipped it.
func (l *FusionLooper) finalizeFusionResponse(
	req *Request,
	cfg fusionExecutionConfig,
	finalResp *ModelResponse,
	trace *FusionTrace,
	usage TokenUsage,
	earlyExit bool,
) (*Response, error) {
	metrics.RecordFusionRequestTokens(req.DecisionName, usage.PromptTokens, usage.CompletionTokens)
	modelsUsed := orderedFusionModelsUsed(cfg.AnalysisModels, cfg.Model)
	iterations := len(cfg.AnalysisModels) + 2
	if earlyExit {
		iterations = len(cfg.AnalysisModels) + 1
	}
	if req.IsStreaming {
		return l.formatFusionStreamingResponse(finalResp, modelsUsed, iterations, cfg, trace, usage)
	}
	return l.formatFusionJSONResponse(finalResp, modelsUsed, iterations, cfg, trace, usage)
}

// runFusionAnalysisStage runs the judge analysis pass unless panel-agreement
// early-exit applies, in which case it returns nil analysis and earlyExit=true so
// the judge synthesizes directly (one fewer judge call).
func (l *FusionLooper) runFusionAnalysisStage(
	ctx context.Context,
	req *Request,
	cfg fusionExecutionConfig,
	groundedPanel []*ModelResponse,
	groundingScores []groundingScore,
	groundingMode string,
) (*FusionAnalysis, *ModelResponse, bool) {
	if shouldFusionEarlyExit(cfg, groundingMode, groundingScores) {
		metrics.RecordFusionEarlyExit(req.DecisionName)
		logging.ComponentEvent("looper", "fusion_early_exit", map[string]interface{}{
			"decision":         req.DecisionName,
			"min_consistency":  cfg.GroundingEarlyExitMinConsistency,
			"panel":            len(groundedPanel),
			"skipped_analysis": true,
		})
		return nil, nil, true
	}
	analysisStart := time.Now()
	analysis, analysisResp := l.runFusionAnalysis(ctx, req, cfg, groundedPanel, groundingScores)
	metrics.RecordFusionStageDuration("analysis", time.Since(analysisStart).Seconds())
	return analysis, analysisResp, false
}

// shouldFusionEarlyExit reports whether the panel is unanimous enough to skip the
// analysis stage. It requires the feature to be enabled, panel-mode scoring (the
// only mode that measures cross-model agreement), at least two scored responses,
// and every response at or above the consistency threshold. The "every response"
// rule means a single dissenter (low score) keeps the full pipeline, so early-exit
// never trades away the minority view.
func shouldFusionEarlyExit(cfg fusionExecutionConfig, referenceMode string, scores []groundingScore) bool {
	if !cfg.GroundingEarlyExitEnabled || len(scores) < 2 {
		return false
	}
	if referenceMode != config.FusionGroundingReferencePanel {
		return false
	}
	for _, s := range scores {
		if s.Score < cfg.GroundingEarlyExitMinConsistency {
			return false
		}
	}
	return true
}

// shouldEscalateToSingleModel reports whether the request is easy enough to skip
// the panel and answer with one judge-model call. It fires only when escalation
// is enabled, this is not a cached-panel eval run, and none of the configured
// hard-complexity rules were matched for the request.
func shouldEscalateToSingleModel(cfg fusionExecutionConfig, req *Request) bool {
	if !cfg.EscalationEnabled || len(req.CachedPanel) > 0 {
		return false
	}
	for _, hard := range cfg.EscalationHardRules {
		for _, matched := range req.MatchedComplexity {
			if hard == matched {
				return false // a hard rule matched -> run the full panel
			}
		}
	}
	return true
}

// runFusionSingleModel answers with a single judge-model call (the escalation
// fast path for easy queries). It returns a normal fusion-shaped response so
// clients see a consistent contract; the trace records that the panel was skipped.
func (l *FusionLooper) runFusionSingleModel(ctx context.Context, req *Request, cfg fusionExecutionConfig) (*Response, error) {
	resp, err := l.callFusionModel(ctx, req, cfg, cfg.Model, true, false, 1)
	if err != nil {
		return nil, fmt.Errorf("fusion escalation single-model call failed for %q: %w", cfg.Model, err)
	}
	usage := SumUsage().Add(resp)
	trace := &FusionTrace{
		JudgeModel:    cfg.Model,
		PromptVersion: cfg.JudgePromptVersion,
	}
	modelsUsed := []string{cfg.Model}
	if req.IsStreaming {
		return l.formatFusionStreamingResponse(resp, modelsUsed, 1, cfg, trace, usage)
	}
	return l.formatFusionJSONResponse(resp, modelsUsed, 1, cfg, trace, usage)
}

// mergeFusionEscalationConfig merges the adaptive-escalation overrides.
func mergeFusionEscalationConfig(dst *fusionExecutionConfig, src *config.FusionEscalationConfig) {
	if src == nil {
		return
	}
	dst.EscalationEnabled = src.Enabled
	dst.EscalationHardRules = append([]string(nil), src.HardComplexityRules...)
}
