package extproc

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// performDecisionEvaluation performs decision evaluation using DecisionEngine
// Returns (decisionName, confidence, reasoningDecision, selectedModel)
// This is the new approach that uses Decision-based routing with AND/OR rule combinations
// Decision evaluation is ALWAYS performed when decisions are configured (for plugin features like
// hallucination detection), but model selection only happens for auto models.
func (r *OpenAIRouter) performDecisionEvaluation(originalModel string, history signalConversationHistory, ctx *RequestContext) (string, float64, entropy.ReasoningDecision, string, error) {
	var decisionName string
	var evaluationConfidence float64
	var reasoningDecision entropy.ReasoningDecision
	var selectedModel string

	// Check if there's content to evaluate
	if len(history.nonUserMessages) == 0 && history.currentUserMessage == "" {
		return "", 0.0, entropy.ReasoningDecision{}, "", nil
	}

	// Check if decisions are configured
	if len(r.Config.Decisions) == 0 {
		if r.Config.IsAutoModelName(originalModel) {
			logging.Warnf("No decisions configured, using default model")
			return "", 0.0, entropy.ReasoningDecision{}, r.Config.DefaultModel, nil
		}
		return "", 0.0, entropy.ReasoningDecision{}, "", nil
	}

	signalInput := r.prepareSignalEvaluationInput(history)
	if signalInput.evaluationText == "" {
		return "", 0.0, entropy.ReasoningDecision{}, "", nil
	}

	signals, authzErr := r.evaluateSignalsForDecision(originalModel, signalInput, history.nonUserMessages, ctx)
	if authzErr != nil {
		return "", 0, entropy.ReasoningDecision{}, "", authzErr
	}

	result, fallbackModel := r.runDecisionEngine(originalModel, ctx, signals)
	if result == nil {
		return "", 0.0, entropy.ReasoningDecision{}, fallbackModel, nil
	}

	decisionName, evaluationConfidence, reasoningDecision, selectedModel = r.finalizeDecisionEvaluation(
		result,
		originalModel,
		history.currentUserMessage,
		ctx,
	)
	return decisionName, evaluationConfidence, reasoningDecision, selectedModel, nil
}

// selectModelFromCandidates uses the configured selection algorithm to choose the best model
// from the decision's candidate models. Falls back to first model if selection fails.
// The algorithm parameter allows per-decision algorithm override (aligned with looper pattern).
// The categoryName parameter is the detected domain category (e.g., "physics", "math") for ML feature vectors.
// Returns the selected model and the method name used for logging.
func (r *OpenAIRouter) selectModelFromCandidates(modelRefs []config.ModelRef, decisionName string, query string, algorithm *config.AlgorithmConfig, categoryName string) (*config.ModelRef, string) {
	if len(modelRefs) == 0 {
		return nil, ""
	}

	// If only one model, no need for selection algorithm
	if len(modelRefs) == 1 {
		return &modelRefs[0], "single"
	}

	// Determine selection method: per-decision algorithm takes precedence over global config
	method := r.getSelectionMethod(algorithm)

	// Get selector from registry
	var selector selection.Selector
	if r.ModelSelector != nil {
		selector, _ = r.ModelSelector.Get(method)
	}

	// Fallback to first model if no selector available
	if selector == nil {
		logging.Warnf("[ModelSelection] No selector available for method %s, using first model", method)
		return &modelRefs[0], string(method)
	}

	// Build selection context with cost/quality weights
	costWeight, qualityWeight := r.getSelectionWeights(algorithm)
	latencyAwareTPOTPercentile, latencyAwareTTFTPercentile := r.getLatencyAwarePercentiles(algorithm)

	selCtx := &selection.SelectionContext{
		Query:                      query,
		DecisionName:               decisionName,
		CategoryName:               categoryName,
		CandidateModels:            modelRefs,
		CostWeight:                 costWeight,
		QualityWeight:              qualityWeight,
		LatencyAwareTPOTPercentile: latencyAwareTPOTPercentile,
		LatencyAwareTTFTPercentile: latencyAwareTTFTPercentile,
	}

	// Perform selection
	result, err := selector.Select(context.Background(), selCtx)
	if err != nil {
		logging.Warnf("[ModelSelection] Selection failed: %v, using first model", err)
		return &modelRefs[0], string(method)
	}

	// Find the selected model in the candidates
	for i := range modelRefs {
		if modelRefs[i].Model == result.SelectedModel ||
			modelRefs[i].LoRAName == result.SelectedModel {
			logging.Infof("[ModelSelection] Selected %s (method=%s, score=%.4f, confidence=%.2f): %s",
				result.SelectedModel, method, result.Score, result.Confidence, result.Reasoning)
			// Record selection metrics
			selection.RecordSelection(string(method), decisionName, result.SelectedModel, result.Score)
			return &modelRefs[i], string(method)
		}
	}

	// Fallback if selected model not found in candidates (shouldn't happen)
	logging.Warnf("[ModelSelection] Selected model %s not found in candidates, using first model", result.SelectedModel)
	return &modelRefs[0], string(method)
}

// getSelectionMethod determines which selection algorithm to use.
// Per-decision algorithm is the primary configuration (aligned with looper pattern).
// Defaults to static selection if no algorithm is specified.
func (r *OpenAIRouter) getSelectionMethod(algorithm *config.AlgorithmConfig) selection.SelectionMethod {
	if algorithm != nil && algorithm.Type != "" {
		if method, ok := selectionMethodByAlgorithmType[algorithm.Type]; ok {
			return method
		}
	}
	return selection.MethodStatic
}

// getSelectionWeights returns cost and quality weights based on algorithm config.
// Uses per-decision config only (aligned with looper pattern).
func (r *OpenAIRouter) getSelectionWeights(algorithm *config.AlgorithmConfig) (float64, float64) {
	// Check per-decision algorithm config
	if algorithm != nil {
		if algorithm.AutoMix != nil && algorithm.AutoMix.CostQualityTradeoff > 0 {
			cost := algorithm.AutoMix.CostQualityTradeoff
			return cost, 1.0 - cost
		}
		if algorithm.Hybrid != nil && algorithm.Hybrid.CostWeight > 0 {
			cost := algorithm.Hybrid.CostWeight
			return cost, 1.0 - cost
		}
	}

	// Default: equal weighting (0.5 cost, 0.5 quality)
	return 0.5, 0.5
}

// getLatencyAwarePercentiles extracts TPOT/TTFT percentile settings for latency_aware selection.
// Returns (0, 0) when latency_aware is not configured for the decision.
func (r *OpenAIRouter) getLatencyAwarePercentiles(algorithm *config.AlgorithmConfig) (int, int) {
	if algorithm == nil || algorithm.LatencyAware == nil {
		return 0, 0
	}
	return algorithm.LatencyAware.TPOTPercentile, algorithm.LatencyAware.TTFTPercentile
}

// processUserFeedbackForElo automatically updates Elo ratings based on detected user feedback signals.
// This implements "automatic scoring by signals" - when the FeedbackDetector classifies user
// follow-up messages as "satisfied" or "wrong_answer", we automatically update Elo ratings.
//
// Signal mapping:
// - "satisfied" → Model performed well, record as implicit win
// - "wrong_answer" → Model performed poorly, record as implicit loss
// - "need_clarification" / "want_different" → Neutral, no Elo update
//
// For single-model feedback (no comparison), we use a "virtual opponent" approach:
// - The model competes against an expected baseline (rating 1500)
// - "satisfied" = win against baseline
// - "wrong_answer" = loss against baseline
func (r *OpenAIRouter) processUserFeedbackForElo(userFeedbackSignals []string, model string, ctx *RequestContext) {
	if len(userFeedbackSignals) == 0 || model == "" {
		return
	}

	// Get Elo selector from registry
	if r.ModelSelector == nil {
		return
	}

	eloSelector, ok := r.ModelSelector.Get(selection.MethodElo)
	if !ok || eloSelector == nil {
		return
	}

	// Process each feedback signal
	// Get decision name safely
	decisionName := ""
	if ctx.VSRSelectedDecision != nil {
		decisionName = ctx.VSRSelectedDecision.Name
	}

	for _, signal := range userFeedbackSignals {
		var feedback *selection.Feedback

		switch signal {
		case "satisfied":
			// Model performed well - record as win against virtual baseline
			feedback = &selection.Feedback{
				Query:        ctx.RequestQuery,
				WinnerModel:  model,
				LoserModel:   "", // Empty = self-feedback mode
				DecisionName: decisionName,
				Tie:          false,
			}
			logging.Infof("[AutoFeedback] User satisfied with %s, recording positive Elo feedback", model)

		case "wrong_answer":
			// Model performed poorly - record as loss against virtual baseline
			feedback = &selection.Feedback{
				Query:        ctx.RequestQuery,
				WinnerModel:  "", // Empty = model loses
				LoserModel:   model,
				DecisionName: decisionName,
				Tie:          false,
			}
			logging.Infof("[AutoFeedback] User reported wrong answer from %s, recording negative Elo feedback", model)

		default:
			// "need_clarification" and "want_different" are neutral - no Elo update
			logging.Debugf("[AutoFeedback] Neutral feedback signal %s, no Elo update", signal)
			continue
		}

		// Submit feedback to Elo selector
		if err := eloSelector.UpdateFeedback(context.Background(), feedback); err != nil {
			logging.Warnf("[AutoFeedback] Failed to update Elo: %v", err)
		}
	}
}
