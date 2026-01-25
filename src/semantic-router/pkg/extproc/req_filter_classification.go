package extproc

import (
	"context"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// performDecisionEvaluation performs decision evaluation using DecisionEngine
// Returns (decisionName, confidence, reasoningDecision, selectedModel)
// This is the new approach that uses Decision-based routing with AND/OR rule combinations
// Decision evaluation is ALWAYS performed when decisions are configured (for plugin features like
// hallucination detection), but model selection only happens for auto models.
func (r *OpenAIRouter) performDecisionEvaluation(originalModel string, userContent string, nonUserMessages []string, ctx *RequestContext) (string, float64, entropy.ReasoningDecision, string) {
	var decisionName string
	var evaluationConfidence float64
	var reasoningDecision entropy.ReasoningDecision
	var selectedModel string

	// Check if there's content to evaluate
	if len(nonUserMessages) == 0 && userContent == "" {
		return "", 0.0, entropy.ReasoningDecision{}, ""
	}

	// Check if decisions are configured
	if len(r.Config.Decisions) == 0 {
		if r.Config.IsAutoModelName(originalModel) {
			logging.Warnf("No decisions configured, using default model")
			return "", 0.0, entropy.ReasoningDecision{}, r.Config.DefaultModel
		}
		return "", 0.0, entropy.ReasoningDecision{}, ""
	}

	// Determine text to use for evaluation
	evaluationText := userContent
	if evaluationText == "" && len(nonUserMessages) > 0 {
		evaluationText = strings.Join(nonUserMessages, " ")
	}

	if evaluationText == "" {
		return "", 0.0, entropy.ReasoningDecision{}, ""
	}

	// For context token counting, we need to include ALL messages (user + non-user)
	// This ensures multi-turn conversations are properly counted
	var allMessagesText string
	if userContent != "" && len(nonUserMessages) > 0 {
		// Combine user content with all non-user messages for full context
		allMessages := make([]string, 0, len(nonUserMessages)+1)
		allMessages = append(allMessages, nonUserMessages...)
		allMessages = append(allMessages, userContent)
		allMessagesText = strings.Join(allMessages, " ")
	} else if userContent != "" {
		allMessagesText = userContent
	} else {
		allMessagesText = strings.Join(nonUserMessages, " ")
	}

	// Start signal evaluation span (Layer 1)
	signalStart := time.Now()
	signalCtx, signalSpan := tracing.StartSpan(ctx.TraceContext, tracing.SpanSignalEvaluation)

	// Evaluate all signals first to get detailed signal information
	// Use evaluationText for most signals, but pass allMessagesText for context counting
	signals := r.Classifier.EvaluateAllSignalsWithContext(evaluationText, allMessagesText)

	signalLatency := time.Since(signalStart).Milliseconds()

	// Store signal results in context for response headers
	ctx.VSRMatchedKeywords = signals.MatchedKeywordRules
	ctx.VSRMatchedEmbeddings = signals.MatchedEmbeddingRules
	ctx.VSRMatchedDomains = signals.MatchedDomainRules
	ctx.VSRMatchedFactCheck = signals.MatchedFactCheckRules
	ctx.VSRMatchedUserFeedback = signals.MatchedUserFeedbackRules
	ctx.VSRMatchedPreference = signals.MatchedPreferenceRules
	ctx.VSRMatchedLanguage = signals.MatchedLanguageRules
	ctx.VSRMatchedLatency = signals.MatchedLatencyRules
	ctx.VSRMatchedContext = signals.MatchedContextRules
	ctx.VSRContextTokenCount = signals.TokenCount

	// Set fact-check context fields from signal results
	// This replaces the old performFactCheckClassification call to avoid duplicate computation
	r.setFactCheckFromSignals(ctx, signals.MatchedFactCheckRules)

	// Log signal evaluation results
	logging.Infof("Signal evaluation results: keyword=%v, embedding=%v, domain=%v, fact_check=%v, user_feedback=%v, preference=%v, language=%v, latency=%v",
		signals.MatchedKeywordRules, signals.MatchedEmbeddingRules, signals.MatchedDomainRules,
		signals.MatchedFactCheckRules, signals.MatchedUserFeedbackRules, signals.MatchedPreferenceRules, signals.MatchedLanguageRules, signals.MatchedLatencyRules)

	// Set signal span attributes
	allMatchedRules := []string{}
	allMatchedRules = append(allMatchedRules, signals.MatchedKeywordRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedEmbeddingRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedDomainRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedFactCheckRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedUserFeedbackRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedPreferenceRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedLanguageRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedLatencyRules...)

	// End signal evaluation span
	tracing.EndSignalSpan(signalSpan, allMatchedRules, 1.0, signalLatency)
	ctx.TraceContext = signalCtx

	// Process user feedback signals to automatically update Elo ratings
	// This implements "automatic scoring by signals" as requested
	r.processUserFeedbackForElo(signals.MatchedUserFeedbackRules, originalModel, ctx)

	// Perform decision evaluation using pre-computed signals
	// This is ALWAYS done when decisions are configured, regardless of model type,
	// because plugins (e.g., hallucination detection) depend on the matched decision

	// Start decision evaluation span (Layer 2)
	decisionStart := time.Now()
	decisionCtx, decisionSpan := tracing.StartDecisionSpan(ctx.TraceContext, "decision_evaluation")

	result, err := r.Classifier.EvaluateDecisionWithEngine(signals)
	decisionLatency := time.Since(decisionStart).Seconds()

	// Record decision evaluation metrics
	metrics.RecordDecisionEvaluation(decisionLatency)

	if err != nil {
		logging.Errorf("Decision evaluation error: %v", err)
		// End decision span with error
		tracing.EndDecisionSpan(decisionSpan, 0.0, []string{}, r.Config.Strategy)
		ctx.TraceContext = decisionCtx
		if r.Config.IsAutoModelName(originalModel) {
			return "", 0.0, entropy.ReasoningDecision{}, r.Config.DefaultModel
		}
		return "", 0.0, entropy.ReasoningDecision{}, ""
	}

	if result == nil || result.Decision == nil {
		// End decision span with no match
		tracing.EndDecisionSpan(decisionSpan, 0.0, []string{}, r.Config.Strategy)
		ctx.TraceContext = decisionCtx
		if r.Config.IsAutoModelName(originalModel) {
			return "", 0.0, entropy.ReasoningDecision{}, r.Config.DefaultModel
		}
		return "", 0.0, entropy.ReasoningDecision{}, ""
	}

	// Record decision match with confidence
	metrics.RecordDecisionMatch(result.Decision.Name, result.Confidence)

	// End decision span with success
	tracing.EndDecisionSpan(decisionSpan, result.Confidence, result.MatchedRules, r.Config.Strategy)
	ctx.TraceContext = decisionCtx

	// Store the selected decision in context for later use (e.g., plugins, header mutations)
	// This is critical for hallucination detection and other per-decision plugins
	ctx.VSRSelectedDecision = result.Decision

	// Set router replay config from system-level configuration if enabled
	if r.Config.RouterReplay.Enabled {
		cfgCopy := r.Config.RouterReplay
		ctx.RouterReplayConfig = &cfgCopy
	}

	// Extract domain category from matched rules (for VSRSelectedCategory header)
	// MatchedRules contains rule names like "domain:math", "keyword:thinking", etc.
	// We extract the first domain rule as the category
	categoryName := ""
	for _, rule := range result.MatchedRules {
		if strings.HasPrefix(rule, "domain:") {
			categoryName = strings.TrimPrefix(rule, "domain:")
			break
		}
	}
	// Store category in context for response headers
	ctx.VSRSelectedCategory = categoryName
	ctx.VSRSelectedDecisionConfidence = evaluationConfidence

	// Note: VSRMatchedKeywords is already set from signals.MatchedKeywordRules (line 61)
	// We should NOT overwrite it with result.MatchedKeywords which contains actual keywords
	// The header should show rule names, not the actual matched keywords

	decisionName = result.Decision.Name
	evaluationConfidence = result.Confidence
	logging.Infof("Decision Evaluation Result: decision=%s, category=%s, confidence=%.3f, matched_rules=%v",
		decisionName, categoryName, evaluationConfidence, result.MatchedRules)

	// Model selection only happens for auto models
	// When a specific model is requested, we keep it but still apply decision plugins
	if !r.Config.IsAutoModelName(originalModel) {
		logging.Infof("Model %s explicitly specified, keeping original model (decision %s plugins will be applied)",
			originalModel, decisionName)
		return decisionName, evaluationConfidence, reasoningDecision, ""
	}

	// Select best model from the decision's ModelRefs using configured selection algorithm
	if len(result.Decision.ModelRefs) > 0 {
		// Use advanced model selection (Elo, RouterDC, AutoMix, Hybrid, or Static)
		// Pass decision's algorithm config for per-decision algorithm override
		selectedModelRef, usedMethod := r.selectModelFromCandidates(result.Decision.ModelRefs, decisionName, userContent, result.Decision.Algorithm)

		// Use LoRA name if specified, otherwise use the base model name
		selectedModel = selectedModelRef.Model
		if selectedModelRef.LoRAName != "" {
			selectedModel = selectedModelRef.LoRAName
			logging.Infof("Selected model from decision %s: %s (LoRA adapter for base model %s) using %s selection",
				decisionName, selectedModel, selectedModelRef.Model, usedMethod)
		} else {
			logging.Infof("Selected model from decision %s: %s using %s selection",
				decisionName, selectedModel, usedMethod)
		}
		ctx.VSRSelectedModel = selectedModel

		// Determine reasoning mode from the selected model's configuration
		if selectedModelRef.UseReasoning != nil {
			useReasoning := *selectedModelRef.UseReasoning
			reasoningDecision = entropy.ReasoningDecision{
				UseReasoning:     useReasoning,
				Confidence:       evaluationConfidence,
				DecisionReason:   "decision_engine_evaluation",
				FallbackStrategy: "decision_based_routing",
				TopCategories: []entropy.CategoryProbability{
					{
						Category:    decisionName,
						Probability: float32(evaluationConfidence),
					},
				},
			}
			if useReasoning {
				ctx.VSRReasoningMode = "on"
			} else {
				ctx.VSRReasoningMode = "off"
			}
			// Note: ReasoningEffort is handled separately in req_filter_reason.go
		}
	} else {
		// No model refs in decision, use default model
		selectedModel = r.Config.DefaultModel
		logging.Infof("No model refs in decision %s, using default model: %s", decisionName, selectedModel)
	}

	return decisionName, evaluationConfidence, reasoningDecision, selectedModel
}

// selectModelFromCandidates uses the configured selection algorithm to choose the best model
// from the decision's candidate models. Falls back to first model if selection fails.
// The algorithm parameter allows per-decision algorithm override (aligned with looper pattern).
// Returns the selected model and the method name used for logging.
func (r *OpenAIRouter) selectModelFromCandidates(modelRefs []config.ModelRef, decisionName string, query string, algorithm *config.AlgorithmConfig) (*config.ModelRef, string) {
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

	selCtx := &selection.SelectionContext{
		Query:           query,
		DecisionName:    decisionName,
		CandidateModels: modelRefs,
		CostWeight:      costWeight,
		QualityWeight:   qualityWeight,
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
	// Check per-decision algorithm (aligned with looper pattern)
	if algorithm != nil && algorithm.Type != "" {
		switch algorithm.Type {
		case "elo":
			return selection.MethodElo
		case "router_dc":
			return selection.MethodRouterDC
		case "automix":
			return selection.MethodAutoMix
		case "hybrid":
			return selection.MethodHybrid
		case "static":
			return selection.MethodStatic
		case "confidence", "ratings":
			// These are looper algorithms, not selection algorithms
			// Fall through to default
		}
	}

	// Default to static selection (use first model)
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
