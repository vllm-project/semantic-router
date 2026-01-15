package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
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

	// Evaluate all signals first to get detailed signal information
	signals := r.Classifier.EvaluateAllSignals(evaluationText)

	// Store signal results in context for response headers
	ctx.VSRMatchedKeywords = signals.MatchedKeywordRules
	ctx.VSRMatchedEmbeddings = signals.MatchedEmbeddingRules
	ctx.VSRMatchedDomains = signals.MatchedDomainRules
	ctx.VSRMatchedFactCheck = signals.MatchedFactCheckRules
	ctx.VSRMatchedUserFeedback = signals.MatchedUserFeedbackRules
	ctx.VSRMatchedPreference = signals.MatchedPreferenceRules

	// Set fact-check context fields from signal results
	// This replaces the old performFactCheckClassification call to avoid duplicate computation
	r.setFactCheckFromSignals(ctx, signals.MatchedFactCheckRules)

	// Log signal evaluation results
	logging.Infof("Signal evaluation results: keyword=%v, embedding=%v, domain=%v, fact_check=%v, user_feedback=%v, preference=%v",
		signals.MatchedKeywordRules, signals.MatchedEmbeddingRules, signals.MatchedDomainRules,
		signals.MatchedFactCheckRules, signals.MatchedUserFeedbackRules, signals.MatchedPreferenceRules)

	// Perform decision evaluation using pre-computed signals
	// This is ALWAYS done when decisions are configured, regardless of model type,
	// because plugins (e.g., hallucination detection) depend on the matched decision
	result, err := r.Classifier.EvaluateDecisionWithEngine(signals)
	if err != nil {
		logging.Errorf("Decision evaluation error: %v", err)
		if r.Config.IsAutoModelName(originalModel) {
			return "", 0.0, entropy.ReasoningDecision{}, r.Config.DefaultModel
		}
		return "", 0.0, entropy.ReasoningDecision{}, ""
	}

	if result == nil || result.Decision == nil {
		if r.Config.IsAutoModelName(originalModel) {
			return "", 0.0, entropy.ReasoningDecision{}, r.Config.DefaultModel
		}
		return "", 0.0, entropy.ReasoningDecision{}, ""
	}

	// Store the selected decision in context for later use (e.g., plugins, header mutations)
	// This is critical for hallucination detection and other per-decision plugins
	ctx.VSRSelectedDecision = result.Decision

	if replayCfg := result.Decision.GetRouterReplayConfig(); replayCfg != nil && replayCfg.Enabled {
		cfgCopy := *replayCfg
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

	// Store matched keywords in context for response headers
	ctx.VSRMatchedKeywords = result.MatchedKeywords

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

	// Select best model from the decision's ModelRefs (only for auto models)
	if len(result.Decision.ModelRefs) > 0 {
		modelRef := result.Decision.ModelRefs[0]
		// Use LoRA name if specified, otherwise use the base model name
		selectedModel = modelRef.Model
		if modelRef.LoRAName != "" {
			selectedModel = modelRef.LoRAName
			logging.Infof("Selected model from decision %s: %s (LoRA adapter for base model %s)",
				decisionName, selectedModel, modelRef.Model)
		} else {
			logging.Infof("Selected model from decision %s: %s", decisionName, selectedModel)
		}
		ctx.VSRSelectedModel = selectedModel

		// Determine reasoning mode from the best model's configuration
		if result.Decision.ModelRefs[0].UseReasoning != nil {
			useReasoning := *result.Decision.ModelRefs[0].UseReasoning
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
