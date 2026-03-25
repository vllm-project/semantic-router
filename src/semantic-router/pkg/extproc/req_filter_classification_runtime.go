package extproc

import (
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

var selectionMethodByAlgorithmType = map[string]selection.SelectionMethod{
	"elo":           selection.MethodElo,
	"router_dc":     selection.MethodRouterDC,
	"automix":       selection.MethodAutoMix,
	"hybrid":        selection.MethodHybrid,
	"rl_driven":     selection.MethodRLDriven,
	"gmtrouter":     selection.MethodGMTRouter,
	"latency_aware": selection.MethodLatencyAware,
	"static":        selection.MethodStatic,
	"knn":           selection.MethodKNN,
	"kmeans":        selection.MethodKMeans,
	"svm":           selection.MethodSVM,
}

func (r *OpenAIRouter) evaluateSignalsForDecision(
	originalModel string,
	signalInput signalEvaluationInput,
	nonUserMessages []string,
	ctx *RequestContext,
) (*classification.SignalResults, error) {
	signalStart := time.Now()
	signalCtx, signalSpan := tracing.StartSpan(ctx.TraceContext, tracing.SpanSignalEvaluation)

	signals, authzErr := r.Classifier.EvaluateAllSignalsWithHeaders(
		signalInput.compressedText,
		signalInput.allMessagesText,
		nonUserMessages,
		ctx.Headers,
		false,
		ctx.RequestImageURL,
		signalInput.evaluationText,
		signalInput.skipCompressionSignals,
	)
	if authzErr != nil {
		signalSpan.End()
		logging.Errorf("[Signal Evaluation] Authz evaluation failed: %v", authzErr)
		return nil, authzErr
	}

	signalLatency := time.Since(signalStart).Milliseconds()
	r.applySignalResultsToContext(ctx, signals)
	logSignalEvaluationResults(signals)
	tracing.EndSignalSpan(signalSpan, collectMatchedSignalRules(signals), 1.0, signalLatency)
	ctx.TraceContext = signalCtx
	r.processUserFeedbackForElo(signals.MatchedUserFeedbackRules, originalModel, ctx)
	return signals, nil
}

func logSignalEvaluationResults(signals *classification.SignalResults) {
	logging.Infof("Signal evaluation results: keyword=%v, embedding=%v, domain=%v, fact_check=%v, user_feedback=%v, preference=%v, language=%v, context=%v, complexity=%v, modality=%v, authz=%v, jailbreak=%v, pii=%v, projection=%v",
		signals.MatchedKeywordRules,
		signals.MatchedEmbeddingRules,
		signals.MatchedDomainRules,
		signals.MatchedFactCheckRules,
		signals.MatchedUserFeedbackRules,
		signals.MatchedPreferenceRules,
		signals.MatchedLanguageRules,
		signals.MatchedContextRules,
		signals.MatchedComplexityRules,
		signals.MatchedModalityRules,
		signals.MatchedAuthzRules,
		signals.MatchedJailbreakRules,
		signals.MatchedPIIRules,
		signals.MatchedProjectionRules)
}

func (r *OpenAIRouter) runDecisionEngine(
	originalModel string,
	ctx *RequestContext,
	signals *classification.SignalResults,
) (*decision.DecisionResult, string) {
	decisionStart := time.Now()
	decisionCtx, decisionSpan := tracing.StartDecisionSpan(ctx.TraceContext, "decision_evaluation")

	result, err := r.Classifier.EvaluateDecisionWithEngine(signals)
	decisionLatency := time.Since(decisionStart).Seconds()
	metrics.RecordDecisionEvaluation(decisionLatency)

	if err != nil {
		logging.Errorf("Decision evaluation error: %v", err)
		tracing.EndDecisionSpan(decisionSpan, 0.0, []string{}, r.Config.Strategy)
		ctx.TraceContext = decisionCtx
		return nil, r.defaultDecisionFallbackModel(originalModel)
	}
	if result == nil || result.Decision == nil {
		tracing.EndDecisionSpan(decisionSpan, 0.0, []string{}, r.Config.Strategy)
		ctx.TraceContext = decisionCtx
		return nil, r.defaultDecisionFallbackModel(originalModel)
	}

	metrics.RecordDecisionMatch(result.Decision.Name, result.Confidence)
	tracing.EndDecisionSpan(decisionSpan, result.Confidence, result.MatchedRules, r.Config.Strategy)
	ctx.TraceContext = decisionCtx
	return result, ""
}

func (r *OpenAIRouter) defaultDecisionFallbackModel(originalModel string) string {
	if r.Config.IsAutoModelName(originalModel) {
		return r.Config.DefaultModel
	}
	return ""
}

func (r *OpenAIRouter) finalizeDecisionEvaluation(
	result *decision.DecisionResult,
	originalModel string,
	userContent string,
	ctx *RequestContext,
) (string, float64, entropy.ReasoningDecision, string) {
	reasoningDecision := entropy.ReasoningDecision{}
	categoryName := r.applyDecisionResultToContext(result, ctx)
	decisionName := result.Decision.Name
	evaluationConfidence := result.Confidence

	ctx.VSRSelectedDecisionConfidence = evaluationConfidence
	logging.Infof("Decision Evaluation Result: decision=%s, category=%s, confidence=%.3f, matched_rules=%v",
		decisionName, categoryName, evaluationConfidence, result.MatchedRules)

	if !r.Config.IsAutoModelName(originalModel) {
		logging.Infof("Model %s explicitly specified, keeping original model (decision %s plugins will be applied)",
			originalModel, decisionName)
		return decisionName, evaluationConfidence, reasoningDecision, ""
	}

	selectedModel, reasoningDecision := r.selectDecisionRuntimeModel(
		result,
		decisionName,
		userContent,
		categoryName,
		evaluationConfidence,
		ctx,
	)
	return decisionName, evaluationConfidence, reasoningDecision, selectedModel
}

func (r *OpenAIRouter) applyDecisionResultToContext(result *decision.DecisionResult, ctx *RequestContext) string {
	ctx.VSRSelectedDecision = result.Decision
	if pluginCfg := result.Decision.GetRouterReplayConfig(); pluginCfg != nil && pluginCfg.Enabled {
		ctx.RouterReplayPluginConfig = pluginCfg
	}

	categoryName := extractDecisionCategory(result.MatchedRules)
	ctx.VSRSelectedCategory = categoryName
	return categoryName
}

func extractDecisionCategory(matchedRules []string) string {
	for _, rule := range matchedRules {
		if strings.HasPrefix(rule, "domain:") {
			return strings.TrimPrefix(rule, "domain:")
		}
	}
	return ""
}

func (r *OpenAIRouter) selectDecisionRuntimeModel(
	result *decision.DecisionResult,
	decisionName string,
	userContent string,
	categoryName string,
	evaluationConfidence float64,
	ctx *RequestContext,
) (string, entropy.ReasoningDecision) {
	if len(result.Decision.ModelRefs) == 0 {
		selectedModel := r.Config.DefaultModel
		ctx.VSRSelectedModel = selectedModel
		ctx.VSRSelectionMethod = "default"
		logging.Infof("No model refs in decision %s, using default model: %s", decisionName, selectedModel)
		return selectedModel, entropy.ReasoningDecision{}
	}

	selectedModelRef, usedMethod := r.selectModelFromCandidates(
		result.Decision.ModelRefs,
		decisionName,
		userContent,
		result.Decision.Algorithm,
		categoryName,
	)
	selectedModel := selectedModelRef.Model
	if selectedModelRef.LoRAName != "" {
		selectedModel = selectedModelRef.LoRAName
		logging.Infof("Selected model from decision %s: %s (LoRA adapter for base model %s) using %s selection",
			decisionName, selectedModel, selectedModelRef.Model, usedMethod)
	} else {
		logging.Infof("Selected model from decision %s: %s using %s selection",
			decisionName, selectedModel, usedMethod)
	}
	ctx.VSRSelectedModel = selectedModel
	ctx.VSRSelectionMethod = usedMethod
	return selectedModel, applyReasoningModeFromSelectedModel(selectedModelRef, decisionName, evaluationConfidence, ctx)
}

func applyReasoningModeFromSelectedModel(
	selectedModelRef *config.ModelRef,
	decisionName string,
	evaluationConfidence float64,
	ctx *RequestContext,
) entropy.ReasoningDecision {
	if selectedModelRef.UseReasoning == nil {
		return entropy.ReasoningDecision{}
	}

	useReasoning := *selectedModelRef.UseReasoning
	if useReasoning {
		ctx.VSRReasoningMode = "on"
	} else {
		ctx.VSRReasoningMode = "off"
	}

	return entropy.ReasoningDecision{
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
}
