package extproc

import (
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

var selectionMethodByAlgorithmType = map[string]selection.SelectionMethod{
	"router_dc":     selection.MethodRouterDC,
	"automix":       selection.MethodAutoMix,
	"hybrid":        selection.MethodHybrid,
	"latency_aware": selection.MethodLatencyAware,
	"static":        selection.MethodStatic,
	"knn":           selection.MethodKNN,
	"kmeans":        selection.MethodKMeans,
	"svm":           selection.MethodSVM,
	"multi_factor":  selection.MethodMultiFactor,
	"mlp":           selection.MethodMLP,
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
		signalInput.currentUserText,
		signalInput.priorUserMessages,
		nonUserMessages,
		signalInput.hasAssistantReply,
		ctx.Headers,
		false,
		ctx.RequestImageURL,
		signalInput.evaluationText,
		signalInput.skipCompressionSignals,
		signalInput.conversationFacts,
	)
	if authzErr != nil {
		signalSpan.End()
		logging.ComponentErrorEvent("extproc", "signal_evaluation_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"stage":      "authz",
			"error":      authzErr.Error(),
		})
		return nil, authzErr
	}

	signalLatency := time.Since(signalStart).Milliseconds()
	r.applySignalResultsToContext(ctx, signals)
	ensureContextTokenCount(ctx, signalInput)
	logSignalEvaluationResults(ctx, signalLatency, signals)
	tracing.EndSignalSpan(signalSpan, collectMatchedSignalRules(signals), 1.0, signalLatency)
	ctx.TraceContext = signalCtx
	return signals, nil
}

func ensureContextTokenCount(ctx *RequestContext, signalInput signalEvaluationInput) {
	if ctx == nil {
		return
	}
	text := contextTokenText(signalInput)
	if text == "" {
		return
	}
	if ctx.VSRContextTextBytes <= 0 {
		ctx.VSRContextTextBytes = len(text)
	}
	if ctx.VSRContextTokenCount > 0 {
		return
	}
	counter := classification.CharacterBasedTokenCounter{}
	count, err := counter.CountTokens(text)
	if err != nil || count <= 0 {
		return
	}
	ctx.VSRContextTokenCount = count
}

func contextTokenText(signalInput signalEvaluationInput) string {
	text := strings.TrimSpace(signalInput.allMessagesText)
	if text == "" {
		text = strings.TrimSpace(signalInput.evaluationText)
	}
	return text
}

func logSignalEvaluationResults(ctx *RequestContext, signalLatencyMs int64, signals *classification.SignalResults) {
	logging.ComponentDebugEvent("extproc", "signal_evaluation_complete", map[string]interface{}{
		"request_id":     ctx.RequestID,
		"latency_ms":     signalLatencyMs,
		"keyword":        signals.MatchedKeywordRules,
		"embedding":      signals.MatchedEmbeddingRules,
		"domain":         signals.MatchedDomainRules,
		"fact_check":     signals.MatchedFactCheckRules,
		"user_feedback":  signals.MatchedUserFeedbackRules,
		"reask":          signals.MatchedReaskRules,
		"preference":     signals.MatchedPreferenceRules,
		"language":       signals.MatchedLanguageRules,
		"context":        signals.MatchedContextRules,
		"structure":      signals.MatchedStructureRules,
		"complexity":     signals.MatchedComplexityRules,
		"modality":       signals.MatchedModalityRules,
		"authz":          signals.MatchedAuthzRules,
		"jailbreak":      signals.MatchedJailbreakRules,
		"pii":            signals.MatchedPIIRules,
		"kb":             signals.MatchedKBRules,
		"conversation":   signals.MatchedConversationRules,
		"event":          signals.MatchedEventRules,
		"projection":     signals.MatchedProjectionRules,
		"context_tokens": signals.TokenCount,
	})
}

func (r *OpenAIRouter) runDecisionEngine(
	originalModel string,
	ctx *RequestContext,
	signals *classification.SignalResults,
	candidates []config.Decision,
) (*decision.DecisionResult, string) {
	// llm_decision_evaluation_latency_seconds and llm_decision_match_total are
	// emitted by decision.DecisionEngine.EvaluateDecisionsWithSignals; do not
	// emit them here or both metrics will be double-counted.
	decisionCtx, decisionSpan := tracing.StartDecisionSpan(ctx.TraceContext, "decision_evaluation")

	var result *decision.DecisionResult
	var err error
	if candidates != nil {
		if len(candidates) == 0 {
			tracing.EndDecisionSpan(decisionSpan, 0.0, []string{}, r.Config.Strategy)
			ctx.TraceContext = decisionCtx
			return nil, ""
		}
		result, err = r.Classifier.EvaluateDecisionWithEngineForDecisions(signals, candidates)
	} else {
		result, err = r.Classifier.EvaluateDecisionWithEngine(signals)
	}
	if err != nil {
		logging.ComponentErrorEvent("extproc", "decision_evaluation_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"strategy":   r.Config.Strategy,
			"error":      err.Error(),
		})
		tracing.EndDecisionSpan(decisionSpan, 0.0, []string{}, r.Config.Strategy)
		ctx.TraceContext = decisionCtx
		return nil, r.defaultModelForUnmatchedDecision(originalModel)
	}
	if result == nil || result.Decision == nil {
		tracing.EndDecisionSpan(decisionSpan, 0.0, []string{}, r.Config.Strategy)
		ctx.TraceContext = decisionCtx
		return nil, r.defaultModelForUnmatchedDecision(originalModel)
	}

	tracing.EndDecisionSpan(decisionSpan, result.Confidence, result.MatchedRules, r.Config.Strategy)
	ctx.TraceContext = decisionCtx
	return result, ""
}

func (r *OpenAIRouter) defaultModelForUnmatchedDecision(originalModel string) string {
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
	logging.ComponentDebugEvent("extproc", "decision_evaluated", map[string]interface{}{
		"request_id":    ctx.RequestID,
		"decision":      decisionName,
		"category":      categoryName,
		"confidence":    evaluationConfidence,
		"matched_rules": result.MatchedRules,
	})

	if !r.Config.IsAutoModelName(originalModel) {
		logging.ComponentDebugEvent("extproc", "explicit_model_preserved", map[string]interface{}{
			"request_id":     ctx.RequestID,
			"original_model": originalModel,
			"decision":       decisionName,
		})
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
	if pluginCfg := r.Config.EffectiveRouterReplayConfigForDecision(result.Decision.Name); pluginCfg != nil {
		ctx.RouterReplayPluginConfig = pluginCfg
	}

	// Snapshot the retention directive emitted by this decision (deep clone)
	// and observe every declared field via log + trace. Both helpers are
	// no-ops when the decision did not emit a retention block.
	applyEmittedRetention(result.Decision, ctx)
	observeRetentionDirective(ctx)

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
		logging.ComponentDebugEvent("extproc", "decision_model_defaulted", map[string]interface{}{
			"request_id":     ctx.RequestID,
			"decision":       decisionName,
			"selected_model": selectedModel,
		})
		return selectedModel, entropy.ReasoningDecision{}
	}

	selCtx := r.buildSelectionContext(
		result.Decision.ModelRefs,
		decisionName,
		userContent,
		result.Decision.Algorithm,
		categoryName,
		result.Decision.CandidateIterations,
		ctx,
	)
	selectedModelRef, usedMethod := r.selectModelFromCandidates(selCtx, result.Decision.Algorithm, ctx)
	if selectedModelRef == nil {
		selectedModel := r.Config.DefaultModel
		ctx.VSRSelectedModel = selectedModel
		ctx.VSRSelectionMethod = "default"
		logging.Warnf("[ModelSelection] No valid decision modelRefs for decision %s, using default model %s", decisionName, selectedModel)
		return selectedModel, entropy.ReasoningDecision{}
	}
	selectedModel := selectedModelRef.Model
	selectionFields := map[string]interface{}{
		"request_id":        ctx.RequestID,
		"decision":          decisionName,
		"selected_model":    selectedModelRef.Model,
		"selection_method":  usedMethod,
		"uses_lora_adapter": selectedModelRef.LoRAName != "",
	}
	if selectedModelRef.LoRAName != "" {
		selectedModel = selectedModelRef.LoRAName
		selectionFields["selected_model"] = selectedModel
		selectionFields["base_model"] = selectedModelRef.Model
	}
	logging.ComponentDebugEvent("extproc", "decision_model_selected", selectionFields)
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
