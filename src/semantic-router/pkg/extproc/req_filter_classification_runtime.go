package extproc

import (
	"fmt"
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
	"prompt":        selection.MethodPrompt,
}

func (r *OpenAIRouter) evaluateSignalsForDecision(
	originalModel string,
	signalInput signalEvaluationInput,
	nonUserMessages []string,
	ctx *RequestContext,
	candidates []config.Decision,
) (*classification.SignalResults, error) {
	signalStart := time.Now()
	signalCtx, signalSpan := tracing.StartSpan(ctx.TraceContext, tracing.SpanSignalEvaluation)

	classifier := r.classifierForRequest(ctx)
	if classifier == nil {
		return nil, fmt.Errorf("classifier for routing recipe %q is unavailable", ctx.Routing.RecipeName())
	}

	signals, authzErr := classifier.EvaluateAllSignalsWithHeaders(classification.SignalEvaluationInput{
		Text:                   signalInput.compressedText,
		ContextText:            signalInput.allMessagesText,
		CurrentUserText:        signalInput.currentUserText,
		PriorUserMessages:      signalInput.priorUserMessages,
		NonUserMessages:        nonUserMessages,
		HasPriorAssistantReply: signalInput.hasAssistantReply,
		Headers:                ctx.Headers,
		ImageURL:               ctx.RequestImageURL,
		UncompressedText:       signalInput.evaluationText,
		SkipCompressionSignals: signalInput.skipCompressionSignals,
		ConversationFacts:      signalInput.conversationFacts,
		RequestFacts:           signalInput.requestFacts,
	})
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
	floor := signalInput.requestFacts.ContextTokenFloor
	ctx.VSRContextHasNonText = ctx.VSRContextHasNonText ||
		signalInput.requestFacts.ContextHasNonText
	if text == "" && floor <= 0 {
		return
	}
	if ctx.VSRContextTextBytes <= 0 {
		ctx.VSRContextTextBytes = signalInput.requestFacts.ContextTextBytes
		if ctx.VSRContextTextBytes <= 0 {
			ctx.VSRContextTextBytes = len(text)
		}
	}
	if ctx.VSRContextEquivalentBytes <= 0 {
		ctx.VSRContextEquivalentBytes = signalInput.requestFacts.ContextEquivalentBytes
	}
	count := ctx.VSRContextTokenCount
	if count <= 0 {
		counter := classification.CharacterBasedTokenCounter{}
		var err error
		count, err = counter.CountTokens(text)
		if err != nil {
			return
		}
	}
	if floor > count {
		count = floor
	}
	if count <= 0 {
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
		"metadata":       signals.MatchedMetadataRules,
		"classifier":     signals.MatchedClassifierRules,
		"input_modality": signals.MatchedInputModalityRules,
		"projection":     signals.MatchedProjectionRules,
		"context_tokens": signals.TokenCount,
	})
}

func (r *OpenAIRouter) runDecisionEngine(
	originalModel string,
	ctx *RequestContext,
	signals *classification.SignalResults,
	candidates []config.Decision,
) (*decision.DecisionResult, string, error) {
	// llm_decision_evaluation_latency_seconds and llm_decision_match_total are
	// emitted by decision.DecisionEngine.EvaluateDecisionsWithSignals; do not
	// emit them here or both metrics will be double-counted.
	decisionCtx, decisionSpan := tracing.StartDecisionSpan(ctx.TraceContext, "decision_evaluation")
	classifier := r.classifierForRequest(ctx)
	if classifier == nil {
		logging.ComponentErrorEvent("extproc", "recipe_classifier_unavailable", map[string]interface{}{
			"request_id": ctx.RequestID,
			"recipe":     ctx.Routing.RecipeName(),
		})
		tracing.EndDecisionSpan(decisionSpan, 0.0, []string{}, "")
		ctx.TraceContext = decisionCtx
		return nil, r.defaultModelForUnmatchedDecision(originalModel), nil
	}
	strategy := classifier.Config.Strategy

	var result *decision.DecisionResult
	var err error
	if candidates != nil {
		if len(candidates) == 0 {
			tracing.EndDecisionSpan(decisionSpan, 0.0, []string{}, string(strategy))
			ctx.TraceContext = decisionCtx
			return nil, r.defaultModelForUnmatchedDecision(originalModel), nil
		}
		result, err = classifier.EvaluateDecisionWithEngineForDecisions(signals, candidates)
	} else {
		result, err = classifier.EvaluateDecisionWithEngine(signals)
	}
	ctx.VSRAppliedUnknownPolicies = cloneReplayStringMap(signals.AppliedUnknownPolicies)
	if err != nil {
		logging.ComponentErrorEvent("extproc", "decision_evaluation_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"strategy":   strategy,
			"error":      err.Error(),
		})
		tracing.EndDecisionSpan(decisionSpan, 0.0, []string{}, string(strategy))
		ctx.TraceContext = decisionCtx
		return nil, "", err
	}
	if result == nil || result.Decision == nil {
		tracing.EndDecisionSpan(decisionSpan, 0.0, []string{}, string(strategy))
		ctx.TraceContext = decisionCtx
		return nil, r.defaultModelForUnmatchedDecision(originalModel), nil
	}

	tracing.EndDecisionSpan(decisionSpan, result.Confidence, result.MatchedRules, string(strategy))
	ctx.TraceContext = decisionCtx
	return result, "", nil
}

func (r *OpenAIRouter) defaultModelForUnmatchedDecision(originalModel string) string {
	if r.requestModelActsAsAuto(originalModel) {
		return r.Config.DefaultModel
	}
	return ""
}

func (r *OpenAIRouter) finalizeDecisionEvaluation(
	result *decision.DecisionResult,
	originalModel string,
	userContent string,
	ctx *RequestContext,
) (string, float64, entropy.ReasoningDecision, string, error) {
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

	if !r.requestModelActsAsAuto(originalModel) {
		logging.ComponentDebugEvent("extproc", "explicit_model_preserved", map[string]interface{}{
			"request_id":     ctx.RequestID,
			"original_model": originalModel,
			"decision":       decisionName,
		})
		return decisionName, evaluationConfidence, reasoningDecision, "", nil
	}

	selectedModel, reasoningDecision, err := r.selectDecisionRuntimeModel(
		result,
		decisionName,
		userContent,
		categoryName,
		evaluationConfidence,
		ctx,
	)
	return decisionName, evaluationConfidence, reasoningDecision, selectedModel, err
}

func (r *OpenAIRouter) applyDecisionResultToContext(result *decision.DecisionResult, ctx *RequestContext) string {
	ctx.VSRSelectedDecision = result.Decision
	if pluginCfg := r.Config.EffectiveRouterReplayConfig(result.Decision); pluginCfg != nil {
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
) (string, entropy.ReasoningDecision, error) {
	if result.Decision.GetFastResponseConfig() != nil {
		return r.selectFastResponseRuntimeModel(result.Decision, ctx), entropy.ReasoningDecision{}, nil
	}
	if ineligible := r.contextIneligibleAlgorithmModelCount(result.Decision, ctx.VSRContextTokenCount); ineligible > 0 {
		return "", entropy.ReasoningDecision{}, fmt.Errorf(
			"%w: decision %q requires %d request tokens but %d explicitly configured algorithm model(s) have smaller context windows",
			errNoContextEligibleDecisionModel,
			decisionName,
			ctx.VSRContextTokenCount,
			ineligible,
		)
	}
	if len(result.Decision.ModelRefs) == 0 {
		return r.selectDecisionDefaultRuntimeModel(result.Decision, decisionName, ctx)
	}

	eligibleModelRefs, err := r.contextEligibleDecisionModelRefs(
		result.Decision.ModelRefs,
		decisionName,
		ctx.VSRContextTokenCount,
		ctx,
	)
	if err != nil {
		return "", entropy.ReasoningDecision{}, err
	}
	if minimumErr := validateMinimumEligibleDecisionModels(
		result.Decision,
		eligibleModelRefs,
		ctx.VSRContextTokenCount,
	); minimumErr != nil {
		return "", entropy.ReasoningDecision{}, minimumErr
	}

	selCtx := r.buildSelectionContext(
		eligibleModelRefs,
		decisionName,
		userContent,
		result.Decision.Algorithm,
		categoryName,
		result.Decision.CandidateIterations,
		ctx,
	)
	selectedModelRef, usedMethod, err := r.selectModelFromCandidates(
		selCtx,
		result.Decision.Algorithm,
		ctx,
	)
	if err != nil {
		return "", entropy.ReasoningDecision{}, err
	}
	if selectedModelRef == nil {
		selectedModel := r.Config.DefaultModel
		ctx.VSRSelectedModel = selectedModel
		ctx.VSRSelectionMethod = "default"
		logging.Warnf("[ModelSelection] No valid decision modelRefs for decision %s, using default model %s", decisionName, selectedModel)
		return selectedModel, entropy.ReasoningDecision{}, nil
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
	return selectedModel, applyReasoningModeFromSelectedModel(
		selectedModelRef,
		decisionName,
		evaluationConfidence,
		ctx,
	), nil
}

func (r *OpenAIRouter) selectFastResponseRuntimeModel(
	decisionConfig *config.Decision,
	ctx *RequestContext,
) string {
	selectedModel := firstDecisionModelName(decisionConfig.ModelRefs)
	if selectedModel == "" {
		selectedModel = r.Config.DefaultModel
	}
	ctx.VSRSelectedModel = selectedModel
	ctx.VSRSelectionMethod = "fast_response"
	return selectedModel
}

func firstDecisionModelName(modelRefs []config.ModelRef) string {
	for _, modelRef := range modelRefs {
		if model := strings.TrimSpace(modelRef.LoRAName); model != "" {
			return model
		}
		if model := strings.TrimSpace(modelRef.Model); model != "" {
			return model
		}
	}
	return ""
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
