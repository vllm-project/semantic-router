package extproc

import (
	"context"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
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
	ctx.VSRConversationFacts = signalInput.conversationFacts
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
// The selCtx parameter carries the pre-built SelectionContext, including request-time
// inputs such as query text, candidate models, and cache-affinity signals.
// Returns the selected model and the method name used for logging.
func (r *OpenAIRouter) selectModelFromCandidates(selCtx *selection.SelectionContext, algorithm *config.AlgorithmConfig, ctx *RequestContext) (*config.ModelRef, string) {
	fallbackModelRef := firstValidCandidateModelRef(selCtx)
	if fallbackModelRef == nil {
		return nil, ""
	}
	if err := selection.ValidateSelectionContext(selCtx); err != nil {
		logging.Warnf("[ModelSelection] Invalid selection context: %v, using first valid model", err)
		recordAgenticSessionDecision(selCtx, nil, fallbackModelRef, ctx)
		return fallbackModelRef, ""
	}

	// If only one model, no need for selection algorithm
	if len(selCtx.CandidateModels) == 1 {
		recordAgenticSessionDecision(selCtx, nil, fallbackModelRef, ctx)
		return fallbackModelRef, "single"
	}

	// Determine selection method: per-decision algorithm takes precedence over global config
	method := r.getSelectionMethod(algorithm)

	// Get selector from registry
	selector := r.selectorForDecisionMethod(method, algorithm)

	// Fallback to first model if no selector available
	if selector == nil {
		logging.Warnf("[ModelSelection] No selector available for method %s, using first model", method)
		recordAgenticSessionDecision(selCtx, nil, fallbackModelRef, ctx)
		return fallbackModelRef, string(method)
	}

	// Perform selection
	result, err := selector.Select(context.Background(), selCtx)
	if err != nil {
		logging.Warnf("[ModelSelection] Selection failed: %v, using first model", err)
		recordAgenticSessionDecision(selCtx, nil, fallbackModelRef, ctx)
		return fallbackModelRef, string(method)
	}
	if err := selection.ValidateSelectionResult(selCtx, result); err != nil {
		logging.Warnf("[ModelSelection] Invalid selection result: %v, using first model", err)
		recordAgenticSessionDecision(selCtx, result, fallbackModelRef, ctx)
		return fallbackModelRef, string(method)
	}

	selectedModelRef := selectedModelRefFromResult(selCtx, result)
	if selectedModelRef == nil {
		logging.Warnf("[ModelSelection] Selected model %s not found in candidates, using first model", result.SelectedModel)
		recordAgenticSessionDecision(selCtx, result, fallbackModelRef, ctx)
		return fallbackModelRef, string(method)
	}
	if result.SessionPolicy != nil && ctx != nil {
		ctx.VSRSessionPolicy = result.SessionPolicy.ToMap()
	}
	selectedModelRef, gateApplied := r.applyPostSelectionGate(selCtx, result, selectedModelRef, ctx, method)
	logSelectionResult(method, result, selectedModelRef, gateApplied)
	selection.RecordSelection(string(method), selCtx.DecisionName, selectedModelRef.Model, result.Tier, result.Score)
	recordAgenticSessionDecision(selCtx, result, selectedModelRef, ctx)
	return selectedModelRef, string(method)
}

func (r *OpenAIRouter) selectorForDecisionMethod(method selection.SelectionMethod, algorithm *config.AlgorithmConfig) selection.Selector {
	if method == selection.MethodSessionAware && algorithm != nil && algorithm.SessionAware != nil {
		return r.newDecisionSessionAwareSelector(algorithm.SessionAware)
	}
	if r.ModelSelector == nil {
		return nil
	}
	selector, _ := r.ModelSelector.Get(method)
	return selector
}

func (r *OpenAIRouter) newDecisionSessionAwareSelector(decisionCfg *config.SessionAwareSelectionConfig) selection.Selector {
	cfg := selection.DefaultSessionAwareConfig()
	if r != nil && r.Config != nil {
		cfg = buildSessionAwareSelectionConfig(r.Config, decisionCfg)
	} else if decisionCfg != nil {
		applySessionAwareSelectionConfig(cfg, *decisionCfg)
	}
	selector := selection.NewSessionAwareSelector(cfg)
	if r == nil {
		return selector
	}
	if r.Config != nil && r.Config.ModelConfig != nil {
		selector.InitializeFromConfig(r.Config.ModelConfig)
	}
	if r.LookupTable != nil {
		selector.SetLookupTable(r.LookupTable)
	}
	if r.ModelSelector != nil && cfg.BaseMethod != "" && cfg.BaseMethod != selection.MethodSessionAware {
		baseSelector, _ := r.ModelSelector.Get(cfg.BaseMethod)
		selector.SetBaseSelector(baseSelector)
	}
	return selector
}

func (r *OpenAIRouter) applyPostSelectionGate(
	selCtx *selection.SelectionContext,
	result *selection.SelectionResult,
	selectedModelRef *config.ModelRef,
	ctx *RequestContext,
	method selection.SelectionMethod,
) (*config.ModelRef, bool) {
	if method == selection.MethodSessionAware || result.Method == selection.MethodSessionAware {
		return selectedModelRef, false
	}
	return r.applyModelSwitchGate(selCtx, result, selectedModelRef, ctx)
}

func selectedModelRefFromResult(selCtx *selection.SelectionContext, result *selection.SelectionResult) *config.ModelRef {
	for i := range selCtx.CandidateModels {
		if selCtx.CandidateModels[i].Model == result.SelectedModel ||
			selCtx.CandidateModels[i].LoRAName == result.SelectedModel {
			return &selCtx.CandidateModels[i]
		}
	}
	return nil
}

func logSelectionResult(method selection.SelectionMethod, result *selection.SelectionResult, selected *config.ModelRef, gateApplied bool) {
	if gateApplied {
		logging.Infof("[ModelSelection] Gate enforced stay on %s (method=%s, score=%.4f, confidence=%.2f): %s",
			selected.Model, method, result.Score, result.Confidence, result.Reasoning)
		return
	}
	logging.Infof("[ModelSelection] Selected %s (method=%s, score=%.4f, confidence=%.2f): %s",
		selected.Model, method, result.Score, result.Confidence, result.Reasoning)
}

func firstValidCandidateModelRef(selCtx *selection.SelectionContext) *config.ModelRef {
	if selCtx == nil {
		return nil
	}
	for i := range selCtx.CandidateModels {
		if strings.TrimSpace(selCtx.CandidateModels[i].Model) != "" {
			return &selCtx.CandidateModels[i]
		}
	}
	return nil
}

// buildSelectionContext assembles the runtime inputs shared by selection
// algorithms. Static policy comes from AlgorithmConfig; dynamic request-time
// signals stay in SelectionContext so selectors do not need to reach back into
// extproc state.
func (r *OpenAIRouter) buildSelectionContext(
	modelRefs []config.ModelRef,
	decisionName string,
	query string,
	algorithm *config.AlgorithmConfig,
	categoryName string,
	candidateIterations []config.CandidateIterationConfig,
	reqCtx *RequestContext,
) *selection.SelectionContext {
	costWeight, qualityWeight := r.getSelectionWeights(algorithm)
	latencyAwareTPOTPercentile, latencyAwareTTFTPercentile := r.getLatencyAwarePercentiles(algorithm)

	sessionID, userID, conversationHistory := r.extractSessionContext(reqCtx)

	return &selection.SelectionContext{
		Query:                      query,
		DecisionName:               decisionName,
		CategoryName:               categoryName,
		CandidateModels:            modelRefs,
		CandidateIterations:        candidateIterations,
		CostWeight:                 costWeight,
		QualityWeight:              qualityWeight,
		LatencyAwareTPOTPercentile: latencyAwareTPOTPercentile,
		LatencyAwareTTFTPercentile: latencyAwareTTFTPercentile,
		UserID:                     userID,
		SessionID:                  sessionID,
		AgenticSession:             r.buildAgenticSessionContext(reqCtx, modelRefs, sessionID, userID),
		ConversationHistory:        conversationHistory,
		CacheAffinityCtx:           r.buildCacheAffinityContext(reqCtx, modelRefs),
	}
}

func (r *OpenAIRouter) buildAgenticSessionContext(
	reqCtx *RequestContext,
	modelRefs []config.ModelRef,
	sessionID string,
	userID string,
) *selection.AgenticSessionContext {
	if reqCtx == nil {
		return nil
	}
	now := time.Now()
	snapshot, hasMemory := sessiontelemetry.GetRouterSessionSnapshot(sessionID, now)
	previousModel := reqCtx.PreviousModel
	if previousModel == "" && hasMemory {
		previousModel = snapshot.CurrentModel
	}
	idleFor := time.Duration(reqCtx.SessionIdleSeconds * float64(time.Second))
	idleKnown := reqCtx.SessionIdleKnown
	if hasMemory {
		idleFor = snapshot.IdleFor
		idleKnown = true
	}
	cacheWarmth, cacheWarmthOK := r.agenticCacheWarmth(reqCtx, previousModel, snapshot, hasMemory, now)
	facts := reqCtx.VSRConversationFacts
	activeToolLoop := conversationFactsIndicateActiveToolLoop(facts)
	phase := selection.AgenticPhaseUserTurn
	if activeToolLoop {
		phase = selection.AgenticPhaseToolLoop
	}
	hasNonPortableContext, nonPortableContextReason := nonPortableContextBinding(reqCtx)
	return &selection.AgenticSessionContext{
		ID:                          sessionID,
		UserID:                      userID,
		TurnIndex:                   reqCtx.TurnIndex,
		PreviousModel:               previousModel,
		PreviousResponseID:          reqCtx.PreviousResponseID,
		MemoryPresent:               hasMemory,
		MemoryTurnCount:             snapshot.TurnCount,
		MemorySwitchCount:           snapshot.SwitchCount,
		MemoryModelTurnCnts:         snapshot.ModelTurns,
		MemoryPromptTokens:          snapshot.CumulativePromptTokens,
		MemoryCachedTokens:          snapshot.CumulativeCachedTokens,
		MemoryEstimatedCachedTokens: snapshot.CumulativeEstimatedCachedTokens,
		MemoryOutputTokens:          snapshot.CumulativeCompletionTokens,
		MemoryCost:                  snapshot.CumulativeCost,
		MemoryEstimatedCacheSavings: snapshot.CumulativeEstimatedCacheSavings,
		MemoryCacheAccountingSource: snapshot.LastCacheAccountingSource,
		LastDecisionName:            snapshot.LastDecisionName,
		LastDecisionReason:          snapshot.LastDecisionReason,
		HistoryTokens:               reqCtx.HistoryTokenCount,
		ContextTokens:               reqCtx.VSRContextTokenCount,
		IdleFor:                     idleFor,
		IdleKnown:                   idleKnown,
		CacheWarmth:                 cacheWarmth,
		CacheWarmthOK:               cacheWarmthOK,
		Phase:                       phase,
		ActiveToolLoop:              activeToolLoop,
		HasNonPortableContext:       hasNonPortableContext,
		NonPortableContextReason:    nonPortableContextReason,
		ToolCallCount:               facts.AssistantToolCallCount,
		ToolResultCount:             facts.ToolResultCount,
		ToolDefinitionCnt:           facts.ToolDefinitionCount,
		ModelContextWindows:         r.modelContextWindows(modelRefs),
	}
}

func nonPortableContextBinding(reqCtx *RequestContext) (bool, string) {
	if reqCtx == nil {
		return false, ""
	}
	if strings.TrimSpace(reqCtx.PreviousResponseID) != "" {
		return true, "previous_response_id"
	}
	return false, ""
}

func conversationFactsIndicateActiveToolLoop(facts classification.ConversationFacts) bool {
	return facts.LastMessageToolResult ||
		facts.LastMessageRole == "tool" ||
		facts.LastUserAfterToolResult ||
		facts.AssistantToolCallCount > facts.ToolResultCount
}

func (r *OpenAIRouter) agenticCacheWarmth(
	reqCtx *RequestContext,
	previousModel string,
	snapshot sessiontelemetry.RouterSessionSnapshot,
	hasMemory bool,
	now time.Time,
) (float64, bool) {
	cacheWarmth := reqCtx.CacheWarmthEstimate
	cacheWarmthOK := cacheWarmth > 0
	if ambient, ok := estimateGateCacheWarmth(previousModel, now); ok {
		cacheWarmth = ambient
		cacheWarmthOK = true
	}
	if hasMemory && snapshot.CumulativePromptTokens > 0 {
		cachedRatio := float64(snapshot.CumulativeCachedTokens) / float64(snapshot.CumulativePromptTokens)
		if cachedRatio > cacheWarmth {
			cacheWarmth = cachedRatio
			cacheWarmthOK = true
		}
	}
	return cacheWarmth, cacheWarmthOK
}

func (r *OpenAIRouter) modelContextWindows(modelRefs []config.ModelRef) map[string]int {
	if r == nil || r.Config == nil || r.Config.ModelConfig == nil {
		return nil
	}
	windows := make(map[string]int, len(modelRefs))
	for _, ref := range modelRefs {
		if params, ok := r.Config.ModelConfig[ref.Model]; ok {
			windows[ref.Model] = params.ContextWindowSize
		}
	}
	return windows
}

// buildCacheAffinityContext extracts the pre-dispatch continuation signals used
// by the cache-affinity estimator. Nil request context cleanly disables the
// estimator without forcing call sites to add extra branching.
func (r *OpenAIRouter) buildCacheAffinityContext(reqCtx *RequestContext, modelRefs []config.ModelRef) *selection.CacheAffinityContext {
	if reqCtx == nil {
		return nil
	}

	// Missing model window metadata is valid; the estimator treats it as a
	// neutral fit score rather than as an error.
	return &selection.CacheAffinityContext{
		TurnIndex:           reqCtx.TurnIndex,
		PreviousModel:       reqCtx.PreviousModel,
		PreviousResponseID:  reqCtx.PreviousResponseID,
		HistoryTokens:       reqCtx.HistoryTokenCount,
		ContextTokens:       reqCtx.VSRContextTokenCount,
		ModelContextWindows: r.modelContextWindows(modelRefs),
	}
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

	// Get decision name safely
	decisionName := ""
	if ctx.VSRSelectedDecision != nil {
		decisionName = ctx.VSRSelectedDecision.Name
	}

	// Process each feedback signal
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

// extractSessionContext extracts session ID, user ID, and conversation history from the RequestContext.
func (r *OpenAIRouter) extractSessionContext(ctx *RequestContext) (sessionID, userID string, conversationHistory []string) {
	if ctx == nil {
		return "", "", nil
	}
	userID = extractUserID(ctx)
	if ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.IsResponseAPIRequest {
		return r.extractResponseAPISessionContext(ctx, userID)
	}
	if len(ctx.ChatCompletionMessages) > 0 {
		return r.extractChatCompletionSessionContext(ctx, userID)
	}
	return ctx.SessionID, userID, nil
}

func (r *OpenAIRouter) extractResponseAPISessionContext(ctx *RequestContext, userID string) (sessionID, userIDOut string, conversationHistory []string) {
	sessionID = ctx.ResponseAPICtx.ConversationID
	if ctx.ResponseAPICtx.ConversationHistory != nil {
		for _, storedResp := range ctx.ResponseAPICtx.ConversationHistory {
			for _, inItem := range storedResp.Input {
				if content := extractContentFromInputItem(inItem); content != "" {
					conversationHistory = append(conversationHistory, content)
				}
			}
			for _, outItem := range storedResp.Output {
				if content := extractContentFromOutputItem(outItem); content != "" {
					conversationHistory = append(conversationHistory, content)
				}
			}
		}
	}
	return sessionID, userID, conversationHistory
}

func (r *OpenAIRouter) extractChatCompletionSessionContext(ctx *RequestContext, userID string) (sessionID, userIDOut string, conversationHistory []string) {
	sessionID = ctx.SessionID
	if sessionID == "" {
		sessionID = deriveSessionIDFromMessages(ctx.ChatCompletionMessages, userID)
	}
	for i, msg := range ctx.ChatCompletionMessages {
		if msg.Content != "" && i < len(ctx.ChatCompletionMessages)-1 {
			conversationHistory = append(conversationHistory, msg.Content)
		}
	}
	return sessionID, userID, conversationHistory
}
