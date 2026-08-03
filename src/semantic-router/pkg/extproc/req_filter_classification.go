package extproc

import (
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
// Decision evaluation runs only for request-facing routing models. Concrete
// backend model IDs are passthrough requests: they do not inherit the default
// recipe's signals, policy, decisions, or plugins. Direct looper slugs use the
// default recipe while limiting candidates to their algorithm type.
func (r *OpenAIRouter) performDecisionEvaluation(originalModel string, history signalConversationHistory, ctx *RequestContext) (string, float64, entropy.ReasoningDecision, string, error) {
	var decisionName string
	var evaluationConfidence float64
	var reasoningDecision entropy.ReasoningDecision
	var selectedModel string

	// Check if there's content to evaluate
	if len(history.nonUserMessages) == 0 && history.currentUserMessage == "" &&
		!hasEnvelopeRoutingFacts(history) {
		return "", 0.0, entropy.ReasoningDecision{}, "", nil
	}

	// Focused callers may invoke this method without the normal pre-routing
	// stage. Resolve here as an idempotent guard so the isolation boundary is
	// never dependent on call order.
	if !ctx.Routing.IsResolved() {
		r.resolveEntrypointForRequest(originalModel, ctx)
	}
	if ctx.Routing.SelectedRecipe() == nil {
		return "", 0.0, entropy.ReasoningDecision{}, "", nil
	}

	// Check if decisions are configured in any routing profile; the flat
	// Decisions field only carries the default recipe's.
	if !r.Config.HasRoutingDecisions() {
		if r.requestModelActsAsAuto(originalModel) {
			logging.Warnf("No decisions configured, using default model")
			return "", 0.0, entropy.ReasoningDecision{}, r.Config.DefaultModel, nil
		}
		return "", 0.0, entropy.ReasoningDecision{}, "", nil
	}

	signalInput := r.prepareSignalEvaluationInput(history)
	signalInput.requestFacts.Context = ctx.TraceContext
	ctx.VSRConversationFacts = signalInput.conversationFacts
	if signalInput.evaluationText == "" && !hasEnvelopeRoutingFacts(history) {
		return "", 0.0, entropy.ReasoningDecision{}, "", nil
	}

	candidates := r.decisionCandidatesForRequest(originalModel, ctx)
	signals, authzErr := r.evaluateSignalsForDecision(originalModel, signalInput, history.nonUserMessages, ctx, candidates)
	if authzErr != nil {
		return "", 0, entropy.ReasoningDecision{}, "", authzErr
	}

	result, defaultModel := r.runDecisionEngine(originalModel, ctx, signals, candidates)
	if result == nil {
		return "", 0.0, entropy.ReasoningDecision{}, defaultModel, nil
	}

	decisionName, evaluationConfidence, reasoningDecision, selectedModel, err := r.finalizeDecisionEvaluation(
		result,
		originalModel,
		history.currentUserMessage,
		ctx,
	)
	return decisionName, evaluationConfidence, reasoningDecision, selectedModel, err
}

func (r *OpenAIRouter) selectorForDecisionMethod(method selection.SelectionMethod, algorithm *config.AlgorithmConfig, ctx *RequestContext) selection.Selector {
	if method == selection.MethodHybrid && algorithm != nil && algorithm.Hybrid != nil {
		return r.newDecisionHybridSelector(algorithm.Hybrid, ctx)
	}
	if method == selection.MethodMultiFactor && algorithm != nil {
		return r.newDecisionMultiFactorSelector(algorithm.MultiFactor)
	}
	if method == selection.MethodPrompt && algorithm != nil &&
		algorithm.Prompt != nil {
		return r.newDecisionPromptSelector(*algorithm.Prompt)
	}
	registry := r.modelSelectorForRequest(ctx)
	if registry == nil {
		return nil
	}
	selector, _ := registry.Get(method)
	return selector
}

func (r *OpenAIRouter) modelSelectorForRequest(ctx *RequestContext) *selection.Registry {
	if r == nil {
		return nil
	}
	if ctx != nil && ctx.Routing.RecipeName() != "" && r.RecipeModelSelectors != nil {
		if registry, ok := r.RecipeModelSelectors[ctx.Routing.RecipeName()]; ok {
			return registry
		}
		return nil
	}
	return r.ModelSelector
}

func (r *OpenAIRouter) newDecisionMultiFactorSelector(decisionCfg *config.MultiFactorSelectionConfig) selection.Selector {
	selector := selection.NewMultiFactorSelector(buildMultiFactorSelectionConfig(decisionCfg))
	if r != nil && r.Config != nil && r.Config.ModelConfig != nil {
		selector.InitializeFromConfig(r.Config.ModelConfig)
	}
	return selector
}

func (r *OpenAIRouter) newDecisionHybridSelector(decisionCfg *config.HybridSelectionConfig, ctx *RequestContext) selection.Selector {
	var cfg *selection.HybridConfig
	if r != nil && r.Config != nil {
		cfg = buildHybridSelectionConfig(r.Config, decisionCfg)
	} else {
		cfg = selection.DefaultHybridConfig()
	}

	eloSelector, routerDCSelector, autoMixSelector := r.hybridComponentSelectors(r.modelSelectorForRequest(ctx))

	selector := selection.NewHybridSelectorWithComponents(cfg, eloSelector, routerDCSelector, autoMixSelector)
	r.applyHybridModelCosts(selector)
	if r != nil && r.LookupTable != nil {
		selector.SetLookupTable(r.LookupTable)
	}
	return selector
}

// hybridComponentSelectors resolves the underlying elo/routerDC/autoMix selectors
// that the hybrid selector composes, when they are registered on the router.
func (r *OpenAIRouter) hybridComponentSelectors(registry *selection.Registry) (*selection.EloSelector, *selection.RouterDCSelector, *selection.AutoMixSelector) {
	if registry == nil {
		return nil, nil, nil
	}
	var eloSelector *selection.EloSelector
	var routerDCSelector *selection.RouterDCSelector
	var autoMixSelector *selection.AutoMixSelector
	if selector, ok := registry.Get(selection.MethodElo); ok {
		eloSelector, _ = selector.(*selection.EloSelector)
	}
	if selector, ok := registry.Get(selection.MethodRouterDC); ok {
		routerDCSelector, _ = selector.(*selection.RouterDCSelector)
	}
	if selector, ok := registry.Get(selection.MethodAutoMix); ok {
		autoMixSelector, _ = selector.(*selection.AutoMixSelector)
	}
	return eloSelector, routerDCSelector, autoMixSelector
}

// applyHybridModelCosts seeds per-model prompt pricing into the hybrid selector.
func (r *OpenAIRouter) applyHybridModelCosts(selector *selection.HybridSelector) {
	if r == nil || r.Config == nil || r.Config.ModelConfig == nil {
		return
	}
	for model, params := range r.Config.ModelConfig {
		if params.Pricing.PromptPer1M > 0 {
			selector.SetModelCost(model, params.Pricing.PromptPer1M)
		}
	}
}

func selectedModelRefFromResult(selCtx *selection.SelectionContext, result *selection.SelectionResult) *config.ModelRef {
	for i := range selCtx.CandidateModels {
		if selCtx.CandidateModels[i].Model == result.SelectedModel {
			return &selCtx.CandidateModels[i]
		}
		if result.Method != selection.MethodPrompt &&
			selCtx.CandidateModels[i].LoRAName == result.SelectedModel {
			return &selCtx.CandidateModels[i]
		}
	}
	return nil
}

func logSelectionResult(method selection.SelectionMethod, result *selection.SelectionResult, selected *config.ModelRef, learningApplied bool) {
	if learningApplied {
		logging.Infof(
			"[ModelSelection] Router Learning adjusted selection to %s (base_method=%s, score=%.4f, confidence=%.2f)",
			selected.Model,
			method,
			result.Score,
			result.Confidence,
		)
		return
	}
	logging.Infof(
		"[ModelSelection] Selected %s (method=%s, score=%.4f, confidence=%.2f)",
		selected.Model,
		method,
		result.Score,
		result.Confidence,
	)
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
	var recipeName config.RecipeName
	if reqCtx != nil {
		recipeName = reqCtx.Routing.RecipeName()
	}

	return &selection.SelectionContext{
		Query:                      query,
		DecisionName:               decisionName,
		RecipeName:                 recipeName,
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
	stateSessionID := config.RoutingNamespaceKey(reqCtx.Routing.RecipeName(), sessionID)
	snapshot, hasMemory := sessiontelemetry.GetRouterSessionSnapshot(stateSessionID, now)
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
	hasNonPortableContext, nonPortableContextReason := nonPortableContextBinding(reqCtx)
	phase := selection.AgenticPhaseUserTurn
	if hasNonPortableContext {
		phase = selection.AgenticPhaseProviderState
	}
	if activeToolLoop {
		phase = selection.AgenticPhaseToolLoop
	}
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
