package extproc

import (
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// performDecisionEvaluation performs decision evaluation using DecisionEngine
// Returns (decisionName, confidence, reasoningDecision, selectedModel)
// This is the new approach that uses Decision-based routing with AND/OR rule combinations
// Decision evaluation runs only for request-facing Entrypoints. Concrete
// backend Model IDs are passthrough requests: they do not inherit a Recipe's
// signals, policy, decisions, or plugins.
func (r *OpenAIRouter) performDecisionEvaluation(originalModel string, history signalConversationHistory, ctx *RequestContext) (string, float64, entropy.ReasoningDecision, string, error) {
	var decisionName string
	var evaluationConfidence float64
	var reasoningDecision entropy.ReasoningDecision
	var selectedModel string

	if fallbackModel, stop := r.prepareDecisionEvaluation(originalModel, history, ctx); stop {
		return "", 0.0, entropy.ReasoningDecision{}, fallbackModel, nil
	}

	signalInput := r.prepareSignalEvaluationInput(history)
	observePromptCompression(ctx, signalInput.compression.outcome, signalInput.compression.elapsed)
	signalInput.requestFacts.Context = selectionRequestContext(ctx)
	ctx.VSRConversationFacts = signalInput.conversationFacts
	ctx.VSRContextHasNonText = ctx.VSRContextHasNonText ||
		signalInput.requestFacts.ContextHasNonText
	if signalInput.evaluationText == "" && !hasEnvelopeRoutingFacts(history) {
		return "", 0.0, entropy.ReasoningDecision{}, "", nil
	}

	candidates := r.decisionCandidatesForRequest(originalModel, ctx)
	signals, authzErr := r.evaluateSignalsForDecision(originalModel, signalInput, history.nonUserMessages, ctx, candidates)
	if authzErr != nil {
		return "", 0, entropy.ReasoningDecision{}, "", authzErr
	}

	result, defaultModel, err := r.runDecisionEngine(originalModel, ctx, signals, candidates)
	if err != nil {
		return "", 0, entropy.ReasoningDecision{}, "", err
	}
	if result == nil {
		return "", 0.0, entropy.ReasoningDecision{}, defaultModel, nil
	}

	decisionName, evaluationConfidence, reasoningDecision, selectedModel, err = r.finalizeDecisionEvaluation(
		result,
		originalModel,
		history.currentUserMessage,
		ctx,
	)
	return decisionName, evaluationConfidence, reasoningDecision, selectedModel, err
}

func (r *OpenAIRouter) prepareDecisionEvaluation(
	originalModel string,
	history signalConversationHistory,
	ctx *RequestContext,
) (string, bool) {
	if len(history.nonUserMessages) == 0 && history.currentUserMessage == "" &&
		!hasEnvelopeRoutingFacts(history) {
		return "", true
	}

	// Focused callers may invoke decision evaluation without the normal
	// pre-routing stage. Resolve idempotently so isolation never depends on
	// call order.
	if !ctx.Routing.IsResolved() {
		r.resolveEntrypointForRequest(originalModel, ctx)
	}
	if ctx.Routing.SelectedRecipe() == nil {
		return "", true
	}
	if r.Config.HasRoutingDecisions() {
		return "", false
	}
	return "", true
}

func (r *OpenAIRouter) selectorForDecisionMethod(method selection.SelectionMethod, algorithm *config.AlgorithmConfig, ctx *RequestContext) selection.Selector {
	if method == selection.MethodMultiFactor && algorithm != nil {
		return r.newDecisionMultiFactorSelector(algorithm.MultiFactor)
	}
	if method == selection.MethodSystemOne && algorithm != nil &&
		algorithm.SystemOne != nil {
		return r.newDecisionModelSelector(*algorithm.SystemOne)
	}
	if selector := r.extensionAlgorithmSelector(method, algorithm, ctx); selector != nil {
		return selector
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

func selectedModelRefFromResult(selCtx *selection.SelectionContext, result *selection.SelectionResult) *config.ModelRef {
	candidate, _ := selection.ResolveSelectionCandidate(selCtx, result)
	return candidate
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
	categoryName string,
	candidateIterations []config.CandidateIterationConfig,
	reqCtx *RequestContext,
) *selection.SelectionContext {
	sessionID, userID, conversationHistory := r.extractSessionContext(reqCtx)
	var recipeName config.RecipeName
	if reqCtx != nil {
		recipeName = reqCtx.Routing.RecipeName()
	}
	inputTokens, expectedOutputTokens := selectionTokenBudget(reqCtx)

	return &selection.SelectionContext{
		Query:                query,
		DecisionName:         decisionName,
		RecipeName:           recipeName,
		CategoryName:         categoryName,
		CandidateModels:      modelRefs,
		CandidateIterations:  candidateIterations,
		InputTokens:          inputTokens,
		ExpectedOutputTokens: expectedOutputTokens,
		UserID:               userID,
		SessionID:            sessionID,
		SessionStateKey:      sessiontelemetry.RoutingSessionKey(recipeName, sessionID),
		AgenticSession:       r.buildAgenticSessionContext(reqCtx, modelRefs, sessionID, userID),
		ConversationHistory:  conversationHistory,
		CacheAffinityCtx:     r.buildCacheAffinityContext(reqCtx, modelRefs),
	}
}

func selectionTokenBudget(reqCtx *RequestContext) (int, int) {
	if reqCtx == nil {
		return 0, 0
	}
	expectedOutput := 0
	if reqCtx.SemanticRequest != nil && reqCtx.SemanticRequest.Sampling.MaxOutputTokens != nil &&
		*reqCtx.SemanticRequest.Sampling.MaxOutputTokens > 0 {
		expectedOutput = int(*reqCtx.SemanticRequest.Sampling.MaxOutputTokens)
	}
	return reqCtx.VSRContextTokenCount, expectedOutput
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
	return r.buildAgenticSessionContextForKey(reqCtx, modelRefs, sessionID, userID,
		sessiontelemetry.RoutingSessionKey(reqCtx.Routing.RecipeName(), sessionID))
}

func (r *OpenAIRouter) buildAgenticSessionContextForKey(
	reqCtx *RequestContext,
	modelRefs []config.ModelRef,
	sessionID, userID, stateKey string,
) *selection.AgenticSessionContext {
	if reqCtx == nil {
		return nil
	}
	now := time.Now()
	var snapshot sessiontelemetry.RouterSessionSnapshot
	var hasMemory bool
	if reqCtx.learningPreview != nil {
		now = reqCtx.learningPreview.CapturedAt
		snapshot, hasMemory = reqCtx.learningPreview.session(stateKey)
	} else {
		snapshot, hasMemory = sessiontelemetry.GetRouterSessionSnapshot(stateKey, now)
	}
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
		PreviousCandidate:           snapshot.CurrentCandidate,
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
		MemoryUserTurnCount:         snapshot.UserTurnCount,
		MemoryLastReleaseUserTurn:   snapshot.LastReleaseUserTurn,
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
		// Router-owned history is fully expanded before signal extraction. Its
		// retained public ID is lineage metadata, not opaque provider state.
		state := reqCtx.ResponseObjectState
		if state != nil && state.ProviderContextApplied && state.PreviousResponseID == reqCtx.PreviousResponseID &&
			reqCtx.SemanticRequest != nil && reqCtx.SemanticRequest.PreviousResponseID == "" {
			return false, ""
		}
		return true, "previous_response_id"
	}
	return false, ""
}

func conversationFactsIndicateActiveToolLoop(facts classification.ConversationFacts) bool {
	return facts.LastMessageToolResult ||
		facts.LastMessageRole == "tool" ||
		facts.LastUserAfterToolResult ||
		facts.LastAssistantToolCall
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
	var ambient float64
	var ambientOK bool
	if reqCtx.learningPreview != nil {
		ambient, ambientOK = reqCtx.learningPreview.warmth(previousModel)
	} else {
		ambient, ambientOK = estimateGateCacheWarmth(previousModel, now)
	}
	if ambientOK {
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
	affinity := &selection.CacheAffinityContext{
		TurnIndex:           reqCtx.TurnIndex,
		PreviousModel:       reqCtx.PreviousModel,
		PreviousResponseID:  reqCtx.PreviousResponseID,
		HistoryTokens:       reqCtx.HistoryTokenCount,
		ContextTokens:       reqCtx.VSRContextTokenCount,
		ModelContextWindows: r.modelContextWindows(modelRefs),
	}
	if reqCtx.SemanticRequest != nil {
		affinity.PromptCacheKey = strings.TrimSpace(reqCtx.SemanticRequest.PromptCacheKey)
	}
	if affinity.PreviousModel == "" {
		affinity.PreviousModel = promptCacheKeyModel(reqCtx)
	}
	return affinity
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
// extractSessionContext extracts session ID, user ID, and conversation history from the RequestContext.
func (r *OpenAIRouter) extractSessionContext(ctx *RequestContext) (sessionID, userID string, conversationHistory []string) {
	if ctx == nil {
		return "", "", nil
	}
	userID = extractUserID(ctx)
	sessionID = ctx.SessionID
	if state := ctx.ResponseObjectState; state != nil {
		if sessionID == "" {
			sessionID = state.ConversationID
		}
		if !state.ProviderContextApplied {
			conversationHistory = appendStoredConversationHistory(conversationHistory, state)
		}
	}
	if ctx.SemanticRequest == nil {
		return sessionID, userID, conversationHistory
	}
	if sessionID == "" {
		sessionID = deriveSessionIDFromSemanticMessages(ctx.SemanticRequest.Messages, userID)
	}
	conversationHistory = appendSemanticConversationHistory(conversationHistory, ctx.SemanticRequest.Messages)
	return sessionID, userID, conversationHistory
}

func appendStoredConversationHistory(
	history []string,
	state *ResponseObjectState,
) []string {
	for _, storedResponse := range state.ConversationHistory {
		for _, input := range storedResponse.Input {
			if content := extractContentFromInputItem(input); content != "" {
				history = append(history, content)
			}
		}
		for _, output := range storedResponse.Output {
			if content := extractContentFromOutputItem(output); content != "" {
				history = append(history, content)
			}
		}
	}
	return history
}

func appendSemanticConversationHistory(
	history []string,
	messages []llmprotocol.Message,
) []string {
	for index := 0; index < len(messages)-1; index++ {
		if content := semanticText(messages[index].Content); content != "" {
			history = append(history, content)
		}
	}
	return history
}
