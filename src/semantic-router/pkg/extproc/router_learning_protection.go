package extproc

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

type routerLearningProtectionPreflight struct {
	enabled         bool
	mode            string
	scope           string
	samplingAllowed bool
	identity        routerLearningIdentity
	config          config.RouterLearningProtectionConfig
	policy          routerLearningPolicy
}

func (r *OpenAIRouter) applyProtectionPreflight(input routerLearningInput) routerLearningProtectionPreflight {
	cfg, ok := r.protectionConfig(input.selCtx, input.baseResult, input.selectedModelRef, input.ctx)
	mode := protectionMode(input.ctx)
	scope := cfg.EffectiveScope()
	preflight := routerLearningProtectionPreflight{
		enabled: false,
		mode:    mode,
		scope:   scope,
		config:  cfg,
	}
	if !ok {
		return preflight
	}
	if mode == config.DecisionAdaptationModeBypass {
		preflight.policy = newProtectionPolicy(input.ctx, cfg, mode, routerLearningActionBypass, "decision_bypass", scope)
		return preflight
	}
	identity, identityOK := r.protectionIdentity(input.ctx, cfg)
	if !identityOK {
		preflight.policy = newProtectionPolicy(input.ctx, cfg, mode, routerLearningActionSuppressSampling, "missing_identity", scope)
		return preflight
	}
	if input.ctx != nil {
		input.ctx.VSRLearningSessionID = identity.memoryKey
		input.ctx.VSRLearningConversationID = identity.conversationID
	}

	preflight.enabled = true
	preflight.identity = identity
	if mode == config.DecisionAdaptationModeObserve {
		preflight.samplingAllowed = true
		preflight.policy = newProtectionPolicy(input.ctx, cfg, mode, routerLearningActionObserve, "observe_only", scope)
		return preflight
	}
	samplingAllowed, reason := protectionSamplingDecision(input, mode)
	preflight.samplingAllowed = samplingAllowed
	action := routerLearningActionSuppressSampling
	if preflight.samplingAllowed {
		action = routerLearningActionAllowSampling
	}
	preflight.policy = newProtectionPolicy(input.ctx, cfg, mode, action, reason, scope)
	preflight.policy.Details.Protection.samplingPolicy = samplingPolicyValue(preflight.samplingAllowed)
	return preflight
}

func protectionSamplingDecision(input routerLearningInput, mode string) (bool, string) {
	if mode != config.DecisionAdaptationModeApply {
		return false, "observe_or_bypass"
	}
	if input.ctx == nil {
		return true, "no_tool_or_protocol_state"
	}
	if decisionSuppressesSampling(input.ctx) {
		return false, "decision_mode_not_apply"
	}
	if facts := input.ctx.VSRConversationFacts; conversationFactsIndicateActiveToolLoop(facts) {
		return false, "tool_or_protocol_state"
	}
	return agenticSessionSamplingDecision(input.selCtx)
}

func decisionSuppressesSampling(ctx *RequestContext) bool {
	return ctx != nil &&
		ctx.VSRSelectedDecision != nil &&
		ctx.VSRSelectedDecision.Adaptations.AdaptationMode() != config.DecisionAdaptationModeApply
}

func agenticSessionSamplingDecision(selCtx *selection.SelectionContext) (bool, string) {
	if selCtx == nil || selCtx.AgenticSession == nil {
		return true, "no_tool_or_protocol_state"
	}
	session := selCtx.AgenticSession
	if session.ActiveToolLoop || session.HasNonPortableContext {
		return false, "tool_or_protocol_state"
	}
	if sessionHasWarmPreviousModel(session) {
		return false, "steady_state"
	}
	return true, "no_tool_or_protocol_state"
}

func sessionHasWarmPreviousModel(session *selection.AgenticSessionContext) bool {
	return session != nil &&
		strings.TrimSpace(session.PreviousModel) != "" &&
		session.CacheWarmthOK &&
		session.CacheWarmth >= 0.65
}

func samplingPolicyValue(allowed bool) string {
	if allowed {
		return "allowed"
	}
	return "suppressed"
}

func (r *OpenAIRouter) applyProtectionSwitch(
	input routerLearningInput,
	preflight routerLearningProtectionPreflight,
	proposal routerLearningDecision,
) routerLearningDecision {
	baseResult := firstNonNilSelectionResult(proposal.selectionResult, input.baseResult)
	baseCtx := firstNonNilSelectionContext(proposal.selectionContext, input.selCtx)
	baseRef := firstNonNilModelRef(proposal.selectedModelRef, input.selectedModelRef)
	decision := routerLearningDecision{
		selectionContext: baseCtx,
		selectionResult:  baseResult,
		selectedModelRef: baseRef,
	}
	if !preflight.enabled || preflight.mode == config.DecisionAdaptationModeBypass {
		decision.changesModel = learningChangesModel(input.baseResult, baseResult)
		decision.policy = preflight.policy
		return decision
	}
	learningCtx := r.protectionSelectionContext(baseCtx, input.ctx, preflight.identity)
	r.addCurrentLearningCandidate(learningCtx, input.ctx)
	if rescue, ok := r.protectionRescueDecision(input, learningCtx, preflight, proposal); ok {
		return rescue
	}
	if protected, ok := sessionScopeProtectedResult(preflight.config, baseResult, learningCtx, preflight.identity); ok {
		return r.protectionDecisionFromResult(input, learningCtx, protected, preflight, proposal)
	}

	result, ok := r.selectProtectionResult(preflight.config, baseResult, learningCtx)
	if !ok {
		decision.selectionContext = input.selCtx
		decision.selectionResult = input.baseResult
		decision.selectedModelRef = input.selectedModelRef
		decision.policy = newProtectionPolicy(input.ctx, preflight.config, preflight.mode, routerLearningActionHoldCurrent, "protection_unavailable", preflight.scope)
		return decision
	}
	return r.protectionDecisionFromResult(input, learningCtx, result, preflight, proposal)
}

func (r *OpenAIRouter) protectionRescueDecision(
	input routerLearningInput,
	learningCtx *selection.SelectionContext,
	preflight routerLearningProtectionPreflight,
	proposal routerLearningDecision,
) (routerLearningDecision, bool) {
	if preflight.mode != config.DecisionAdaptationModeApply || learningCtx == nil {
		return routerLearningDecision{}, false
	}
	current := currentLearningModel(learningCtx)
	proposalResult := protectionRescueProposalResult(input, proposal, current)
	proposalModel := selectedModelName(proposalResult)
	if current == "" || proposalModel == "" || current == proposalModel {
		return routerLearningDecision{}, false
	}
	if !r.protectionRescueEvidence(preflight.config, learningCtx, input.ctx, current, proposalModel, proposalResult) {
		return routerLearningDecision{}, false
	}
	selected := selectedModelRefFromResult(learningCtx, proposalResult)
	if selected == nil {
		return routerLearningDecision{}, false
	}
	result := *proposalResult
	result.AllScores = cloneSelectionScores(proposalResult.AllScores)
	result.Reasoning = "router_learning protection: rescue switch"
	result.SessionPolicy = rescueProtectionTrace(
		preflight.config,
		input.baseResult,
		proposalResult,
		learningCtx,
		preflight.identity,
		current,
		proposalModel,
	)
	policy := protectionRescuePolicyFromSelectionResult(
		&result,
		preflight.identity,
		preflight.mode,
		selectedModelName(input.baseResult),
		proposalModel,
		proposalModel,
		preflight.config,
	)
	return routerLearningDecision{
		selectionContext: learningCtx,
		selectionResult:  &result,
		selectedModelRef: selected,
		changesModel:     learningChangesModel(input.baseResult, &result),
		policy:           policy,
	}, true
}

func protectionRescueProposalResult(
	input routerLearningInput,
	proposal routerLearningDecision,
	current string,
) *selection.SelectionResult {
	proposalResult := firstNonNilSelectionResult(proposal.selectionResult, input.baseResult)
	if selectedModelName(proposalResult) != current {
		return proposalResult
	}
	if selectedModelName(input.baseResult) != current {
		return input.baseResult
	}
	return proposalResult
}

func (r *OpenAIRouter) protectionRescueEvidence(
	cfg config.RouterLearningProtectionConfig,
	learningCtx *selection.SelectionContext,
	ctx *RequestContext,
	current string,
	proposal string,
	proposalResult *selection.SelectionResult,
) bool {
	tier := decisionTier(ctx)
	decision := ""
	if learningCtx != nil {
		decision = learningCtx.DecisionName
	}
	currentExp := r.routerLearningRuntimeState().experienceSnapshot(decision, tier, current)
	proposalExp := r.routerLearningRuntimeState().experienceSnapshot(decision, tier, proposal)
	currentWeak := currentExp.UnderpoweredCount >= 2 && currentExp.UnderpoweredCount > currentExp.GoodFitCount
	currentUnreliable := currentExp.FailedCount >= 2
	if !currentWeak && !currentUnreliable {
		return false
	}
	if proposalExp.GoodFitCount > proposalExp.UnderpoweredCount {
		return true
	}
	proposalScore := scoreFromSelectionResult(proposalResult, proposal)
	currentScore := scoreFromSelectionResult(proposalResult, current)
	return proposalScore-currentScore >= learningSwitchMargin(cfg)
}

func (r *OpenAIRouter) protectionDecisionFromResult(
	input routerLearningInput,
	learningCtx *selection.SelectionContext,
	result *selection.SelectionResult,
	preflight routerLearningProtectionPreflight,
	proposal routerLearningDecision,
) routerLearningDecision {
	enteredCtx := firstNonNilSelectionContext(proposal.selectionContext, input.selCtx)
	enteredResult := firstNonNilSelectionResult(proposal.selectionResult, input.baseResult)
	enteredRef := firstNonNilModelRef(proposal.selectedModelRef, input.selectedModelRef)
	baseModel := selectedModelName(input.baseResult)
	proposalModel := selectedModelName(enteredResult)
	if preflight.mode == config.DecisionAdaptationModeObserve {
		policy := protectionPolicyFromSelectionResult(
			result,
			preflight.identity,
			preflight.mode,
			baseModel,
			proposalModel,
			proposalModel,
			preflight.config,
		)
		policy.Action = routerLearningActionObserve
		policy.Reason = "observe_only"
		return routerLearningDecision{
			selectionContext: enteredCtx,
			selectionResult:  enteredResult,
			selectedModelRef: enteredRef,
			changesModel:     learningChangesModel(input.baseResult, enteredResult),
			policy:           policy,
		}
	}
	selected := selectedModelRefFromResult(learningCtx, result)
	if selected == nil {
		logging.Warnf("[RouterLearning] protection selected %s but no model ref was available", result.SelectedModel)
		return routerLearningDecision{
			selectionContext: input.selCtx,
			selectionResult:  input.baseResult,
			selectedModelRef: input.selectedModelRef,
			policy:           newProtectionPolicy(input.ctx, preflight.config, preflight.mode, routerLearningActionHoldCurrent, "selected_model_missing", preflight.scope),
		}
	}
	finalModel := selectedModelName(result)
	policy := protectionPolicyFromSelectionResult(
		result,
		preflight.identity,
		preflight.mode,
		baseModel,
		proposalModel,
		finalModel,
		preflight.config,
	)
	return routerLearningDecision{
		selectionContext: learningCtx,
		selectionResult:  result,
		selectedModelRef: selected,
		changesModel:     learningChangesModel(input.baseResult, result),
		policy:           policy,
	}
}

func (r *OpenAIRouter) protectionConfig(
	selCtx *selection.SelectionContext,
	baseResult *selection.SelectionResult,
	selectedModelRef *config.ModelRef,
	ctx *RequestContext,
) (config.RouterLearningProtectionConfig, bool) {
	if r == nil || r.Config == nil || selCtx == nil || baseResult == nil || selectedModelRef == nil || ctx == nil {
		return config.RouterLearningProtectionConfig{}, false
	}
	learningCfg := r.Config.RouterLearning
	cfg := learningCfg.Protection
	if ctx.VSRSelectedDecision != nil {
		cfg.Tuning = ctx.VSRSelectedDecision.Adaptations.ApplyProtectionTuning(cfg.Tuning)
	}
	return cfg, learningCfg.Enabled && cfg.EffectiveEnabled()
}

func protectionMode(ctx *RequestContext) string {
	if ctx != nil && ctx.VSRSelectedDecision != nil {
		return ctx.VSRSelectedDecision.Adaptations.ProtectionMode()
	}
	return config.DecisionAdaptationModeApply
}

func (r *OpenAIRouter) addCurrentLearningCandidate(learningCtx *selection.SelectionContext, ctx *RequestContext) {
	current := currentLearningModel(learningCtx)
	if current == "" || selectionContextContainsModel(learningCtx, current) || !r.configuredBackendModel(current) {
		return
	}
	learningCtx.CandidateModels = append(learningCtx.CandidateModels, config.ModelRef{Model: current})
	learningCtx.CacheAffinityCtx = r.buildCacheAffinityContext(ctx, learningCtx.CandidateModels)
	if learningCtx.AgenticSession != nil {
		learningCtx.AgenticSession.ModelContextWindows = r.modelContextWindows(learningCtx.CandidateModels)
	}
}

func (r *OpenAIRouter) selectProtectionResult(
	cfg config.RouterLearningProtectionConfig,
	baseResult *selection.SelectionResult,
	learningCtx *selection.SelectionContext,
) (*selection.SelectionResult, bool) {
	selector := selection.NewSessionAwareSelector(protectionSelectionConfig(cfg))
	selector.SetBaseSelector(learningSelectionResult{result: baseResult})
	if r.Config.ModelConfig != nil {
		selector.InitializeFromConfig(r.Config.ModelConfig)
	}
	if r.LookupTable != nil {
		selector.SetLookupTable(r.LookupTable)
	}

	result, err := selector.Select(context.Background(), learningCtx)
	if err != nil {
		logging.Warnf("[RouterLearning] protection failed: %v", err)
		return nil, false
	}
	if err := selection.ValidateSelectionResult(learningCtx, result); err != nil {
		logging.Warnf("[RouterLearning] protection produced invalid result: %v", err)
		return nil, false
	}
	return result, true
}

func (r *OpenAIRouter) protectionIdentity(
	ctx *RequestContext,
	cfg config.RouterLearningProtectionConfig,
) (routerLearningIdentity, bool) {
	scope := cfg.EffectiveScope()
	sessionHeader := cfg.HeaderName("session")
	conversationHeader := cfg.HeaderName("conversation")
	sessionID := strings.TrimSpace(headerValueCI(ctx, sessionHeader))
	conversationID := strings.TrimSpace(headerValueCI(ctx, conversationHeader))
	if sessionID == "" {
		return routerLearningIdentity{}, false
	}
	memoryKey := sessionID
	if scope == config.RouterLearningScopeConversation {
		if conversationID == "" {
			return routerLearningIdentity{}, false
		}
		memoryKey = fmt.Sprintf("%s/%s", sessionID, conversationID)
	}
	return routerLearningIdentity{
		sessionID:          sessionID,
		conversationID:     conversationID,
		memoryKey:          memoryKey,
		scope:              scope,
		sessionHeader:      sessionHeader,
		conversationHeader: conversationHeader,
	}, true
}

func (r *OpenAIRouter) protectionSelectionContext(
	selCtx *selection.SelectionContext,
	ctx *RequestContext,
	identity routerLearningIdentity,
) *selection.SelectionContext {
	learningReqCtx := *ctx
	learningReqCtx.SessionID = identity.memoryKey
	learningReqCtx.PreviousModel = ""
	learningReqCtx.SessionIdleKnown = false
	learningReqCtx.SessionIdleSeconds = 0

	learningCtx := *selCtx
	learningCtx.SessionID = identity.memoryKey
	learningCtx.AgenticSession = r.buildAgenticSessionContext(
		&learningReqCtx,
		learningCtx.CandidateModels,
		identity.memoryKey,
		learningCtx.UserID,
	)
	learningCtx.CacheAffinityCtx = r.buildCacheAffinityContext(&learningReqCtx, learningCtx.CandidateModels)
	return &learningCtx
}

func protectionSelectionConfig(cfg config.RouterLearningProtectionConfig) *selection.SessionAwareConfig {
	result := selection.DefaultSessionAwareConfig()
	result.DecisionDriftReset = false
	tuning := cfg.Tuning
	if tuning.IdleTimeoutSeconds != nil {
		result.IdleTimeoutSeconds = *tuning.IdleTimeoutSeconds
	}
	if tuning.MinTurnsBeforeSwitch != nil {
		result.MinTurnsBeforeSwitch = *tuning.MinTurnsBeforeSwitch
	}
	if tuning.SwitchMargin != nil {
		result.SwitchMargin = *tuning.SwitchMargin
	}
	applyProtectionStabilityWeight(result, learningProtectionStabilityWeight(cfg))
	return result
}

func applyProtectionStabilityWeight(result *selection.SessionAwareConfig, stabilityWeight float64) {
	if result == nil || stabilityWeight == 1 {
		return
	}
	if stabilityWeight < 0 {
		stabilityWeight = 1
	}
	result.PrefixCacheWeight *= stabilityWeight
	result.HandoffPenaltyWeight *= stabilityWeight
	result.SwitchHistoryWeight *= stabilityWeight
	result.ToolLoopStayBias *= stabilityWeight
}

func sessionScopeProtectedResult(
	cfg config.RouterLearningProtectionConfig,
	baseResult *selection.SelectionResult,
	learningCtx *selection.SelectionContext,
	identity routerLearningIdentity,
) (*selection.SelectionResult, bool) {
	if identity.scope != config.RouterLearningScopeSession ||
		baseResult == nil ||
		learningCtx == nil ||
		learningCtx.AgenticSession == nil {
		return nil, false
	}
	session := learningCtx.AgenticSession
	current := strings.TrimSpace(session.PreviousModel)
	if current == "" || !selectionContextContainsModel(learningCtx, current) {
		return nil, false
	}
	if sessionScopeIdleExpired(cfg, session) {
		return nil, false
	}
	score := 0.0
	if baseResult.AllScores != nil {
		score = baseResult.AllScores[current]
	}
	allScores := cloneSelectionScores(baseResult.AllScores)
	if allScores == nil {
		allScores = map[string]float64{}
	}
	allScores[current] = score
	trace := sessionScopeProtectionTrace(cfg, baseResult, learningCtx, identity, current, score)
	return &selection.SelectionResult{
		SelectedModel: current,
		LoRAName:      sessionScopeProtectedLoRAName(learningCtx, current),
		Score:         score,
		Confidence:    1,
		Method:        baseResult.Method,
		Tier:          selection.TierSupported,
		Reasoning:     "router_learning protection: session scope protects current model",
		AllScores:     allScores,
		SessionPolicy: trace,
	}, true
}

func sessionScopeProtectionTrace(
	cfg config.RouterLearningProtectionConfig,
	baseResult *selection.SelectionResult,
	learningCtx *selection.SelectionContext,
	identity routerLearningIdentity,
	current string,
	score float64,
) *selection.SessionPolicyTrace {
	session := learningCtx.AgenticSession
	trace := &selection.SessionPolicyTrace{
		Algorithm:                   string(routerLearningMethodProtection),
		BaseMethod:                  string(baseResult.Method),
		SessionID:                   identity.memoryKey,
		UserID:                      learningCtx.UserID,
		Phase:                       session.Phase,
		CurrentModel:                current,
		BaseSelectedModel:           baseResult.SelectedModel,
		SelectedModel:               current,
		TurnIndex:                   session.TurnIndex,
		MemoryTurnCount:             session.MemoryTurnCount,
		SwitchCount:                 session.MemorySwitchCount,
		LastDecisionName:            session.LastDecisionName,
		MemoryPromptTokens:          session.MemoryPromptTokens,
		MemoryCachedTokens:          session.MemoryCachedTokens,
		MemoryEstimatedCachedTokens: session.MemoryEstimatedCachedTokens,
		MemoryEstimatedCacheSavings: session.MemoryEstimatedCacheSavings,
		LastCacheAccountingSource:   session.MemoryCacheAccountingSource,
		ActiveToolLoop:              session.ActiveToolLoop,
		IdleKnown:                   session.IdleKnown,
		IdleForSeconds:              session.IdleFor.Seconds(),
		IdleExpired:                 false,
		HasNonPortableContext:       session.HasNonPortableContext,
		NonPortableContextReason:    session.NonPortableContextReason,
		DecisionDrift:               session.LastDecisionName != "" && session.LastDecisionName != learningCtx.DecisionName,
		DecisionReason:              "session_scope_protect",
		CacheWarmth:                 session.CacheWarmth,
		CacheWarmthOK:               session.CacheWarmthOK,
		SwitchMargin:                learningSwitchMargin(cfg),
		BaseScores:                  cloneSelectionScores(baseResult.AllScores),
		FinalScores:                 cloneSelectionScores(baseResult.AllScores),
		CandidateTraces:             map[string]selection.SessionCandidateTrace{},
	}
	if trace.FinalScores == nil {
		trace.FinalScores = map[string]float64{}
	}
	trace.FinalScores[current] = score
	trace.CandidateTraces[current] = selection.SessionCandidateTrace{
		Current:    true,
		BaseScore:  score,
		FinalScore: score,
	}
	return trace
}

func rescueProtectionTrace(
	cfg config.RouterLearningProtectionConfig,
	baseResult *selection.SelectionResult,
	proposalResult *selection.SelectionResult,
	learningCtx *selection.SelectionContext,
	identity routerLearningIdentity,
	current string,
	proposal string,
) *selection.SessionPolicyTrace {
	session := learningCtx.AgenticSession
	baseMethod := ""
	baseSelected := ""
	var baseScores map[string]float64
	if baseResult != nil {
		baseMethod = string(baseResult.Method)
		baseSelected = selectedModelName(baseResult)
		baseScores = cloneSelectionScores(baseResult.AllScores)
	}
	finalScores := cloneSelectionScores(nil)
	if proposalResult != nil {
		finalScores = cloneSelectionScores(proposalResult.AllScores)
	}
	if finalScores == nil {
		finalScores = map[string]float64{}
	}
	currentScore := scoreFromSelectionResult(proposalResult, current)
	proposalScore := scoreFromSelectionResult(proposalResult, proposal)
	finalScores[current] = currentScore
	finalScores[proposal] = proposalScore
	trace := &selection.SessionPolicyTrace{
		Algorithm:         string(routerLearningMethodProtection),
		BaseMethod:        baseMethod,
		SessionID:         identity.memoryKey,
		UserID:            learningCtx.UserID,
		CurrentModel:      current,
		BaseSelectedModel: baseSelected,
		SelectedModel:     proposal,
		DecisionReason:    "rescue_underpowered_model",
		SwitchMargin:      learningSwitchMargin(cfg),
		BaseScores:        baseScores,
		FinalScores:       finalScores,
		CandidateTraces: map[string]selection.SessionCandidateTrace{
			current: {
				Current:    true,
				BaseScore:  currentScore,
				FinalScore: currentScore,
			},
			proposal: {
				BaseScore:          proposalScore,
				FinalScore:         proposalScore,
				NetSwitchAdvantage: proposalScore - currentScore,
			},
		},
	}
	if session == nil {
		return trace
	}
	trace.Phase = session.Phase
	trace.TurnIndex = session.TurnIndex
	trace.MemoryTurnCount = session.MemoryTurnCount
	trace.SwitchCount = session.MemorySwitchCount
	trace.LastDecisionName = session.LastDecisionName
	trace.MemoryPromptTokens = session.MemoryPromptTokens
	trace.MemoryCachedTokens = session.MemoryCachedTokens
	trace.MemoryEstimatedCachedTokens = session.MemoryEstimatedCachedTokens
	trace.MemoryEstimatedCacheSavings = session.MemoryEstimatedCacheSavings
	trace.LastCacheAccountingSource = session.MemoryCacheAccountingSource
	trace.ActiveToolLoop = session.ActiveToolLoop
	trace.IdleKnown = session.IdleKnown
	trace.IdleForSeconds = session.IdleFor.Seconds()
	trace.IdleExpired = sessionScopeIdleExpired(cfg, session)
	trace.HasNonPortableContext = session.HasNonPortableContext
	trace.NonPortableContextReason = session.NonPortableContextReason
	trace.DecisionDrift = session.LastDecisionName != "" && session.LastDecisionName != learningCtx.DecisionName
	trace.CacheWarmth = session.CacheWarmth
	trace.CacheWarmthOK = session.CacheWarmthOK
	return trace
}

func sessionScopeProtectedLoRAName(learningCtx *selection.SelectionContext, current string) string {
	for _, candidate := range learningCtx.CandidateModels {
		if candidate.Model == current || candidate.LoRAName == current {
			return candidate.LoRAName
		}
	}
	return ""
}

func sessionScopeIdleExpired(cfg config.RouterLearningProtectionConfig, session *selection.AgenticSessionContext) bool {
	if session == nil || !session.IdleKnown {
		return false
	}
	timeoutSeconds := selection.DefaultSessionAwareConfig().IdleTimeoutSeconds
	if cfg.Tuning.IdleTimeoutSeconds != nil {
		timeoutSeconds = *cfg.Tuning.IdleTimeoutSeconds
	}
	if timeoutSeconds <= 0 {
		return false
	}
	return session.IdleFor >= time.Duration(timeoutSeconds)*time.Second
}

func learningSwitchMargin(cfg config.RouterLearningProtectionConfig) float64 {
	if cfg.Tuning.SwitchMargin != nil {
		return *cfg.Tuning.SwitchMargin
	}
	return selection.DefaultSessionAwareConfig().SwitchMargin
}

func learningProtectionStabilityWeight(cfg config.RouterLearningProtectionConfig) float64 {
	if cfg.Tuning.StabilityWeight != nil {
		return *cfg.Tuning.StabilityWeight
	}
	return 1
}

func scoreFromSelectionResult(result *selection.SelectionResult, model string) float64 {
	if result == nil || strings.TrimSpace(model) == "" {
		return 0
	}
	if result.SelectedModel == model {
		return result.Score
	}
	if result.AllScores != nil {
		return result.AllScores[model]
	}
	return 0
}
