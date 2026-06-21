package extproc

import (
	"context"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func (r *OpenAIRouter) applySessionAwareLearning(
	input routerLearningInput,
) (routerLearningAdaptationResult, bool) {
	sessionAwareCfg, ok := r.sessionAwareLearningConfig(input.selCtx, input.baseResult, input.selectedModelRef, input.ctx)
	if !ok {
		return routerLearningAdaptationResult{}, false
	}

	mode := sessionAwareAdaptationMode(input.ctx)
	if mode == config.DecisionAdaptationModeBypass {
		result := sessionAwareNoChangeResult(input.ctx, sessionAwareCfg, mode, routerLearningActionBypass, "decision_bypass")
		return attachRouterLearningExperience(result, input.experience), true
	}

	identity, ok := r.sessionAwareLearningIdentity(input.ctx, sessionAwareCfg)
	if !ok {
		result := sessionAwareNoChangeResult(input.ctx, sessionAwareCfg, mode, routerLearningActionNoop, "identity_missing")
		return attachRouterLearningExperience(result, input.experience), true
	}
	if input.ctx != nil {
		input.ctx.VSRLearningSessionID = identity.memoryKey
		input.ctx.VSRLearningConversationID = identity.conversationID
	}

	learningCtx := r.learningSelectionContext(input.selCtx, input.ctx, identity)
	r.addCurrentLearningCandidate(learningCtx, input.ctx)

	protectedResult, protected := sessionScopeProtectedResult(sessionAwareCfg, input.baseResult, learningCtx, identity)
	if protected {
		result := sessionAwareSelectionAdaptationResult(learningCtx, protectedResult, identity, mode, input.baseResult)
		return attachRouterLearningExperience(result, input.experience), true
	}

	learningResult, ok := r.selectSessionAwareLearningResult(sessionAwareCfg, input.baseResult, learningCtx)
	if !ok {
		result := sessionAwareNoChangeResult(input.ctx, sessionAwareCfg, mode, routerLearningActionNoop, routerLearningReasonNoopResult)
		return attachRouterLearningExperience(result, input.experience), true
	}

	result := sessionAwareSelectionAdaptationResult(learningCtx, learningResult, identity, mode, input.baseResult)
	return attachRouterLearningExperience(result, input.experience), true
}

func sessionAwareNoChangeResult(
	ctx *RequestContext,
	cfg config.SessionAwareLearningConfig,
	mode string,
	action routerLearningAction,
	reason string,
) routerLearningAdaptationResult {
	scope := cfg.EffectiveScope()
	return routerLearningAdaptationResult{
		method: routerLearningMethodSessionAware,
		mode:   mode,
		scope:  scope,
		action: action,
		reason: reason,
		policy: buildSessionAwareLearningPolicy(ctx, cfg, mode, action, reason, scope),
	}
}

func sessionAwareSelectionAdaptationResult(
	learningCtx *selection.SelectionContext,
	learningResult *selection.SelectionResult,
	identity sessionAwareLearningIdentity,
	mode string,
	baseResult *selection.SelectionResult,
) routerLearningAdaptationResult {
	learningSelected := selectedModelRefFromResult(learningCtx, learningResult)
	if learningSelected == nil {
		logging.Warnf("[RouterLearning] session_aware selected %s but no model ref was available", learningResult.SelectedModel)
		return routerLearningAdaptationResult{
			method:           routerLearningMethodSessionAware,
			mode:             mode,
			scope:            identity.scope,
			action:           routerLearningActionNoop,
			reason:           "selected_model_missing",
			selectionContext: learningCtx,
			selectionResult:  learningResult,
			policy:           learningPolicyFromSessionAwareResult(learningResult, identity, mode),
		}
	}
	policy := learningPolicyFromSessionAwareResult(learningResult, identity, mode)
	return routerLearningAdaptationResult{
		method:           routerLearningMethodSessionAware,
		mode:             mode,
		scope:            identity.scope,
		action:           policy.Action,
		reason:           policy.Reason,
		hard:             sessionAwarePolicyHardLocked(policy),
		changesModel:     mode != config.DecisionAdaptationModeObserve && learningChangesModel(baseResult, learningResult),
		selectionContext: learningCtx,
		selectionResult:  learningResult,
		selectedModelRef: learningSelected,
		policy:           policy,
	}
}

func sessionAwarePolicyHardLocked(policy routerLearningPolicy) bool {
	return policy.HardLocked()
}

func (r *OpenAIRouter) sessionAwareLearningConfig(
	selCtx *selection.SelectionContext,
	baseResult *selection.SelectionResult,
	selectedModelRef *config.ModelRef,
	ctx *RequestContext,
) (config.SessionAwareLearningConfig, bool) {
	if r == nil || r.Config == nil || selCtx == nil || baseResult == nil || selectedModelRef == nil || ctx == nil {
		return config.SessionAwareLearningConfig{}, false
	}
	learningCfg := r.Config.RouterLearning
	sessionAwareCfg := learningCfg.Adaptations.SessionAware
	sessionAwareCfg = sessionAwareConfigWithDecisionOverrides(sessionAwareCfg, ctx.VSRSelectedDecision)
	return sessionAwareCfg, learningCfg.Enabled && sessionAwareCfg.Enabled
}

func sessionAwareAdaptationMode(ctx *RequestContext) string {
	if ctx != nil && ctx.VSRSelectedDecision != nil {
		return ctx.VSRSelectedDecision.Adaptations.SessionAwareMode()
	}
	return config.DecisionAdaptationModeApply
}

func sessionAwareConfigWithDecisionOverrides(
	base config.SessionAwareLearningConfig,
	decision *config.Decision,
) config.SessionAwareLearningConfig {
	if decision == nil || decision.Adaptations.SessionAware == nil {
		return base
	}
	override := decision.Adaptations.SessionAware
	if strings.TrimSpace(override.Scope) != "" {
		base.Scope = override.Scope
	}
	base.Tuning = mergeSessionAwareLearningTuning(base.Tuning, override.Tuning)
	return base
}

func mergeSessionAwareLearningTuning(
	base config.SessionAwareLearningTuning,
	override config.SessionAwareLearningTuning,
) config.SessionAwareLearningTuning {
	if override.IdleTimeoutSeconds != nil {
		base.IdleTimeoutSeconds = override.IdleTimeoutSeconds
	}
	if override.MinTurnsBeforeSwitch != nil {
		base.MinTurnsBeforeSwitch = override.MinTurnsBeforeSwitch
	}
	if override.SwitchMargin != nil {
		base.SwitchMargin = override.SwitchMargin
	}
	if override.CacheWeight != nil {
		base.CacheWeight = override.CacheWeight
	}
	if override.HandoffPenalty != nil {
		base.HandoffPenalty = override.HandoffPenalty
	}
	if override.HandoffPenaltyWeight != nil {
		base.HandoffPenaltyWeight = override.HandoffPenaltyWeight
	}
	if override.SwitchHistoryWeight != nil {
		base.SwitchHistoryWeight = override.SwitchHistoryWeight
	}
	if override.MaxCacheCostMultiplier != nil {
		base.MaxCacheCostMultiplier = override.MaxCacheCostMultiplier
	}
	return base
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

func (r *OpenAIRouter) selectSessionAwareLearningResult(
	cfg config.SessionAwareLearningConfig,
	baseResult *selection.SelectionResult,
	learningCtx *selection.SelectionContext,
) (*selection.SelectionResult, bool) {
	selector := selection.NewSessionAwareSelector(sessionAwareSelectionConfigFromLearning(cfg))
	selector.SetBaseSelector(learningSelectionResult{result: baseResult})
	if r.Config.ModelConfig != nil {
		selector.InitializeFromConfig(r.Config.ModelConfig)
	}
	if r.LookupTable != nil {
		selector.SetLookupTable(r.LookupTable)
	}

	learningResult, err := selector.Select(context.Background(), learningCtx)
	if err != nil {
		logging.Warnf("[RouterLearning] session_aware adaptation failed: %v", err)
		return nil, false
	}
	if err := selection.ValidateSelectionResult(learningCtx, learningResult); err != nil {
		logging.Warnf("[RouterLearning] session_aware produced invalid result: %v", err)
		return nil, false
	}
	return learningResult, true
}

func (r *OpenAIRouter) sessionAwareLearningIdentity(
	ctx *RequestContext,
	cfg config.SessionAwareLearningConfig,
) (sessionAwareLearningIdentity, bool) {
	scope := cfg.EffectiveScope()
	sessionHeader := cfg.HeaderName("session")
	conversationHeader := cfg.HeaderName("conversation")
	sessionID := strings.TrimSpace(headerValueCI(ctx, sessionHeader))
	conversationID := strings.TrimSpace(headerValueCI(ctx, conversationHeader))
	if sessionID == "" {
		return sessionAwareLearningIdentity{}, false
	}
	memoryKey := sessionID
	if scope == config.RouterLearningScopeConversation {
		if conversationID == "" {
			return sessionAwareLearningIdentity{}, false
		}
		memoryKey = fmt.Sprintf("%s/%s", sessionID, conversationID)
	}
	return sessionAwareLearningIdentity{
		sessionID:          sessionID,
		conversationID:     conversationID,
		memoryKey:          memoryKey,
		scope:              scope,
		sessionHeader:      sessionHeader,
		conversationHeader: conversationHeader,
	}, true
}

func (r *OpenAIRouter) learningSelectionContext(
	selCtx *selection.SelectionContext,
	ctx *RequestContext,
	identity sessionAwareLearningIdentity,
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

func sessionAwareSelectionConfigFromLearning(cfg config.SessionAwareLearningConfig) *selection.SessionAwareConfig {
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
	if tuning.CacheWeight != nil {
		result.PrefixCacheWeight = *tuning.CacheWeight
	}
	if tuning.HandoffPenalty != nil {
		result.DefaultHandoffPenalty = *tuning.HandoffPenalty
	}
	if tuning.HandoffPenaltyWeight != nil {
		result.HandoffPenaltyWeight = *tuning.HandoffPenaltyWeight
	}
	if tuning.SwitchHistoryWeight != nil {
		result.SwitchHistoryWeight = *tuning.SwitchHistoryWeight
	}
	if tuning.MaxCacheCostMultiplier != nil {
		result.MaxCacheCostMultiplier = *tuning.MaxCacheCostMultiplier
	}
	return result
}
