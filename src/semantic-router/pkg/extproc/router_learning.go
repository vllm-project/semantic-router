package extproc

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

type learningSelectionResult struct {
	result *selection.SelectionResult
}

func (s learningSelectionResult) Select(_ context.Context, selCtx *selection.SelectionContext) (*selection.SelectionResult, error) {
	if s.result == nil {
		return nil, selection.ErrSelectionResultRequired
	}
	result := *s.result
	result.AllScores = cloneSelectionScores(s.result.AllScores)
	if result.AllScores == nil {
		result.AllScores = make(map[string]float64, len(selCtx.CandidateModels))
	}
	for _, candidate := range selCtx.CandidateModels {
		if _, ok := result.AllScores[candidate.Model]; !ok {
			result.AllScores[candidate.Model] = 0
		}
	}
	if _, ok := result.AllScores[result.SelectedModel]; !ok {
		result.AllScores[result.SelectedModel] = result.Score
	}
	return &result, nil
}

func (s learningSelectionResult) Method() selection.SelectionMethod {
	if s.result == nil || s.result.Method == "" {
		return selection.MethodStatic
	}
	return s.result.Method
}

func (s learningSelectionResult) UpdateFeedback(context.Context, *selection.Feedback) error {
	return nil
}
func (s learningSelectionResult) Tier() selection.AlgorithmTier                { return selection.TierSupported }
func (s learningSelectionResult) ExternalDependencies() []selection.Dependency { return nil }

type sessionAwareLearningIdentity struct {
	sessionID          string
	conversationID     string
	memoryKey          string
	scope              string
	sessionHeader      string
	conversationHeader string
}

func (r *OpenAIRouter) applyRouterLearning(
	selCtx *selection.SelectionContext,
	baseResult *selection.SelectionResult,
	selectedModelRef *config.ModelRef,
	ctx *RequestContext,
) (*selection.SelectionContext, *selection.SelectionResult, *config.ModelRef, bool) {
	learningSelCtx, learningResult, learningSelected, applied := r.applySessionAwareLearning(
		selCtx,
		baseResult,
		selectedModelRef,
		ctx,
	)
	if applied {
		return learningSelCtx, learningResult, learningSelected, true
	}
	return selCtx, baseResult, selectedModelRef, false
}

func (r *OpenAIRouter) applySessionAwareLearning(
	selCtx *selection.SelectionContext,
	baseResult *selection.SelectionResult,
	selectedModelRef *config.ModelRef,
	ctx *RequestContext,
) (*selection.SelectionContext, *selection.SelectionResult, *config.ModelRef, bool) {
	sessionAwareCfg, ok := r.sessionAwareLearningConfig(selCtx, baseResult, selectedModelRef, ctx)
	if !ok {
		return selCtx, baseResult, selectedModelRef, false
	}

	mode := sessionAwareAdaptationMode(ctx)
	if mode == config.DecisionAdaptationModeBypass {
		setSessionAwareLearningPolicy(ctx, sessionAwareCfg, mode, "bypass", "decision_bypass", sessionAwareCfg.EffectiveScope())
		return selCtx, baseResult, selectedModelRef, false
	}

	identity, ok := r.sessionAwareLearningIdentity(ctx, sessionAwareCfg)
	if !ok {
		setSessionAwareLearningPolicy(ctx, sessionAwareCfg, mode, "noop", "identity_missing", sessionAwareCfg.EffectiveScope())
		return selCtx, baseResult, selectedModelRef, false
	}

	learningCtx := r.learningSelectionContext(selCtx, ctx, identity)
	r.addCurrentLearningCandidate(learningCtx, ctx)

	protectedResult, protected := sessionScopeProtectedResult(sessionAwareCfg, baseResult, learningCtx, identity)
	if protected {
		recordSessionAwareLearningPolicy(ctx, protectedResult, identity, mode)
		if mode == config.DecisionAdaptationModeObserve {
			return learningCtx, baseResult, selectedModelRef, true
		}
		learningSelected := selectedModelRefFromResult(learningCtx, protectedResult)
		if learningSelected == nil {
			logging.Warnf("[RouterLearning] session_aware session-scope protection selected %s but no model ref was available", protectedResult.SelectedModel)
			return selCtx, baseResult, selectedModelRef, false
		}
		return learningCtx, protectedResult, learningSelected, true
	}

	learningResult, ok := r.selectSessionAwareLearningResult(sessionAwareCfg, baseResult, learningCtx)
	if !ok {
		return selCtx, baseResult, selectedModelRef, false
	}

	recordSessionAwareLearningPolicy(ctx, learningResult, identity, mode)

	if mode == config.DecisionAdaptationModeObserve {
		return learningCtx, baseResult, selectedModelRef, true
	}

	learningSelected := selectedModelRefFromResult(learningCtx, learningResult)
	if learningSelected == nil {
		logging.Warnf("[RouterLearning] session_aware selected %s but no model ref was available", learningResult.SelectedModel)
		return selCtx, baseResult, selectedModelRef, false
	}
	return learningCtx, learningResult, learningSelected, true
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

func setSessionAwareLearningPolicy(ctx *RequestContext, cfg config.SessionAwareLearningConfig, mode, action, reason, scope string) {
	policy := sessionAwareLearningPolicyMap(map[string]interface{}{
		"adaptation": "session_aware",
		"mode":       mode,
		"action":     action,
		"reason":     reason,
	})
	if scope != "" {
		policy["scope"] = scope
	}
	policy["identity"] = sessionAwareIdentityDiagnostics(
		scope,
		cfg.HeaderName("session"),
		cfg.HeaderName("conversation"),
		strings.TrimSpace(headerValueCI(ctx, cfg.HeaderName("session"))),
		strings.TrimSpace(headerValueCI(ctx, cfg.HeaderName("conversation"))),
		"",
	)
	ctx.VSRLearningPolicy = policy
	ctx.VSRSessionPolicy = policy
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

func recordSessionAwareLearningPolicy(
	ctx *RequestContext,
	learningResult *selection.SelectionResult,
	identity sessionAwareLearningIdentity,
	mode string,
) {
	policy := learningPolicyFromSessionAwareResult(learningResult, identity, mode)
	ctx.VSRLearningPolicy = policy
	ctx.VSRSessionPolicy = policy
	ctx.VSRLearningSessionID = identity.memoryKey
	ctx.VSRLearningConversationID = identity.conversationID
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

func sessionScopeProtectedResult(
	cfg config.SessionAwareLearningConfig,
	baseResult *selection.SelectionResult,
	learningCtx *selection.SelectionContext,
	identity sessionAwareLearningIdentity,
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
	idleExpired := sessionScopeIdleExpired(cfg, session)
	if idleExpired {
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
		Method:        selection.MethodSessionAware,
		Tier:          selection.TierSupported,
		Reasoning:     "session_aware: session scope protects current model",
		AllScores:     allScores,
		SessionPolicy: trace,
	}, true
}

func sessionScopeProtectionTrace(
	cfg config.SessionAwareLearningConfig,
	baseResult *selection.SelectionResult,
	learningCtx *selection.SelectionContext,
	identity sessionAwareLearningIdentity,
	current string,
	score float64,
) *selection.SessionPolicyTrace {
	session := learningCtx.AgenticSession
	trace := &selection.SessionPolicyTrace{
		Algorithm:                   string(selection.MethodSessionAware),
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

func sessionScopeProtectedLoRAName(learningCtx *selection.SelectionContext, current string) string {
	for _, candidate := range learningCtx.CandidateModels {
		if candidate.Model == current || candidate.LoRAName == current {
			return candidate.LoRAName
		}
	}
	return ""
}

func sessionScopeIdleExpired(cfg config.SessionAwareLearningConfig, session *selection.AgenticSessionContext) bool {
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

func learningSwitchMargin(cfg config.SessionAwareLearningConfig) float64 {
	if cfg.Tuning.SwitchMargin != nil {
		return *cfg.Tuning.SwitchMargin
	}
	return selection.DefaultSessionAwareConfig().SwitchMargin
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

func learningPolicyFromSessionAwareResult(
	result *selection.SelectionResult,
	identity sessionAwareLearningIdentity,
	mode string,
) map[string]interface{} {
	policy := map[string]interface{}{}
	if result != nil && result.SessionPolicy != nil {
		policy = result.SessionPolicy.ToMap()
	}
	policy["adaptation"] = "session_aware"
	policy["mode"] = mode
	policy["scope"] = identity.scope
	delete(policy, "session_id")
	delete(policy, "user_id")
	policy["identity"] = sessionAwareIdentityDiagnostics(
		identity.scope,
		identity.sessionHeader,
		identity.conversationHeader,
		identity.sessionID,
		identity.conversationID,
		identity.memoryKey,
	)
	annotateSessionAwareLearningAction(policy)
	return sessionAwareLearningPolicyMap(policy)
}

func sessionAwareIdentityDiagnostics(
	scope string,
	sessionHeader string,
	conversationHeader string,
	sessionID string,
	conversationID string,
	memoryKey string,
) map[string]interface{} {
	conversationRequired := scope == config.RouterLearningScopeConversation
	identity := map[string]interface{}{
		"scope": scope,
		"headers": map[string]interface{}{
			"session":      sessionHeader,
			"conversation": conversationHeader,
		},
		"session":      sessionAwareIdentityPart(sessionHeader, sessionID, true),
		"conversation": sessionAwareIdentityPart(conversationHeader, conversationID, conversationRequired),
	}
	if memoryKey != "" {
		identity["memory_key_hash"] = shortLearningIdentityHash(memoryKey)
	}
	return identity
}

func sessionAwareIdentityPart(headerName string, value string, required bool) map[string]interface{} {
	status := "not_required"
	if required {
		status = "missing"
	}
	part := map[string]interface{}{
		"source":   "header:" + headerName,
		"required": required,
		"status":   status,
	}
	if strings.TrimSpace(value) != "" {
		part["status"] = "present"
		part["hash"] = shortLearningIdentityHash(value)
	}
	return part
}

func shortLearningIdentityHash(value string) string {
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])[:16]
}

func annotateSessionAwareLearningAction(policy map[string]interface{}) {
	if policy == nil {
		return
	}
	if policy["action"] == nil {
		current := replayPolicyString(policy, "current_model")
		selected := replayPolicyString(policy, "selected_model")
		switch {
		case replayPolicyBool(policy, "hard_locked"):
			policy["action"] = "hard_lock"
		case current == "":
			policy["action"] = "select"
		case selected == current:
			policy["action"] = "stay"
		default:
			policy["action"] = "switch"
		}
	}
	if policy["reason"] == nil {
		reason := firstNonEmpty(
			replayPolicyString(policy, "hard_lock_reason"),
			replayPolicyString(policy, "decision_reason"),
		)
		if reason != "" {
			policy["reason"] = reason
		}
	}
}

func sessionAwareLearningPolicyMap(policy map[string]interface{}) map[string]interface{} {
	if policy == nil {
		policy = map[string]interface{}{}
	}
	policy["learning"] = "router_learning"
	return policy
}

func currentLearningModel(selCtx *selection.SelectionContext) string {
	if selCtx == nil || selCtx.AgenticSession == nil {
		return ""
	}
	return strings.TrimSpace(selCtx.AgenticSession.PreviousModel)
}

func selectionContextContainsModel(selCtx *selection.SelectionContext, model string) bool {
	if selCtx == nil || model == "" {
		return false
	}
	for _, candidate := range selCtx.CandidateModels {
		if candidate.Model == model || candidate.LoRAName == model {
			return true
		}
	}
	return false
}

func (r *OpenAIRouter) configuredBackendModel(model string) bool {
	if r == nil || r.Config == nil || strings.TrimSpace(model) == "" {
		return false
	}
	if _, ok := r.Config.ModelConfig[model]; ok {
		return true
	}
	return model == r.Config.DefaultModel
}

func cloneSelectionScores(scores map[string]float64) map[string]float64 {
	if scores == nil {
		return nil
	}
	clone := make(map[string]float64, len(scores))
	for key, value := range scores {
		clone[key] = value
	}
	return clone
}
