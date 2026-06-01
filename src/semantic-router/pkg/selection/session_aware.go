package selection

import (
	"context"
	"fmt"
	"math"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/lookuptable"
)

// SessionAwareConfig configures agentic session-aware model selection.
type SessionAwareConfig struct {
	BaseMethod                   SelectionMethod
	IdleTimeoutSeconds           int
	MinTurnsBeforeSwitch         int
	SwitchMargin                 float64
	StayBias                     float64
	ToolLoopHardLock             bool
	ContextPortabilityHardLock   bool
	DecisionDriftReset           bool
	ToolLoopStayBias             float64
	PrefixCacheWeight            float64
	HandoffPenaltyWeight         float64
	DefaultHandoffPenalty        float64
	QualityGapMultiplier         float64
	MaxCacheCostMultiplier       float64
	SwitchHistoryWeight          float64
	RemainingTurnPriorWeight     float64
	RemainingTurnPriorHorizon    int
	MinRemainingTurnPriorSamples int
}

// DefaultSessionAwareConfig returns the production-oriented default policy.
func DefaultSessionAwareConfig() *SessionAwareConfig {
	return &SessionAwareConfig{
		BaseMethod:                   MethodHybrid,
		IdleTimeoutSeconds:           300,
		MinTurnsBeforeSwitch:         1,
		SwitchMargin:                 0.05,
		StayBias:                     0.10,
		ToolLoopHardLock:             true,
		ContextPortabilityHardLock:   true,
		DecisionDriftReset:           true,
		ToolLoopStayBias:             0.35,
		PrefixCacheWeight:            0.20,
		HandoffPenaltyWeight:         1.0,
		DefaultHandoffPenalty:        0.05,
		QualityGapMultiplier:         1.0,
		MaxCacheCostMultiplier:       2.5,
		SwitchHistoryWeight:          0.04,
		RemainingTurnPriorWeight:     1.0,
		RemainingTurnPriorHorizon:    8,
		MinRemainingTurnPriorSamples: 3,
	}
}

// SessionAwareSelector wraps a base selector with explicit session policy.
type SessionAwareSelector struct {
	config       *SessionAwareConfig
	baseSelector Selector
	modelParams  map[string]config.ModelParams
	lookupTable  lookuptable.LookupTable
}

func NewSessionAwareSelector(cfg *SessionAwareConfig) *SessionAwareSelector {
	if cfg == nil {
		cfg = DefaultSessionAwareConfig()
	} else {
		cfg = normalizeSessionAwareConfig(*cfg)
	}
	return &SessionAwareSelector{
		config:      cfg,
		modelParams: make(map[string]config.ModelParams),
	}
}

func normalizeSessionAwareConfig(cfg SessionAwareConfig) *SessionAwareConfig {
	defaults := DefaultSessionAwareConfig()
	cfg.BaseMethod = defaultSessionAwareBaseMethod(cfg.BaseMethod, defaults.BaseMethod)
	cfg.IdleTimeoutSeconds = defaultNonNegativeInt(cfg.IdleTimeoutSeconds, defaults.IdleTimeoutSeconds)
	cfg.MinTurnsBeforeSwitch = defaultNonNegativeInt(cfg.MinTurnsBeforeSwitch, defaults.MinTurnsBeforeSwitch)
	cfg.SwitchMargin = defaultNonNegativeFloat(cfg.SwitchMargin, defaults.SwitchMargin)
	cfg.StayBias = defaultNonNegativeFloat(cfg.StayBias, defaults.StayBias)
	cfg.ToolLoopStayBias = defaultNonNegativeFloat(cfg.ToolLoopStayBias, defaults.ToolLoopStayBias)
	cfg.PrefixCacheWeight = defaultNonNegativeFloat(cfg.PrefixCacheWeight, defaults.PrefixCacheWeight)
	cfg.HandoffPenaltyWeight = defaultNonNegativeFloat(cfg.HandoffPenaltyWeight, defaults.HandoffPenaltyWeight)
	cfg.DefaultHandoffPenalty = defaultNonNegativeFloat(cfg.DefaultHandoffPenalty, defaults.DefaultHandoffPenalty)
	cfg.QualityGapMultiplier = defaultPositiveFloat(cfg.QualityGapMultiplier, defaults.QualityGapMultiplier)
	cfg.MaxCacheCostMultiplier = defaultAtLeastFloat(cfg.MaxCacheCostMultiplier, defaults.MaxCacheCostMultiplier, 1)
	cfg.SwitchHistoryWeight = defaultNonNegativeFloat(cfg.SwitchHistoryWeight, defaults.SwitchHistoryWeight)
	cfg.RemainingTurnPriorWeight = defaultNonNegativeFloat(cfg.RemainingTurnPriorWeight, defaults.RemainingTurnPriorWeight)
	cfg.RemainingTurnPriorHorizon = defaultPositiveInt(cfg.RemainingTurnPriorHorizon, defaults.RemainingTurnPriorHorizon)
	cfg.MinRemainingTurnPriorSamples = defaultNonNegativeInt(cfg.MinRemainingTurnPriorSamples, defaults.MinRemainingTurnPriorSamples)
	return &cfg
}

func defaultSessionAwareBaseMethod(method, defaultMethod SelectionMethod) SelectionMethod {
	if method == "" || method == MethodSessionAware {
		return defaultMethod
	}
	return method
}

func defaultNonNegativeInt(value, defaultValue int) int {
	if value < 0 {
		return defaultValue
	}
	return value
}

func defaultPositiveInt(value, defaultValue int) int {
	if value <= 0 {
		return defaultValue
	}
	return value
}

func defaultNonNegativeFloat(value, defaultValue float64) float64 {
	if value < 0 {
		return defaultValue
	}
	return value
}

func defaultPositiveFloat(value, defaultValue float64) float64 {
	if value <= 0 {
		return defaultValue
	}
	return value
}

func defaultAtLeastFloat(value, defaultValue, minimum float64) float64 {
	if value < minimum {
		return defaultValue
	}
	return value
}

func (s *SessionAwareSelector) Method() SelectionMethod {
	return MethodSessionAware
}

func (s *SessionAwareSelector) SetBaseSelector(selector Selector) {
	s.baseSelector = selector
}

func (s *SessionAwareSelector) SetLookupTable(lt lookuptable.LookupTable) {
	s.lookupTable = lt
}

func (s *SessionAwareSelector) InitializeFromConfig(modelConfig map[string]config.ModelParams) {
	s.modelParams = make(map[string]config.ModelParams, len(modelConfig))
	for k, v := range modelConfig {
		s.modelParams[k] = v
	}
}

func (s *SessionAwareSelector) Select(ctx context.Context, selCtx *SelectionContext) (*SelectionResult, error) {
	if err := ValidateSelectionContext(selCtx); err != nil {
		return nil, err
	}

	base, err := s.selectBase(ctx, selCtx)
	if err != nil {
		return nil, err
	}
	session := selCtx.AgenticSession
	if session == nil {
		trace := s.newPolicyTrace(selCtx, base, nil, "", false, false)
		trace.MissingSignals = append(trace.MissingSignals, "session_context")
		return s.wrapBaseSelection(base, "missing_session_context", trace), nil
	}

	current := strings.TrimSpace(session.PreviousModel)
	driftDetected := s.decisionDriftDetected(selCtx, session)
	trace := s.newPolicyTrace(selCtx, base, session, current, false, driftDetected)
	if signal, reason := currentModelIssue(selCtx.CandidateModels, current); signal != "" {
		trace.MissingSignals = append(trace.MissingSignals, signal)
		return s.wrapBaseSelection(base, reason, trace), nil
	}

	timeout := secondsDuration(s.config.IdleTimeoutSeconds)
	idleExpired := session.idleExpired(timeout)
	trace.IdleExpired = idleExpired
	continuityReset := idleExpired || driftDetected
	if reason := s.sessionHardLockReason(session, continuityReset); reason != "" {
		return s.forceCurrent(selCtx, base, current, reason, trace), nil
	}

	adjusted, candidateTraces := s.adjustScores(selCtx, base, session, current, continuityReset)
	bestRef, bestScore := bestCandidateByScore(selCtx.CandidateModels, adjusted)
	if bestRef == nil {
		trace.MissingSignals = append(trace.MissingSignals, "adjusted_scores")
		return s.wrapBaseSelection(base, "missing_adjusted_scores", trace), nil
	}

	confidence := adjustedConfidence(bestRef.Model, adjusted)
	reason := fmt.Sprintf(
		"session_aware: base=%s current=%s selected=%s idle_expired=%t active_tool_loop=%t",
		base.Method, current, bestRef.Model, idleExpired, session.ActiveToolLoop,
	)
	trace.SelectedModel = bestRef.Model
	trace.FinalScores = cloneScores(adjusted)
	trace.CandidateTraces = candidateTraces
	trace.DecisionReason = "switch_allowed"
	if bestRef.Model == current {
		trace.DecisionReason = "stay_has_best_adjusted_score"
	}
	logging.Infof("[SessionAwareSelector] %s score=%.4f confidence=%.2f", reason, bestScore, confidence)
	return &SelectionResult{
		SelectedModel: bestRef.Model,
		LoRAName:      bestRef.LoRAName,
		Score:         bestScore,
		Confidence:    confidence,
		Method:        MethodSessionAware,
		Tier:          TierSupported,
		Reasoning:     reason,
		AllScores:     adjusted,
		SessionPolicy: trace,
	}, nil
}

func currentModelIssue(candidates []config.ModelRef, current string) (signal string, reason string) {
	if current == "" {
		return "previous_model", "missing_previous_model"
	}
	if !candidateSetContains(candidates, current) {
		return "previous_model_not_in_candidates", "previous_model_not_in_candidates"
	}
	return "", ""
}

func (s *SessionAwareSelector) sessionHardLockReason(session *AgenticSessionContext, continuityReset bool) string {
	if session.ActiveToolLoop && s.config.ToolLoopHardLock {
		return "hard_lock=tool_loop"
	}
	if session.HasNonPortableContext && s.config.ContextPortabilityHardLock {
		return contextPortabilityHardLockReason(session)
	}
	if effectiveSessionTurns(session) < s.config.MinTurnsBeforeSwitch && !continuityReset {
		return "hard_lock=min_turns"
	}
	return ""
}

func contextPortabilityHardLockReason(session *AgenticSessionContext) string {
	reason := "hard_lock=context_portability"
	if session.NonPortableContextReason != "" {
		reason += ":" + session.NonPortableContextReason
	}
	return reason
}

func effectiveSessionTurns(session *AgenticSessionContext) int {
	if session == nil {
		return 0
	}
	if session.MemoryTurnCount > session.TurnIndex {
		return session.MemoryTurnCount
	}
	return session.TurnIndex
}

func (s *SessionAwareSelector) decisionDriftDetected(selCtx *SelectionContext, session *AgenticSessionContext) bool {
	if !s.config.DecisionDriftReset || selCtx == nil || session == nil {
		return false
	}
	if selCtx.DecisionName == "" || session.LastDecisionName == "" {
		return false
	}
	return selCtx.DecisionName != session.LastDecisionName
}

func (s *SessionAwareSelector) selectBase(ctx context.Context, selCtx *SelectionContext) (*SelectionResult, error) {
	if s.baseSelector != nil && s.baseSelector.Method() != MethodSessionAware {
		result, err := s.baseSelector.Select(ctx, selCtx)
		if err == nil && result != nil {
			ensureScoresForCandidates(result, selCtx.CandidateModels)
			return result, nil
		}
		logging.Warnf("[SessionAwareSelector] base selector %s failed: %v", s.baseSelector.Method(), err)
	}
	result := staticBaseResult(selCtx)
	return result, nil
}

func (s *SessionAwareSelector) wrapBaseSelection(base *SelectionResult, reason string, trace *SessionPolicyTrace) *SelectionResult {
	wrapped := *base
	wrapped.Method = MethodSessionAware
	wrapped.Tier = TierSupported
	wrapped.Reasoning = fmt.Sprintf("session_aware: %s; base=%s selected=%s", reason, base.Method, base.SelectedModel)
	if trace != nil {
		trace.SelectedModel = base.SelectedModel
		trace.DecisionReason = reason
		trace.FinalScores = cloneScores(base.AllScores)
		wrapped.SessionPolicy = trace
	}
	return &wrapped
}

func (s *SessionAwareSelector) forceCurrent(selCtx *SelectionContext, base *SelectionResult, current, reason string, trace *SessionPolicyTrace) *SelectionResult {
	allScores := cloneScores(base.AllScores)
	if allScores == nil {
		allScores = make(map[string]float64, len(selCtx.CandidateModels))
	}
	currentScore := allScores[current]
	allScores[current] = math.Max(currentScore, maxScore(allScores)+s.config.ToolLoopStayBias+s.config.StayBias)
	ref := modelRefForName(selCtx.CandidateModels, current)
	loRA := ""
	if ref != nil {
		loRA = ref.LoRAName
	}
	if trace != nil {
		trace.HardLocked = true
		trace.HardLockReason = reason
		trace.DecisionReason = reason
		trace.SelectedModel = current
		trace.FinalScores = cloneScores(allScores)
		baseScore := 0.0
		if base != nil && base.AllScores != nil {
			baseScore = base.AllScores[current]
		}
		trace.CandidateTraces = map[string]SessionCandidateTrace{
			current: {
				Current:    true,
				BaseScore:  baseScore,
				FinalScore: allScores[current],
			},
		}
	}
	result := &SelectionResult{
		SelectedModel: current,
		LoRAName:      loRA,
		Score:         allScores[current],
		Confidence:    1.0,
		Method:        MethodSessionAware,
		Tier:          TierSupported,
		Reasoning:     fmt.Sprintf("session_aware: %s current=%s base=%s", reason, current, base.Method),
		AllScores:     allScores,
		SessionPolicy: trace,
	}
	return result
}

func (s *SessionAwareSelector) newPolicyTrace(
	selCtx *SelectionContext,
	base *SelectionResult,
	session *AgenticSessionContext,
	current string,
	idleExpired bool,
	decisionDrift bool,
) *SessionPolicyTrace {
	trace := &SessionPolicyTrace{
		Algorithm:     "agentic_continuity_routing",
		CurrentModel:  current,
		SwitchMargin:  s.config.SwitchMargin,
		StayBias:      s.config.StayBias,
		IdleExpired:   idleExpired,
		DecisionDrift: decisionDrift,
	}
	if base != nil {
		trace.BaseMethod = string(base.Method)
		trace.BaseSelectedModel = base.SelectedModel
		trace.BaseScores = cloneScores(base.AllScores)
	}
	if session == nil {
		return trace
	}
	trace.SessionID = session.ID
	trace.UserID = session.UserID
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
	trace.HasNonPortableContext = session.HasNonPortableContext
	trace.NonPortableContextReason = session.NonPortableContextReason
	trace.IdleKnown = session.IdleKnown
	trace.IdleForSeconds = session.IdleFor.Seconds()
	continuation := s.continuationEvidence(selCtx, session)
	trace.ContinuationMass = continuation.Mass
	trace.RemainingTurnPrior = continuation.RemainingTurnPrior
	trace.RemainingTurnPriorOK = continuation.RemainingTurnPriorOK
	trace.RemainingTurnsEstimate = continuation.RemainingTurnsEstimate
	trace.RemainingTurnPriorSource = continuation.RemainingTurnPriorSource
	trace.RemainingTurnPriorSampleCount = continuation.RemainingTurnPriorSampleCount
	trace.RemainingTurnPriorRejected = continuation.RemainingTurnPriorRejected
	trace.CacheWarmth = sessionCacheWarmth(session, idleExpired)
	trace.CacheWarmthOK = session.CacheWarmthOK
	if session.MemoryPresent && trace.MemoryTurnCount == 0 {
		trace.MemoryTurnCount = session.TurnIndex + 1
	}
	if selCtx != nil && trace.SessionID == "" {
		trace.SessionID = selCtx.SessionID
	}
	return trace
}

func (s *SessionAwareSelector) UpdateFeedback(ctx context.Context, feedback *Feedback) error {
	if s.baseSelector == nil {
		return nil
	}
	return s.baseSelector.UpdateFeedback(ctx, feedback)
}

func (s *SessionAwareSelector) Tier() AlgorithmTier {
	return TierSupported
}

func (s *SessionAwareSelector) ExternalDependencies() []Dependency {
	if s.baseSelector == nil {
		return nil
	}
	return s.baseSelector.ExternalDependencies()
}
