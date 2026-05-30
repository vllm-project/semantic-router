package selection

import "math"

func (s *SessionAwareSelector) adjustScores(
	selCtx *SelectionContext,
	base *SelectionResult,
	session *AgenticSessionContext,
	current string,
	idleExpired bool,
) (map[string]float64, map[string]SessionCandidateTrace) {
	baseScores := cloneScores(base.AllScores)
	ensureScoresForCandidates(&SelectionResult{AllScores: baseScores, SelectedModel: base.SelectedModel, Score: base.Score}, selCtx.CandidateModels)
	currentBaseScore := baseScores[current]
	currentAdjustedScore, currentTrace := s.scoreCurrentCandidate(
		currentBaseScore,
		session,
		current,
		idleExpired,
		SessionCandidateTrace{
			Current:    true,
			BaseScore:  currentBaseScore,
			FinalScore: currentBaseScore,
		},
	)

	adjusted := make(map[string]float64, len(selCtx.CandidateModels))
	traces := make(map[string]SessionCandidateTrace, len(selCtx.CandidateModels))
	for _, candidate := range selCtx.CandidateModels {
		model := candidate.Model
		score := baseScores[model]
		trace := SessionCandidateTrace{
			Current:    model == current,
			BaseScore:  score,
			FinalScore: score,
		}
		if model == current {
			adjusted[model] = currentAdjustedScore
			traces[model] = currentTrace
			continue
		}

		score, trace = s.scoreSwitchCandidate(selCtx, session, current, model, score, currentBaseScore, currentAdjustedScore, idleExpired, trace)
		adjusted[model] = score
		traces[model] = trace
	}
	return adjusted, traces
}

func (s *SessionAwareSelector) scoreCurrentCandidate(
	score float64,
	session *AgenticSessionContext,
	current string,
	idleExpired bool,
	trace SessionCandidateTrace,
) (float64, SessionCandidateTrace) {
	prefixBenefit := 0.0
	if !idleExpired {
		score += s.config.StayBias
		prefixBenefit = s.prefixCacheBenefit(session, current)
		score += prefixBenefit
	}
	if session.ActiveToolLoop {
		score += s.config.ToolLoopStayBias
	}
	trace.PrefixCacheBenefit = prefixBenefit
	trace.FinalScore = score
	return score, trace
}

func (s *SessionAwareSelector) scoreSwitchCandidate(
	selCtx *SelectionContext,
	session *AgenticSessionContext,
	current string,
	model string,
	score float64,
	currentBaseScore float64,
	currentAdjustedScore float64,
	idleExpired bool,
	trace SessionCandidateTrace,
) (float64, SessionCandidateTrace) {
	qualityGap := s.lookupQualityGap(selCtx, current, model)
	handoffPenalty := s.lookupHandoffPenalty(current, model)
	prefixPenalty := s.prefixCachePenalty(session, current, model, idleExpired)
	frontierMultiplier := s.cacheCostMultiplier(current, model)
	toolPenalty := 0.0
	if session.ActiveToolLoop {
		toolPenalty = s.config.ToolLoopStayBias
	}
	switchHistoryPenalty := s.switchHistoryPenalty(session)
	switchMargin := s.config.SwitchMargin
	if idleExpired {
		handoffPenalty = 0
		prefixPenalty = 0
		switchHistoryPenalty = 0
		switchMargin = 0
	}
	selectorDelta := score - currentBaseScore
	score += s.config.QualityGapMultiplier*qualityGap -
		s.config.HandoffPenaltyWeight*handoffPenalty -
		prefixPenalty -
		toolPenalty -
		switchHistoryPenalty -
		switchMargin
	if selectorDelta <= 0 && qualityGap <= 0 && !idleExpired {
		score -= s.config.StayBias
	}

	trace.SelectorDelta = selectorDelta
	trace.QualityGap = qualityGap
	trace.HandoffPenalty = handoffPenalty
	trace.PrefixCachePenalty = prefixPenalty
	trace.ToolLoopPenalty = toolPenalty
	trace.SwitchHistoryPenalty = switchHistoryPenalty
	trace.FrontierCostMultiplier = frontierMultiplier
	trace.NetSwitchAdvantage = score - currentAdjustedScore
	trace.FinalScore = score
	return score, trace
}

func (s *SessionAwareSelector) prefixCacheBenefit(session *AgenticSessionContext, current string) float64 {
	return s.config.PrefixCacheWeight * sessionContinuationMass(session) * sessionCacheWarmth(session, false) * s.cacheCostMultiplier(current, current)
}

func (s *SessionAwareSelector) prefixCachePenalty(
	session *AgenticSessionContext,
	current string,
	candidate string,
	idleExpired bool,
) float64 {
	if session == nil || current == "" || candidate == "" || current == candidate {
		return 0
	}
	return s.config.PrefixCacheWeight *
		sessionContinuationMass(session) *
		sessionCacheWarmth(session, idleExpired) *
		s.cacheCostMultiplier(current, candidate)
}

func (s *SessionAwareSelector) switchHistoryPenalty(session *AgenticSessionContext) float64 {
	if session == nil || session.MemorySwitchCount <= 0 || s.config.SwitchHistoryWeight <= 0 {
		return 0
	}
	return s.config.SwitchHistoryWeight * clampF(float64(session.MemorySwitchCount)/8.0, 0, 1)
}

func (s *SessionAwareSelector) cacheCostMultiplier(current, candidate string) float64 {
	maxPressure := math.Max(s.modelCostPressure(current), s.modelCostPressure(candidate))
	return 1.0 + (s.config.MaxCacheCostMultiplier-1.0)*maxPressure
}

func (s *SessionAwareSelector) modelCostPressure(model string) float64 {
	if model == "" || len(s.modelParams) == 0 {
		return 0.5
	}
	maxCost := 0.0
	modelCost := 0.0
	for name, params := range s.modelParams {
		cost := params.Pricing.PromptPer1M + params.Pricing.CompletionPer1M
		if cost > maxCost {
			maxCost = cost
		}
		if name == model {
			modelCost = cost
		}
	}
	if maxCost > 0 && modelCost > 0 {
		return clamp01(modelCost / maxCost)
	}
	if params, ok := s.modelParams[model]; ok && params.QualityScore > 0 {
		return clamp01(params.QualityScore)
	}
	return 0.5
}

func (s *SessionAwareSelector) lookupQualityGap(selCtx *SelectionContext, current, candidate string) float64 {
	if s.lookupTable == nil || selCtx == nil {
		return 0
	}
	for _, family := range []string{selCtx.CategoryName, selCtx.DecisionName} {
		if family == "" {
			continue
		}
		if gap, ok := s.lookupTable.QualityGap(family, current, candidate); ok {
			return gap
		}
	}
	return 0
}

func (s *SessionAwareSelector) lookupHandoffPenalty(current, candidate string) float64 {
	if s.lookupTable != nil {
		if penalty, ok := s.lookupTable.HandoffPenalty(current, candidate); ok {
			return penalty
		}
	}
	return s.config.DefaultHandoffPenalty
}

func sessionContinuationMass(session *AgenticSessionContext) float64 {
	if session == nil {
		return 0
	}
	contextTokens := math.Max(float64(session.ContextTokens), 1)
	reuseRatio := clampF(float64(session.HistoryTokens)/contextTokens, 0, 1)
	historyMass := clampF(float64(session.HistoryTokens)/8192.0, 0, 1)
	turnDepth := clampF(float64(effectiveSessionTurns(session))/8.0, 0, 1)
	if session.PreviousResponseID != "" && reuseRatio < 0.15 {
		reuseRatio = 0.15
	}
	return clampF(0.45*reuseRatio+0.35*historyMass+0.20*turnDepth, 0, 1)
}

func sessionCacheWarmth(session *AgenticSessionContext, idleExpired bool) float64 {
	warmth := 0.5
	if session != nil && session.CacheWarmthOK {
		warmth = clamp01(session.CacheWarmth)
	}
	if idleExpired {
		warmth *= 0.2
	}
	return warmth
}
