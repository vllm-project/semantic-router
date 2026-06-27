package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func (p routerLearningPolicy) replayCommon() routerreplay.LearningPolicyDiagnostics {
	common := routerreplay.LearningPolicyDiagnostics{}
	if !p.Empty() {
		common.Learning = routerLearningPolicyName
	}
	common.Method = string(p.Method)
	common.Mode = strings.TrimSpace(p.Mode)
	common.Scope = strings.TrimSpace(p.Scope)
	common.Action = string(p.Action)
	common.Reason = strings.TrimSpace(p.Reason)
	return common
}

func (p routerLearningPolicy) toReplayAdaptation() *routerreplay.LearningAdaptationDiagnostics {
	if p.Empty() {
		return nil
	}
	out := &routerreplay.LearningAdaptationDiagnostics{
		LearningPolicyDiagnostics: p.replayCommon(),
	}
	diag := p.Details.Adaptation
	if diag == nil {
		return out
	}
	out.CandidateSet = strings.TrimSpace(diag.candidateSet)
	out.Strategy = strings.TrimSpace(diag.strategy)
	out.BaseModel = strings.TrimSpace(diag.baseModel)
	out.ProposalModel = strings.TrimSpace(diag.proposalModel)
	out.Decision = strings.TrimSpace(diag.decision)
	out.DecisionTier = diag.decisionTier
	out.Sampling = &routerreplay.LearningSamplingDiagnostics{
		Used: diag.sampling.used,
		Seed: diag.sampling.seed,
	}
	out.Scores = replayCandidateScores(diag.scores)
	return out
}

func replayCandidateScores(scores []routerLearningCandidateScore) map[string]routerreplay.LearningCandidateScore {
	if len(scores) == 0 {
		return nil
	}
	out := make(map[string]routerreplay.LearningCandidateScore, len(scores))
	for _, score := range scores {
		model := strings.TrimSpace(score.model)
		if model == "" {
			continue
		}
		out[model] = routerreplay.LearningCandidateScore{
			Score:              roundLearningFloat(score.score),
			PosteriorMean:      roundLearningFloat(score.posteriorMean),
			PredictedQuality:   roundLearningFloat(score.predictedQuality),
			CostPenalty:        roundLearningFloat(score.costPenalty),
			OverusePenalty:     roundLearningFloat(score.overusePenalty),
			ReliabilityPenalty: roundLearningFloat(score.reliabilityPenalty),
			LatencyAdjustment:  roundLearningFloat(score.latencyAdjustment),
			CacheAdjustment:    roundLearningFloat(score.cacheAdjustment),
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func (p routerLearningPolicy) toReplayProtection() *routerreplay.LearningProtectionDiagnostics {
	if p.Empty() {
		return nil
	}
	out := &routerreplay.LearningProtectionDiagnostics{
		LearningPolicyDiagnostics: p.replayCommon(),
	}
	diag := p.Details.Protection
	if diag == nil {
		return out
	}
	out.Identity = diag.identity.toReplay()
	out.Sampling = strings.TrimSpace(diag.samplingPolicy)
	out.BaseModel = strings.TrimSpace(diag.baseModel)
	out.ProposalModel = strings.TrimSpace(diag.proposalModel)
	out.FinalModel = strings.TrimSpace(diag.finalModel)
	out.SwitchCost = diag.switchCost
	out.SwitchMargin = diag.switchMargin
	out.StabilityWeight = diag.stabilityWeight
	if diag.rescue {
		out.Rescue = &routerreplay.LearningRescueDiagnostics{Active: true}
	}
	applyReplayProtectionTrace(out, diag.trace)
	if out.SwitchMargin == 0 && diag.trace != nil {
		out.SwitchMargin = diag.trace.SwitchMargin
	}
	return out
}

func (d routerLearningIdentityDiagnostics) toReplay() *routerreplay.LearningIdentityDiagnostics {
	if d.scope == "" && d.sessionHeader == "" && d.convoHeader == "" {
		return nil
	}
	return &routerreplay.LearningIdentityDiagnostics{
		Scope: strings.TrimSpace(d.scope),
		Headers: routerreplay.LearningIdentityHeaders{
			Session:      strings.TrimSpace(d.sessionHeader),
			Conversation: strings.TrimSpace(d.convoHeader),
		},
		Session:       d.session.toReplay(),
		Conversation:  d.conversation.toReplay(),
		MemoryKeyHash: strings.TrimSpace(d.memoryKeyHash),
	}
}

func (p routerLearningIdentityPart) toReplay() routerreplay.LearningIdentityPart {
	return routerreplay.LearningIdentityPart{
		Source:   strings.TrimSpace(p.source),
		Required: p.required,
		Status:   string(p.status),
		Hash:     strings.TrimSpace(p.hash),
	}
}

func applyReplayProtectionTrace(out *routerreplay.LearningProtectionDiagnostics, trace *selection.SessionPolicyTrace) {
	if out == nil || trace == nil {
		return
	}
	applyReplayProtectionBase(out, trace)
	applyReplayProtectionMemory(out, trace)
	applyReplayProtectionDecision(out, trace)
	applyReplayProtectionScores(out, trace)
}

func applyReplayProtectionBase(out *routerreplay.LearningProtectionDiagnostics, trace *selection.SessionPolicyTrace) {
	out.Algorithm = string(routerLearningMethodProtection)
	out.BaseMethod = strings.TrimSpace(trace.BaseMethod)
	out.Phase = strings.TrimSpace(string(trace.Phase))
	out.CurrentModel = strings.TrimSpace(trace.CurrentModel)
	out.BaseSelectedModel = strings.TrimSpace(trace.BaseSelectedModel)
	out.SelectedModel = strings.TrimSpace(trace.SelectedModel)
	out.TurnIndex = trace.TurnIndex
	out.SwitchCount = trace.SwitchCount
	out.LastDecisionName = strings.TrimSpace(trace.LastDecisionName)
}

func applyReplayProtectionMemory(out *routerreplay.LearningProtectionDiagnostics, trace *selection.SessionPolicyTrace) {
	out.MemoryTurnCount = trace.MemoryTurnCount
	out.MemoryPromptTokens = trace.MemoryPromptTokens
	out.MemoryCachedTokens = trace.MemoryCachedTokens
	out.MemoryEstimatedCachedTokens = trace.MemoryEstimatedCachedTokens
	out.MemoryEstimatedCacheSavings = trace.MemoryEstimatedCacheSavings
	out.LastCacheAccountingSource = strings.TrimSpace(trace.LastCacheAccountingSource)
	out.CacheWarmth = trace.CacheWarmth
	out.CacheWarmthOK = trace.CacheWarmthOK
}

func applyReplayProtectionDecision(out *routerreplay.LearningProtectionDiagnostics, trace *selection.SessionPolicyTrace) {
	out.ActiveToolLoop = trace.ActiveToolLoop
	out.IdleKnown = trace.IdleKnown
	out.IdleForSeconds = trace.IdleForSeconds
	out.IdleExpired = trace.IdleExpired
	out.HasNonPortableContext = trace.HasNonPortableContext
	out.NonPortableContextReason = strings.TrimSpace(trace.NonPortableContextReason)
	out.DecisionDrift = trace.DecisionDrift
	out.HardLocked = trace.HardLocked
	out.HardLockReason = strings.TrimSpace(trace.HardLockReason)
	out.DecisionReason = strings.TrimSpace(trace.DecisionReason)
	out.MissingSignals = cloneLearningStringSlice(trace.MissingSignals)
	out.ContinuationMass = trace.ContinuationMass
	out.RemainingTurnPrior = trace.RemainingTurnPrior
	out.RemainingTurnPriorOK = trace.RemainingTurnPriorOK
	out.RemainingTurnsEstimate = trace.RemainingTurnsEstimate
	out.RemainingTurnPriorSource = strings.TrimSpace(trace.RemainingTurnPriorSource)
	out.RemainingTurnPriorSampleCount = trace.RemainingTurnPriorSampleCount
	out.RemainingTurnPriorRejected = strings.TrimSpace(trace.RemainingTurnPriorRejected)
	out.StayBias = trace.StayBias
}

func applyReplayProtectionScores(out *routerreplay.LearningProtectionDiagnostics, trace *selection.SessionPolicyTrace) {
	out.BaseScores = cloneLearningScores(trace.BaseScores)
	out.FinalScores = cloneLearningScores(trace.FinalScores)
	out.CandidateTraces = replayCandidateTraces(trace.CandidateTraces)
}

func cloneLearningStringSlice(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	return append([]string(nil), values...)
}

func cloneLearningScores(scores map[string]float64) map[string]float64 {
	if len(scores) == 0 {
		return nil
	}
	out := make(map[string]float64, len(scores))
	for model, score := range scores {
		if strings.TrimSpace(model) == "" {
			continue
		}
		out[model] = score
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func replayCandidateTraces(traces map[string]selection.SessionCandidateTrace) map[string]routerreplay.LearningCandidateTrace {
	if len(traces) == 0 {
		return nil
	}
	out := make(map[string]routerreplay.LearningCandidateTrace, len(traces))
	for model, trace := range traces {
		if strings.TrimSpace(model) == "" {
			continue
		}
		out[model] = routerreplay.LearningCandidateTrace{
			Current:                trace.Current,
			BaseScore:              trace.BaseScore,
			FinalScore:             trace.FinalScore,
			SelectorDelta:          trace.SelectorDelta,
			QualityGap:             trace.QualityGap,
			HandoffPenalty:         trace.HandoffPenalty,
			PrefixCacheBenefit:     trace.PrefixCacheBenefit,
			PrefixCachePenalty:     trace.PrefixCachePenalty,
			ToolLoopPenalty:        trace.ToolLoopPenalty,
			SwitchHistoryPenalty:   trace.SwitchHistoryPenalty,
			FrontierCostMultiplier: trace.FrontierCostMultiplier,
			NetSwitchAdvantage:     trace.NetSwitchAdvantage,
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
