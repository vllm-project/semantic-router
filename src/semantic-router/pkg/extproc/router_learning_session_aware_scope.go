package extproc

import (
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

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
