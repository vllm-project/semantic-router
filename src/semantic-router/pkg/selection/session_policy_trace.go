package selection

// SessionPolicyTrace is the auditable output of the session-aware policy.
// It records the facts and costs used to decide whether an agentic session may
// switch models, so replay and experiments can reconstruct every stay/switch.
type SessionPolicyTrace struct {
	Algorithm  string
	BaseMethod string

	SessionID string
	UserID    string
	Phase     AgenticPhase

	CurrentModel      string
	BaseSelectedModel string
	SelectedModel     string

	TurnIndex                   int
	MemoryTurnCount             int
	SwitchCount                 int
	LastDecisionName            string
	MemoryPromptTokens          int64
	MemoryCachedTokens          int64
	MemoryEstimatedCachedTokens int64
	MemoryEstimatedCacheSavings float64
	LastCacheAccountingSource   string

	ActiveToolLoop bool
	IdleKnown      bool
	IdleForSeconds float64
	IdleExpired    bool

	HasNonPortableContext    bool
	NonPortableContextReason string
	DecisionDrift            bool

	HardLocked     bool
	HardLockReason string
	DecisionReason string
	MissingSignals []string

	ContinuationMass              float64
	RemainingTurnPrior            float64
	RemainingTurnPriorOK          bool
	RemainingTurnsEstimate        float64
	RemainingTurnPriorSource      string
	RemainingTurnPriorSampleCount int
	RemainingTurnPriorRejected    string
	CacheWarmth                   float64
	CacheWarmthOK                 bool
	SwitchMargin                  float64
	StayBias                      float64

	BaseScores      map[string]float64
	FinalScores     map[string]float64
	CandidateTraces map[string]SessionCandidateTrace
}

// SessionCandidateTrace captures the per-model decomposition used by
// SessionPolicyTrace.
type SessionCandidateTrace struct {
	Current bool

	BaseScore  float64
	FinalScore float64

	SelectorDelta          float64
	QualityGap             float64
	HandoffPenalty         float64
	PrefixCacheBenefit     float64
	PrefixCachePenalty     float64
	ToolLoopPenalty        float64
	SwitchHistoryPenalty   float64
	FrontierCostMultiplier float64
	NetSwitchAdvantage     float64
}

// ToMap converts the trace into a JSON-friendly representation for packages
// that should not import selection types into their storage schema.
func (t *SessionPolicyTrace) ToMap() map[string]interface{} {
	if t == nil {
		return nil
	}
	out := map[string]interface{}{
		"algorithm":                         t.Algorithm,
		"base_method":                       t.BaseMethod,
		"session_id":                        t.SessionID,
		"user_id":                           t.UserID,
		"phase":                             string(t.Phase),
		"current_model":                     t.CurrentModel,
		"base_selected_model":               t.BaseSelectedModel,
		"selected_model":                    t.SelectedModel,
		"turn_index":                        t.TurnIndex,
		"memory_turn_count":                 t.MemoryTurnCount,
		"switch_count":                      t.SwitchCount,
		"last_decision_name":                t.LastDecisionName,
		"memory_prompt_tokens":              t.MemoryPromptTokens,
		"memory_cached_tokens":              t.MemoryCachedTokens,
		"memory_estimated_cached_tokens":    t.MemoryEstimatedCachedTokens,
		"memory_estimated_cache_savings":    t.MemoryEstimatedCacheSavings,
		"last_cache_accounting_source":      t.LastCacheAccountingSource,
		"active_tool_loop":                  t.ActiveToolLoop,
		"idle_known":                        t.IdleKnown,
		"idle_for_seconds":                  t.IdleForSeconds,
		"idle_expired":                      t.IdleExpired,
		"has_non_portable_context":          t.HasNonPortableContext,
		"non_portable_context_reason":       t.NonPortableContextReason,
		"decision_drift":                    t.DecisionDrift,
		"hard_locked":                       t.HardLocked,
		"hard_lock_reason":                  t.HardLockReason,
		"decision_reason":                   t.DecisionReason,
		"missing_signals":                   append([]string(nil), t.MissingSignals...),
		"continuation_mass":                 t.ContinuationMass,
		"remaining_turn_prior":              t.RemainingTurnPrior,
		"remaining_turn_prior_ok":           t.RemainingTurnPriorOK,
		"remaining_turns_estimate":          t.RemainingTurnsEstimate,
		"remaining_turn_prior_source":       t.RemainingTurnPriorSource,
		"remaining_turn_prior_sample_count": t.RemainingTurnPriorSampleCount,
		"remaining_turn_prior_rejected":     t.RemainingTurnPriorRejected,
		"cache_warmth":                      t.CacheWarmth,
		"cache_warmth_ok":                   t.CacheWarmthOK,
		"switch_margin":                     t.SwitchMargin,
		"stay_bias":                         t.StayBias,
		"base_scores":                       cloneScores(t.BaseScores),
		"final_scores":                      cloneScores(t.FinalScores),
	}
	if len(t.CandidateTraces) > 0 {
		candidates := make(map[string]interface{}, len(t.CandidateTraces))
		for model, trace := range t.CandidateTraces {
			candidates[model] = map[string]interface{}{
				"current":                  trace.Current,
				"base_score":               trace.BaseScore,
				"final_score":              trace.FinalScore,
				"selector_delta":           trace.SelectorDelta,
				"quality_gap":              trace.QualityGap,
				"handoff_penalty":          trace.HandoffPenalty,
				"prefix_cache_benefit":     trace.PrefixCacheBenefit,
				"prefix_cache_penalty":     trace.PrefixCachePenalty,
				"tool_loop_penalty":        trace.ToolLoopPenalty,
				"switch_history_penalty":   trace.SwitchHistoryPenalty,
				"frontier_cost_multiplier": trace.FrontierCostMultiplier,
				"net_switch_advantage":     trace.NetSwitchAdvantage,
			}
		}
		out["candidate_traces"] = candidates
	}
	return out
}
