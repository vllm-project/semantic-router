package store

// LearningDiagnostics stores router-learning outputs in a typed replay-facing
// shape. Compact response headers carry only method/action/scope/reason; route
// explanation details live here.
type LearningDiagnostics struct {
	ProtectionPreflight *LearningProtectionDiagnostics `json:"protection_preflight,omitempty"`
	Adaptation          *LearningAdaptationDiagnostics `json:"adaptation,omitempty"`
	Protection          *LearningProtectionDiagnostics `json:"protection,omitempty"`
}

// LearningPolicyDiagnostics captures fields common to every learning method.
type LearningPolicyDiagnostics struct {
	Learning string `json:"learning,omitempty"`
	Method   string `json:"method,omitempty"`
	Mode     string `json:"mode,omitempty"`
	Scope    string `json:"scope,omitempty"`
	Action   string `json:"action,omitempty"`
	Reason   string `json:"reason,omitempty"`
}

// LearningAdaptationDiagnostics explains an online adaptation proposal.
type LearningAdaptationDiagnostics struct {
	LearningPolicyDiagnostics

	CandidateSet  string                            `json:"candidate_set,omitempty"`
	Strategy      string                            `json:"strategy,omitempty"`
	BaseModel     string                            `json:"base_model,omitempty"`
	ProposalModel string                            `json:"proposal_model,omitempty"`
	Decision      string                            `json:"decision,omitempty"`
	DecisionTier  int                               `json:"decision_tier,omitempty"`
	Sampling      *LearningSamplingDiagnostics      `json:"sampling,omitempty"`
	Scores        map[string]LearningCandidateScore `json:"scores,omitempty"`
}

// LearningSamplingDiagnostics records whether routing_sampling used a random
// posterior draw or deterministic posterior means.
type LearningSamplingDiagnostics struct {
	Used bool  `json:"used"`
	Seed int64 `json:"seed,omitempty"`
}

// LearningCandidateScore records the score decomposition for one candidate.
type LearningCandidateScore struct {
	Score              float64 `json:"score,omitempty"`
	PosteriorMean      float64 `json:"posterior_mean,omitempty"`
	PredictedQuality   float64 `json:"predicted_quality,omitempty"`
	CostPenalty        float64 `json:"cost_penalty,omitempty"`
	OverusePenalty     float64 `json:"overuse_penalty,omitempty"`
	ReliabilityPenalty float64 `json:"reliability_penalty,omitempty"`
	LatencyAdjustment  float64 `json:"latency_adjustment,omitempty"`
	CacheAdjustment    float64 `json:"cache_adjustment,omitempty"`
}

// LearningProtectionDiagnostics explains protection preflight and switch guard
// decisions.
type LearningProtectionDiagnostics struct {
	LearningPolicyDiagnostics

	Identity        *LearningIdentityDiagnostics `json:"identity,omitempty"`
	Sampling        string                       `json:"sampling,omitempty"`
	BaseModel       string                       `json:"base_model,omitempty"`
	ProposalModel   string                       `json:"proposal_model,omitempty"`
	FinalModel      string                       `json:"final_model,omitempty"`
	SwitchCost      float64                      `json:"switch_cost,omitempty"`
	SwitchMargin    float64                      `json:"switch_margin,omitempty"`
	StabilityWeight float64                      `json:"stability_weight,omitempty"`
	Rescue          *LearningRescueDiagnostics   `json:"rescue,omitempty"`

	Algorithm                     string                            `json:"algorithm,omitempty"`
	BaseMethod                    string                            `json:"base_method,omitempty"`
	Phase                         string                            `json:"phase,omitempty"`
	CurrentModel                  string                            `json:"current_model,omitempty"`
	BaseSelectedModel             string                            `json:"base_selected_model,omitempty"`
	SelectedModel                 string                            `json:"selected_model,omitempty"`
	TurnIndex                     int                               `json:"turn_index,omitempty"`
	MemoryTurnCount               int                               `json:"memory_turn_count,omitempty"`
	SwitchCount                   int                               `json:"switch_count,omitempty"`
	LastDecisionName              string                            `json:"last_decision_name,omitempty"`
	MemoryPromptTokens            int64                             `json:"memory_prompt_tokens,omitempty"`
	MemoryCachedTokens            int64                             `json:"memory_cached_tokens,omitempty"`
	MemoryEstimatedCachedTokens   int64                             `json:"memory_estimated_cached_tokens,omitempty"`
	MemoryEstimatedCacheSavings   float64                           `json:"memory_estimated_cache_savings,omitempty"`
	LastCacheAccountingSource     string                            `json:"last_cache_accounting_source,omitempty"`
	ActiveToolLoop                bool                              `json:"active_tool_loop,omitempty"`
	IdleKnown                     bool                              `json:"idle_known,omitempty"`
	IdleForSeconds                float64                           `json:"idle_for_seconds,omitempty"`
	IdleExpired                   bool                              `json:"idle_expired,omitempty"`
	HasNonPortableContext         bool                              `json:"has_non_portable_context,omitempty"`
	NonPortableContextReason      string                            `json:"non_portable_context_reason,omitempty"`
	DecisionDrift                 bool                              `json:"decision_drift,omitempty"`
	HardLocked                    bool                              `json:"hard_locked,omitempty"`
	HardLockReason                string                            `json:"hard_lock_reason,omitempty"`
	DecisionReason                string                            `json:"decision_reason,omitempty"`
	MissingSignals                []string                          `json:"missing_signals,omitempty"`
	ContinuationMass              float64                           `json:"continuation_mass,omitempty"`
	RemainingTurnPrior            float64                           `json:"remaining_turn_prior,omitempty"`
	RemainingTurnPriorOK          bool                              `json:"remaining_turn_prior_ok,omitempty"`
	RemainingTurnsEstimate        float64                           `json:"remaining_turns_estimate,omitempty"`
	RemainingTurnPriorSource      string                            `json:"remaining_turn_prior_source,omitempty"`
	RemainingTurnPriorSampleCount int                               `json:"remaining_turn_prior_sample_count,omitempty"`
	RemainingTurnPriorRejected    string                            `json:"remaining_turn_prior_rejected,omitempty"`
	CacheWarmth                   float64                           `json:"cache_warmth,omitempty"`
	CacheWarmthOK                 bool                              `json:"cache_warmth_ok,omitempty"`
	StayBias                      float64                           `json:"stay_bias,omitempty"`
	BaseScores                    map[string]float64                `json:"base_scores,omitempty"`
	FinalScores                   map[string]float64                `json:"final_scores,omitempty"`
	CandidateTraces               map[string]LearningCandidateTrace `json:"candidate_traces,omitempty"`
}

// LearningIdentityDiagnostics records bounded, hashed identity evidence.
type LearningIdentityDiagnostics struct {
	Scope         string                  `json:"scope,omitempty"`
	Headers       LearningIdentityHeaders `json:"headers,omitempty"`
	Session       LearningIdentityPart    `json:"session,omitempty"`
	Conversation  LearningIdentityPart    `json:"conversation,omitempty"`
	MemoryKeyHash string                  `json:"memory_key_hash,omitempty"`
}

type LearningIdentityHeaders struct {
	Session      string `json:"session,omitempty"`
	Conversation string `json:"conversation,omitempty"`
}

type LearningIdentityPart struct {
	Source   string `json:"source,omitempty"`
	Required bool   `json:"required,omitempty"`
	Status   string `json:"status,omitempty"`
	Hash     string `json:"hash,omitempty"`
}

type LearningRescueDiagnostics struct {
	Active bool `json:"active"`
}

// LearningCandidateTrace captures the protection score decomposition for one
// candidate model.
type LearningCandidateTrace struct {
	Current                bool    `json:"current,omitempty"`
	BaseScore              float64 `json:"base_score,omitempty"`
	FinalScore             float64 `json:"final_score,omitempty"`
	SelectorDelta          float64 `json:"selector_delta,omitempty"`
	QualityGap             float64 `json:"quality_gap,omitempty"`
	HandoffPenalty         float64 `json:"handoff_penalty,omitempty"`
	PrefixCacheBenefit     float64 `json:"prefix_cache_benefit,omitempty"`
	PrefixCachePenalty     float64 `json:"prefix_cache_penalty,omitempty"`
	ToolLoopPenalty        float64 `json:"tool_loop_penalty,omitempty"`
	SwitchHistoryPenalty   float64 `json:"switch_history_penalty,omitempty"`
	FrontierCostMultiplier float64 `json:"frontier_cost_multiplier,omitempty"`
	NetSwitchAdvantage     float64 `json:"net_switch_advantage,omitempty"`
}
