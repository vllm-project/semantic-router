package config

// ModelSelectionConfig represents configuration for advanced model selection algorithms.
type ModelSelectionConfig struct {
	// Method specifies the selection algorithm to use.
	Method string `yaml:"method,omitempty"`

	// Enabled indicates if model selection is enabled.
	Enabled bool `yaml:"enabled,omitempty"`

	// Family-specific configuration blocks.
	Elo      EloSelectionConfig      `yaml:"-"`
	RouterDC RouterDCSelectionConfig `yaml:"router_dc,omitempty"`
	AutoMix  AutoMixSelectionConfig  `yaml:"automix,omitempty"`
	Hybrid   HybridSelectionConfig   `yaml:"hybrid,omitempty"`
	ML       MLSelectionConfig       `yaml:"ml,omitempty"`

	SessionAware SessionAwareSelectionConfig `yaml:"-"`
	Momentum     MomentumSelectionConfig     `yaml:"momentum,omitempty"`

	// ModelSwitchGate configures session-aware stay-vs-switch evaluation.
	ModelSwitchGate ModelSwitchGateConfig `yaml:"-"`

	// LookupTables configures persisted lookup tables for session-aware routing.
	LookupTables LookupTableConfig `yaml:"-"`
}

// LookupTableConfig configures session-routing lookup tables that replace
// hardcoded constants with data-driven values.
type LookupTableConfig struct {
	// Enabled activates lookup table resolution during model selection.
	Enabled bool `yaml:"enabled,omitempty"`

	// StoragePath is the path to the JSON file used for persistence.
	// When empty, an in-memory backend is used (data lost on restart).
	StoragePath string `yaml:"storage_path,omitempty"`

	// AutoSaveInterval is the interval at which dirty entries are flushed to
	// disk (e.g. "5m"). Requires StoragePath to be set.
	AutoSaveInterval string `yaml:"auto_save_interval,omitempty"`

	// PopulateFromReplay enables automatic derivation of lookup table entries
	// from router replay records. Requires a replay store to be configured.
	PopulateFromReplay bool `yaml:"populate_from_replay,omitempty"`

	// PopulateInterval is how often to re-derive entries from the replay store
	// (e.g. "15m"). When empty, entries are only derived once at startup.
	// Requires PopulateFromReplay to be true.
	PopulateInterval string `yaml:"populate_interval,omitempty"`

	// QualityGaps are manual overrides for quality_gap entries.
	// Override values take precedence over replay-derived values.
	QualityGaps []QualityGapOverride `yaml:"quality_gaps,omitempty"`

	// HandoffPenalties are manual overrides for handoff_penalty entries.
	HandoffPenalties []HandoffPenaltyOverride `yaml:"handoff_penalties,omitempty"`

	// RemainingTurnPriors are manual overrides for remaining_turn_prior entries.
	RemainingTurnPriors []RemainingTurnPriorOverride `yaml:"remaining_turn_priors,omitempty"`
}

// QualityGapOverride manually sets a quality_gap entry.
type QualityGapOverride struct {
	TaskFamily     string  `yaml:"task_family"`
	CurrentModel   string  `yaml:"current_model"`
	CandidateModel string  `yaml:"candidate_model"`
	Value          float64 `yaml:"value"`
}

// HandoffPenaltyOverride manually sets a handoff_penalty entry.
type HandoffPenaltyOverride struct {
	FromModel string  `yaml:"from_model"`
	ToModel   string  `yaml:"to_model"`
	Value     float64 `yaml:"value"`
}

// RemainingTurnPriorOverride manually sets a remaining_turn_prior entry.
type RemainingTurnPriorOverride struct {
	IntentOrDomain string  `yaml:"intent_or_domain"`
	Value          float64 `yaml:"value"`
}

// MomentumSelectionConfig configures Conversational Routing Momentum (CRM).
type MomentumSelectionConfig struct {
	Enabled   bool    `yaml:"enabled,omitempty"`
	Attack    float64 `yaml:"attack,omitempty"`
	Release   float64 `yaml:"release,omitempty"`
	Threshold float64 `yaml:"threshold,omitempty"`
}

// ModelSwitchGateConfig configures auditable session-aware model switching.
//
// Note: enforce mode requires per-turn model history (the previous model the
// session used). Response API requests carry that signal via the conversation
// chain; Chat Completions carry it from the first follow-up turn via an
// in-memory, session-keyed last-model store (per-replica, lost on restart).
// When enforce is configured but the signal is still missing — the first turn of
// a session, an unresolvable session, or after a restart — the router emits a
// model_switch_gate_enforce_unavailable log event so operators can spot
// misconfiguration without an incident.
type ModelSwitchGateConfig struct {
	// Enabled activates stay-vs-switch evaluation after the configured selector runs.
	Enabled bool `yaml:"enabled,omitempty"`

	// Mode controls whether the gate only audits decisions ("shadow") or applies
	// them to keep the current model ("enforce"). Empty defaults to shadow.
	// Enforce only takes effect when previous-model history is available
	// (Response API via the conversation chain; Chat Completions from the first
	// follow-up turn via the in-memory last-model store).
	Mode string `yaml:"mode,omitempty"`

	// MinSwitchAdvantage is the minimum net advantage required to allow switching.
	MinSwitchAdvantage float64 `yaml:"min_switch_advantage,omitempty"`

	// DefaultHandoffPenalty is used when lookup_tables has no handoff_penalty entry.
	DefaultHandoffPenalty float64 `yaml:"default_handoff_penalty,omitempty"`

	// CacheWarmthWeight turns cache warmth into a switch penalty.
	CacheWarmthWeight float64 `yaml:"cache_warmth_weight,omitempty"`
}

// MLSelectionConfig holds configuration for the shared ML-based selectors.
type MLSelectionConfig struct {
	ModelsPath   string         `yaml:"models_path,omitempty"`
	EmbeddingDim int            `yaml:"embedding_dim,omitempty"`
	KNN          MLKNNConfig    `yaml:"knn,omitempty"`
	KMeans       MLKMeansConfig `yaml:"kmeans,omitempty"`
	SVM          MLSVMConfig    `yaml:"svm,omitempty"`
	MLP          MLMLPConfig    `yaml:"mlp,omitempty"`
}

type MLKNNConfig struct {
	K              int    `yaml:"k,omitempty"`
	PretrainedPath string `yaml:"pretrained_path,omitempty"`
}

type MLKMeansConfig struct {
	NumClusters      int     `yaml:"num_clusters,omitempty"`
	EfficiencyWeight float64 `yaml:"efficiency_weight,omitempty"`
	PretrainedPath   string  `yaml:"pretrained_path,omitempty"`
}

type MLSVMConfig struct {
	Kernel         string  `yaml:"kernel,omitempty"`
	Gamma          float64 `yaml:"gamma,omitempty"`
	PretrainedPath string  `yaml:"pretrained_path,omitempty"`
}

type MLMLPConfig struct {
	Device         string `yaml:"device,omitempty"`
	PretrainedPath string `yaml:"pretrained_path,omitempty"`
}

type EloSelectionConfig struct {
	InitialRating     float64 `yaml:"initial_rating,omitempty"`
	KFactor           float64 `yaml:"k_factor,omitempty"`
	CategoryWeighted  bool    `yaml:"category_weighted,omitempty"`
	DecayFactor       float64 `yaml:"decay_factor,omitempty"`
	MinComparisons    int     `yaml:"min_comparisons,omitempty"`
	CostScalingFactor float64 `yaml:"cost_scaling_factor,omitempty"`
	StoragePath       string  `yaml:"storage_path,omitempty"`
	AutoSaveInterval  string  `yaml:"auto_save_interval,omitempty"`
}

type RouterDCSelectionConfig struct {
	Temperature         float64 `yaml:"temperature,omitempty"`
	DimensionSize       int     `yaml:"dimension_size,omitempty"`
	MinSimilarity       float64 `yaml:"min_similarity,omitempty"`
	UseQueryContrastive bool    `yaml:"use_query_contrastive"`
	UseModelContrastive bool    `yaml:"use_model_contrastive"`
	RequireDescriptions bool    `yaml:"require_descriptions"`
	UseCapabilities     bool    `yaml:"use_capabilities"`
}

type AutoMixSelectionConfig struct {
	VerificationThreshold  float64 `yaml:"verification_threshold,omitempty"`
	MaxEscalations         int     `yaml:"max_escalations,omitempty"`
	CostAwareRouting       bool    `yaml:"cost_aware_routing,omitempty"`
	CostQualityTradeoff    float64 `yaml:"cost_quality_tradeoff,omitempty"`
	DiscountFactor         float64 `yaml:"discount_factor,omitempty"`
	UseLogprobVerification bool    `yaml:"use_logprob_verification,omitempty"`
}

type HybridSelectionConfig struct {
	ExperienceWeight    float64 `yaml:"experience_weight,omitempty"`
	RouterDCWeight      float64 `yaml:"router_dc_weight,omitempty"`
	AutoMixWeight       float64 `yaml:"automix_weight,omitempty"`
	CostWeight          float64 `yaml:"cost_weight,omitempty"`
	QualityGapThreshold float64 `yaml:"quality_gap_threshold,omitempty"`
	NormalizeScores     bool    `yaml:"normalize_scores"`
}

// SessionAwareSelectionConfig configures the session_aware selector. It wraps
// a base selector and adds agentic session stay-vs-switch policy.
type SessionAwareSelectionConfig struct {
	BaseMethod                   string   `yaml:"base_method,omitempty"`
	IdleTimeoutSeconds           *int     `yaml:"idle_timeout_seconds,omitempty"`
	MinTurnsBeforeSwitch         *int     `yaml:"min_turns_before_switch,omitempty"`
	SwitchMargin                 *float64 `yaml:"switch_margin,omitempty"`
	StayBias                     *float64 `yaml:"stay_bias,omitempty"`
	ToolLoopHardLock             *bool    `yaml:"tool_loop_hard_lock,omitempty"`
	ContextPortabilityHardLock   *bool    `yaml:"context_portability_hard_lock,omitempty"`
	DecisionDriftReset           *bool    `yaml:"decision_drift_reset,omitempty"`
	ToolLoopStayBias             *float64 `yaml:"tool_loop_stay_bias,omitempty"`
	PrefixCacheWeight            *float64 `yaml:"prefix_cache_weight,omitempty"`
	HandoffPenaltyWeight         *float64 `yaml:"handoff_penalty_weight,omitempty"`
	DefaultHandoffPenalty        *float64 `yaml:"default_handoff_penalty,omitempty"`
	QualityGapMultiplier         *float64 `yaml:"quality_gap_multiplier,omitempty"`
	MaxCacheCostMultiplier       *float64 `yaml:"max_cache_cost_multiplier,omitempty"`
	SwitchHistoryWeight          *float64 `yaml:"switch_history_weight,omitempty"`
	RemainingTurnPriorWeight     *float64 `yaml:"remaining_turn_prior_weight,omitempty"`
	RemainingTurnPriorHorizon    *int     `yaml:"remaining_turn_prior_horizon,omitempty"`
	MinRemainingTurnPriorSamples *int     `yaml:"min_remaining_turn_prior_samples,omitempty"`
}

// MultiFactorSelectionConfig configures the multi_factor selector, which
// composes raw quality/latency/cost/load signals into a weighted score per
// candidate model. See issue #37.
type MultiFactorSelectionConfig struct {
	Weights *MultiFactorWeightsConfig `yaml:"weights,omitempty"`
	SLO     *MultiFactorSLOConfig     `yaml:"slo,omitempty"`

	// LatencyPercentile selects which percentile (e.g. 95) is read from
	// pkg/latency when computing the latency signal. Defaults to 95.
	LatencyPercentile int `yaml:"latency_percentile,omitempty"`

	// OnNoCandidates controls behavior when SLO filtering removes every
	// candidate. Valid values: "cheapest" (default), "first", "fail".
	OnNoCandidates string `yaml:"on_no_candidates,omitempty"`
}

// MultiFactorWeightsConfig holds per-signal weights for the multi_factor
// scoring formula score = w_q*quality + w_l*latency + w_c*cost + w_L*load.
// Weights are normalized to sum to 1 if they do not. All four default to 0.25.
type MultiFactorWeightsConfig struct {
	Quality float64 `yaml:"quality,omitempty"`
	Latency float64 `yaml:"latency,omitempty"`
	Cost    float64 `yaml:"cost,omitempty"`
	Load    float64 `yaml:"load,omitempty"`
}

// MultiFactorSLOConfig sets hard ceilings that prune candidates before
// scoring. A zero value means "no ceiling" for that dimension.
type MultiFactorSLOConfig struct {
	MaxTPOTMs    float64 `yaml:"max_tpot_ms,omitempty"`
	MaxTTFTMs    float64 `yaml:"max_ttft_ms,omitempty"`
	MaxCostPer1M float64 `yaml:"max_cost_per_1m,omitempty"`
	MaxInflight  int     `yaml:"max_inflight,omitempty"`
}

// RLDrivenSelectionConfig configures Router-R1 style reinforcement-learning-based routing.
type RLDrivenSelectionConfig struct {
	ExplorationRate             float64 `yaml:"exploration_rate,omitempty"`
	ExplorationDecay            float64 `yaml:"exploration_decay,omitempty"`
	MinExploration              float64 `yaml:"min_exploration,omitempty"`
	UseThompsonSampling         bool    `yaml:"use_thompson_sampling,omitempty"`
	EnablePersonalization       bool    `yaml:"enable_personalization,omitempty"`
	PersonalizationBlend        float64 `yaml:"personalization_blend,omitempty"`
	SessionContextWeight        float64 `yaml:"session_context_weight,omitempty"`
	ImplicitFeedbackWeight      float64 `yaml:"implicit_feedback_weight,omitempty"`
	CostAwareness               bool    `yaml:"cost_awareness,omitempty"`
	CostWeight                  float64 `yaml:"cost_weight,omitempty"`
	StoragePath                 string  `yaml:"storage_path,omitempty"`
	AutoSaveInterval            string  `yaml:"auto_save_interval,omitempty"`
	UseRouterR1Rewards          bool    `yaml:"use_router_r1_rewards,omitempty"`
	CostRewardAlpha             float64 `yaml:"cost_reward_alpha,omitempty"`
	FormatRewardPenalty         float64 `yaml:"format_reward_penalty,omitempty"`
	EnableLLMRouting            bool    `yaml:"enable_llm_routing,omitempty"`
	RouterR1ServerURL           string  `yaml:"router_r1_server_url,omitempty"`
	LLMRoutingFallback          string  `yaml:"llm_routing_fallback,omitempty"`
	EnableMultiRoundAggregation bool    `yaml:"enable_multi_round_aggregation,omitempty"`
	MaxAggregationRounds        int     `yaml:"max_aggregation_rounds,omitempty"`
}

// GMTRouterSelectionConfig configures graph-based personalized routing.
type GMTRouterSelectionConfig struct {
	EnablePersonalization             bool     `yaml:"enable_personalization,omitempty"`
	HistorySampleSize                 int      `yaml:"history_sample_size,omitempty"`
	EmbeddingDimension                int      `yaml:"embedding_dimension,omitempty"`
	NumGNNLayers                      int      `yaml:"num_gnn_layers,omitempty"`
	AttentionHeads                    int      `yaml:"attention_heads,omitempty"`
	MinInteractionsForPersonalization int      `yaml:"min_interactions_for_personalization,omitempty"`
	MaxInteractionsPerUser            int      `yaml:"max_interactions_per_user,omitempty"`
	FeedbackTypes                     []string `yaml:"feedback_types,omitempty"`
	ModelPath                         string   `yaml:"model_path,omitempty"`
	StoragePath                       string   `yaml:"storage_path,omitempty"`
}

// LatencyAwareAlgorithmConfig configures TPOT/TTFT percentile routing policies.
type LatencyAwareAlgorithmConfig struct {
	TPOTPercentile int    `yaml:"tpot_percentile,omitempty"`
	TTFTPercentile int    `yaml:"ttft_percentile,omitempty"`
	Description    string `yaml:"description,omitempty"`
}

// MLModelSelectionConfig configures the ML-based algorithm used from per-decision policies.
type MLModelSelectionConfig struct {
	Type             string             `yaml:"type"`
	ModelsPath       string             `yaml:"models_path,omitempty"`
	K                int                `yaml:"k,omitempty"`
	NumClusters      int                `yaml:"num_clusters,omitempty"`
	Kernel           string             `yaml:"kernel,omitempty"`
	Gamma            float64            `yaml:"gamma,omitempty"`
	EfficiencyWeight *float64           `yaml:"efficiency_weight,omitempty"`
	Device           string             `yaml:"device,omitempty"`
	FeatureWeights   map[string]float64 `yaml:"feature_weights,omitempty"`
}
